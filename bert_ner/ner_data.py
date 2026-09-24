"""CLUENER JSONL loading and conversion to character-level BIO labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABEL_TYPES = (
    "address", "book", "company", "game", "government", "movie",
    "name", "organization", "position", "scene",
)
LABELS = ("O",) + tuple(
    f"{prefix}-{entity_type}"
    for entity_type in LABEL_TYPES
    for prefix in ("B", "I")
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def load_jsonl(path: str | Path, require_labels: bool = True) -> list[dict[str, Any]]:
    """Load and validate one CLUENER JSONL split."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: missing string 'text'")
            if require_labels and "label" not in row:
                raise ValueError(f"{path}:{line_number}: labels are required")
            if "label" in row:
                _extract_spans(row["text"], row["label"], path, line_number)
            rows.append(row)
    return rows


def _extract_spans(text: str, labels: dict[str, Any], path: str | Path, line_number: int):
    spans = []
    for entity_type, mentions in labels.items():
        if entity_type not in LABEL_TYPES:
            raise ValueError(f"{path}:{line_number}: unknown entity type {entity_type!r}")
        for entity, ranges in mentions.items():
            for span in ranges:
                if len(span) != 2:
                    raise ValueError(f"{path}:{line_number}: invalid span {span!r}")
                start, end = span
                if start < 0 or end < start or end >= len(text):
                    raise ValueError(f"{path}:{line_number}: out-of-range span {span!r}")
                if text[start : end + 1] != entity:
                    raise ValueError(f"{path}:{line_number}: span {span!r} does not match {entity!r}")
                spans.append((start, end, entity_type))
    return spans


def to_bio(text: str, labels: dict[str, Any]) -> tuple[list[str], list[str], int]:
    """Convert inclusive character offsets to BIO; keep the longest overlap."""
    spans = _extract_spans(text, labels, "<memory>", 1)
    spans.sort(key=lambda item: (-(item[1] - item[0] + 1), item[0], item[2]))
    tags = ["O"] * len(text)
    occupied = [False] * len(text)
    ignored_overlaps = 0
    for start, end, entity_type in spans:
        if any(occupied[start : end + 1]):
            ignored_overlaps += 1
            continue
        tags[start] = f"B-{entity_type}"
        for position in range(start + 1, end + 1):
            tags[position] = f"I-{entity_type}"
        occupied[start : end + 1] = [True] * (end - start + 1)
    return list(text), tags, ignored_overlaps


def align_subword_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """Label only the first subword of each character; ignore other pieces."""
    aligned = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word_id:
            aligned.append(word_labels[word_id])
        else:
            aligned.append(-100)
        previous_word_id = word_id
    return aligned


def encode_example(example: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    chars, tags, _ = to_bio(example["text"], example["label"])
    encoded = tokenizer(chars, is_split_into_words=True, truncation=True, max_length=max_length)
    encoded["labels"] = align_subword_labels(
        encoded.word_ids(), [LABEL_TO_ID[tag] for tag in tags]
    )
    return encoded
