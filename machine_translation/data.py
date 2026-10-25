"""Data loading and tokenization utilities for English-French translation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any


def _language_codes(language_pair: str | Sequence[str]) -> tuple[str, str]:
    if isinstance(language_pair, str):
        parts = language_pair.split("-")
    else:
        parts = list(language_pair)
    if len(parts) != 2 or not all(isinstance(part, str) and part for part in parts):
        raise ValueError("language_pair must be a pair such as 'en-fr' or ('en', 'fr')")
    return parts[0], parts[1]


def _normalize_row(
    row: Mapping[str, Any], src_lang: str, tgt_lang: str
) -> dict[str, str] | None:
    translation = row.get("translation")
    if not isinstance(translation, Mapping):
        return None
    source = translation.get(src_lang)
    target = translation.get(tgt_lang)
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    source, target = source.strip(), target.strip()
    if not source or not target:
        return None
    return {"src": source, "tgt": target}


def load_opus_books_splits(
    dataset_name: str,
    language_pair: str | Sequence[str],
    seed: int,
    max_examples: int | None,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
    dataset_revision: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Load and deterministically split an OPUS Books language pair.

    ``max_examples`` caps the normalized, shuffled corpus before splitting. The
    remainder after train and validation is returned as ``test``.
    """
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative or None")
    if train_ratio < 0 or validation_ratio < 0 or train_ratio + validation_ratio > 1:
        raise ValueError("split ratios must be non-negative and sum to at most 1")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Loading OPUS Books requires the 'datasets' package") from exc

    src_lang, tgt_lang = _language_codes(language_pair)
    dataset_kwargs = {"revision": dataset_revision} if dataset_revision else {}
    dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", **dataset_kwargs)
    if isinstance(dataset, Mapping):
        if "train" not in dataset:
            raise ValueError("loaded dataset has no 'train' split")
        dataset = dataset["train"]

    pairs = []
    for row in dataset:
        if isinstance(row, Mapping):
            normalized = _normalize_row(row, src_lang, tgt_lang)
            if normalized is not None:
                pairs.append(normalized)

    random.Random(seed).shuffle(pairs)
    if max_examples is not None:
        pairs = pairs[:max_examples]

    count = len(pairs)
    train_end = int(count * train_ratio)
    validation_end = train_end + int(count * validation_ratio)
    return {
        "train": pairs[:train_end],
        "validation": pairs[train_end:validation_end],
        "test": pairs[validation_end:],
    }


def load_translation_splits(
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Load immutable prepared files, or preserve the original OPUS loading path."""
    if not config.get("prepared_data_dir"):
        return load_opus_books_splits(
            config["dataset_name"],
            config["language_pair"],
            config["seed"],
            config["max_examples"],
            config["train_ratio"],
            config["validation_ratio"],
            config["dataset_revision"],
        )
    directory = Path(config["prepared_data_dir"])
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[1] / directory
    manifest_bytes = (directory / "manifest.json").read_bytes()
    expected = config.get("prepared_manifest_sha256")
    if not expected or hashlib.sha256(manifest_bytes).hexdigest() != expected:
        raise ValueError("Prepared data manifest checksum mismatch or missing pin")
    manifest = json.loads(manifest_bytes)
    for field in ("dataset_name", "dataset_revision", "language_pair"):
        if manifest[field] != config[field]:
            raise ValueError(
                f"Prepared data {field} differs from training configuration"
            )
    splits = {}
    for name in ("train", "validation", "test"):
        spec = manifest["files"][name]
        data = (directory / spec["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"Prepared {name} file checksum mismatch")
        rows = [
            json.loads(line)
            for line in data.decode("utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != spec["count"]:
            raise ValueError(f"Prepared {name} row count mismatch")
        if any(
            not isinstance(row.get(side), str) or not row[side].strip()
            for row in rows
            for side in ("src", "tgt")
        ):
            raise ValueError(f"Prepared {name} contains an empty or invalid sentence")
        splits[name] = rows
    cap = config.get("max_examples")
    if cap is not None and cap < sum(map(len, splits.values())):
        sizes = {
            "train": int(cap * config["train_ratio"]),
            "validation": int(cap * config["validation_ratio"]),
        }
        sizes["test"] = cap - sum(sizes.values())
        splits = {name: rows[: sizes[name]] for name, rows in splits.items()}
    return splits


def audit_split_overlap(
    splits: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    """Count exact duplicates without normalizing, filtering, or changing splits."""
    counts = {
        name: {
            "pair": Counter((row["src"], row["tgt"]) for row in rows),
            "source": Counter(row["src"] for row in rows),
            "target": Counter(row["tgt"] for row in rows),
        }
        for name, rows in splits.items()
    }
    overlaps = {}
    for left, right in combinations(counts, 2):
        fields = {}
        for field in ("pair", "source", "target"):
            shared = counts[left][field].keys() & counts[right][field].keys()
            fields[field] = {
                "shared_unique_keys": len(shared),
                "left_occurrences": sum(counts[left][field][key] for key in shared),
                "right_occurrences": sum(counts[right][field][key] for key in shared),
            }
        overlaps[f"{left}/{right}"] = fields
    return {
        "scope": "Exact text only; not language identification or semantic alignment",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "within_split_duplicates_beyond_first": {
            name: {field: sum(c.values()) - len(c) for field, c in fields.items()}
            for name, fields in counts.items()
        },
        "cross_split": overlaps,
        "identical_source_target": {
            name: sum(row["src"] == row["tgt"] for row in rows)
            for name, rows in splits.items()
        },
    }


class SentencePieceTokenizer:
    """Small shared SentencePiece wrapper with fixed translation token IDs."""

    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def __init__(self, model_file: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "SentencePiece tokenization requires 'sentencepiece'"
            ) from exc
        self._processor = spm.SentencePieceProcessor(model_file=str(model_file))

    @property
    def vocab_size(self) -> int:
        return int(self._processor.get_piece_size())

    def _encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(self._processor.encode(text, out_type=int))

    def encode_source(self, text: str, max_length: int) -> list[int]:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        tokens = self._encode(text)[: max_length - 1]
        return tokens + [self.eos_id]

    def encode_target(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("target max_length must be at least 2 for BOS and EOS")
        tokens = self._encode(text)[: max_length - 2]
        return [self.bos_id, *tokens, self.eos_id]

    def decode(self, ids: Iterable[int]) -> str:
        clean_ids = [
            int(token_id)
            for token_id in ids
            if int(token_id) not in (self.pad_id, self.bos_id, self.eos_id)
        ]
        return self._processor.decode(clean_ids)


def train_sentencepiece_tokenizer(
    train_pairs: Iterable[Mapping[str, str]],
    model_prefix: str | Path,
    vocab_size: int = 8000,
    character_coverage: float = 1.0,
) -> SentencePieceTokenizer:
    """Train one shared SentencePiece model using only supplied training pairs."""
    if vocab_size < 4:
        raise ValueError("vocab_size must be at least 4")

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("Training a tokenizer requires 'sentencepiece'") from exc

    sentences = []
    for pair in train_pairs:
        source, target = pair.get("src"), pair.get("tgt")
        if isinstance(source, str) and source.strip():
            sentences.append(source.strip())
        if isinstance(target, str) and target.strip():
            sentences.append(target.strip())
    if not sentences:
        raise ValueError("cannot train SentencePiece on an empty training set")

    prefix = str(model_prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        hard_vocab_limit=False,
        shuffle_input_sentence=False,
        num_threads=1,
        pad_id=SentencePieceTokenizer.pad_id,
        unk_id=SentencePieceTokenizer.unk_id,
        bos_id=SentencePieceTokenizer.bos_id,
        eos_id=SentencePieceTokenizer.eos_id,
    )
    return SentencePieceTokenizer(f"{prefix}.model")


class TranslationDataset:
    """Map normalized sentence pairs to lists of token IDs."""

    def __init__(
        self,
        pairs: Sequence[Mapping[str, str]],
        tokenizer: SentencePieceTokenizer,
        max_length: int,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        pair = self.pairs[index]
        return {
            "src_ids": self.tokenizer.encode_source(pair["src"], self.max_length),
            "target_ids": self.tokenizer.encode_target(pair["tgt"], self.max_length),
        }


def collate_translation(
    items: Sequence[Mapping[str, Sequence[int]]], pad_id: int
) -> dict[str, Any]:
    """Pad a batch and shift BOS/EOS targets once for teacher forcing."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Batch collation requires the 'torch' package") from exc

    sources = [list(item["src_ids"]) for item in items]
    targets = [list(item["target_ids"]) for item in items]
    if any(not source for source in sources) or any(
        len(target) < 2 for target in targets
    ):
        raise ValueError(
            "source sequences must be nonempty and targets must contain BOS/EOS"
        )

    source_width = max(map(len, sources))
    decoder_width = max(len(target) - 1 for target in targets)
    src_ids = torch.full((len(items), source_width), pad_id, dtype=torch.long)
    decoder_input_ids = torch.full(
        (len(items), decoder_width), pad_id, dtype=torch.long
    )
    labels = torch.full((len(items), decoder_width), pad_id, dtype=torch.long)

    for row, (source, target) in enumerate(zip(sources, targets)):
        src_ids[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        decoder_target = target[:-1]
        label_target = target[1:]
        decoder_input_ids[row, : len(decoder_target)] = torch.tensor(
            decoder_target, dtype=torch.long
        )
        labels[row, : len(label_target)] = torch.tensor(label_target, dtype=torch.long)

    return {
        "src_ids": src_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "src_key_padding_mask": src_ids.eq(pad_id),
        "tgt_key_padding_mask": decoder_input_ids.eq(pad_id),
    }
