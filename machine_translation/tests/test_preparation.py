"""Focused tests for predeclared translation cleaning and immutable splits."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from machine_translation.data import audit_split_overlap, load_translation_splits
from machine_translation.prepare import (
    filter_rows,
    grouped_split,
    normalize_text,
    rejection_reason,
)

POLICY = {
    "min_alpha_chars": 10,
    "max_chars": 500,
    "max_length_ratio": 4.0,
    "min_language_probability": 0.8,
}


def pair(
    source="This is an English sentence.",
    target="Ceci est une phrase francaise.",
    **extra,
):
    return {
        "src": source,
        "tgt": target,
        "src_language": "en",
        "tgt_language": "fr",
        "src_probability": 0.95,
        "tgt_probability": 0.95,
        **extra,
    }


def test_normalization_is_explicit_and_stable():
    assert normalize_text("  hello\n\tworld  ") == "hello world"
    assert normalize_text(normalize_text("hello\nworld")) == "hello world"


def test_filter_rejects_reversed_and_uncertain_without_swapping():
    assert (
        rejection_reason(pair(src_language="fr", tgt_language="en"), POLICY)
        == "reversed_language"
    )
    assert (
        rejection_reason(pair(src_probability=0.7), POLICY) == "low_language_confidence"
    )
    assert rejection_reason(pair(src_language="de"), POLICY) == "unexpected_language"
    assert rejection_reason(pair(source="Yes", target="Oui"), POLICY) == "too_short"


def test_filter_deduplicates_after_normalization_without_mutation():
    rows = [pair(), pair(source="This is an  English sentence.")]
    kept, rejected = filter_rows(rows, POLICY)
    assert len(kept) == len(rejected) == 1
    assert rejected[0]["reason"] == "duplicate_pair"
    assert "  " in rows[1]["src"]


def test_connected_group_split_blocks_transitive_leakage():
    rows = [pair(source=f"source {i}", target=f"target {i}") for i in range(30)]
    rows += [
        pair(source="source 0", target="target 1"),
        pair(source="source 1", target="other"),
    ]
    splits = grouped_split(rows, 42, 0.8, 0.1)
    assert splits == grouped_split(rows, 42, 0.8, 0.1)
    assert sum(map(len, splits.values())) == len(rows)
    audit = audit_split_overlap(splits)
    assert all(
        field["shared_unique_keys"] == 0
        for side in audit["cross_split"].values()
        for field in side.values()
    )


def test_grouped_split_refuses_empty_splits():
    with pytest.raises(ValueError):
        grouped_split([pair(), pair()], 42, 0.9, 0.05)


def make_prepared(directory):
    manifest = {
        "dataset_name": "example",
        "dataset_revision": "abc",
        "language_pair": "en-fr",
        "files": {},
    }
    for name in ("train", "validation", "test"):
        data = (json.dumps(pair()) + "\n").encode()
        (directory / f"{name}.jsonl").write_bytes(data)
        manifest["files"][name] = {
            "file": f"{name}.jsonl",
            "sha256": hashlib.sha256(data).hexdigest(),
            "count": 1,
        }
    payload = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(payload)
    return {
        "dataset_name": "example",
        "dataset_revision": "abc",
        "language_pair": "en-fr",
        "prepared_data_dir": str(directory),
        "prepared_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "max_examples": 20,
        "train_ratio": 0.9,
        "validation_ratio": 0.05,
    }


def test_prepared_loader_validates_files():
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        config = make_prepared(directory)
        assert len(load_translation_splits(config)["test"]) == 1
        (directory / "test.jsonl").write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum"):
            load_translation_splits(config)


def test_prepared_loader_requires_pinned_matching_manifest():
    with TemporaryDirectory() as tmp:
        config = make_prepared(Path(tmp))
        with pytest.raises(ValueError, match="checksum"):
            load_translation_splits({**config, "prepared_manifest_sha256": None})
        with pytest.raises(ValueError, match="dataset_revision"):
            load_translation_splits({**config, "dataset_revision": "other"})
