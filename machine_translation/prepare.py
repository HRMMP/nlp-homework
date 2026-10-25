"""Prepare a language-filtered OPUS Books version before any model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .data import audit_split_overlap


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def rejection_reason(row: dict[str, Any], policy: dict[str, Any]) -> str | None:
    source, target = row["src"], row["tgt"]
    if not source or not target:
        return "empty"
    if (
        min(sum(c.isalpha() for c in text) for text in (source, target))
        < policy["min_alpha_chars"]
    ):
        return "too_short"
    if max(len(source), len(target)) > policy["max_chars"]:
        return "too_long"
    if source == target:
        return "identical"
    if (
        max(len(source), len(target)) / min(len(source), len(target))
        > policy["max_length_ratio"]
    ):
        return "length_ratio"
    if (row["src_language"], row["tgt_language"]) == ("fr", "en"):
        return "reversed_language"
    if (row["src_language"], row["tgt_language"]) != ("en", "fr"):
        return "unexpected_language"
    if (
        min(row["src_probability"], row["tgt_probability"])
        < policy["min_language_probability"]
    ):
        return "low_language_confidence"
    return None


def grouped_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Keep connected same-source OR same-target rows in one split."""
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("All three split ratios must be positive")
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for side in ("src", "tgt"):
            key = (side, row[side])
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    result = {"train": [], "validation": [], "test": []}
    train_end = int(len(rows) * train_ratio)
    validation_end = train_end + int(len(rows) * validation_ratio)
    count = 0
    for group in groups:
        name = (
            "train"
            if count < train_end
            else "validation"
            if count < validation_end
            else "test"
        )
        result[name].extend(group)
        count += len(group)
    if any(not split for split in result.values()):
        raise ValueError("Grouped data is too small for three nonempty splits")
    return result


def filter_rows(rows: list[dict[str, Any]], policy: dict[str, Any]):
    kept, rejected = [], []
    seen = set()
    for raw in rows:
        row = {
            **raw,
            "src": normalize_text(raw["src"]),
            "tgt": normalize_text(raw["tgt"]),
        }
        reason = rejection_reason(row, policy)
        key = (row["src"], row["tgt"])
        if reason is None and key in seen:
            reason = "duplicate_pair"
        if reason is None:
            seen.add(key)
            kept.append(row)
        else:
            rejected.append({**row, "reason": reason})
    return kept, rejected


def prepare(
    config_path: Path, output_dir: Path, training_config_output: Path | None
) -> dict[str, Any]:
    import fasttext
    from datasets import load_dataset

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = Path(config["language_model_path"])
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != config["language_model_sha256"]:
        raise ValueError("Language identification model checksum mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Prepared output directory is not empty: {output_dir}")
    if training_config_output is not None and training_config_output.exists():
        raise FileExistsError(
            f"Training config already exists: {training_config_output}"
        )
    dataset = load_dataset(
        config["dataset_name"],
        "en-fr",
        revision=config["dataset_revision"],
        split="train",
    )
    model = fasttext.load_model(str(model_path))
    rows = []
    batch_size = int(config["language_batch_size"])
    if batch_size < 1:
        raise ValueError("language_batch_size must be positive")
    for start in range(0, len(dataset), batch_size):
        raw = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        texts = [
            normalize_text(row["translation"][side])
            for row in raw
            for side in ("en", "fr")
        ]
        # List prediction uses the supported native batch path with NumPy 2.
        labels, probabilities = model.predict(texts, k=1)
        for offset, record in enumerate(raw):
            source, target = 2 * offset, 2 * offset + 1
            rows.append(
                {
                    "row_id": str(record.get("id", start + offset)),
                    "src": texts[source],
                    "tgt": texts[target],
                    "src_language": labels[source][0].removeprefix("__label__"),
                    "tgt_language": labels[target][0].removeprefix("__label__"),
                    "src_probability": float(probabilities[source][0]),
                    "tgt_probability": float(probabilities[target][0]),
                }
            )
    kept, rejected = filter_rows(rows, config["policy"])
    splits = grouped_split(
        kept, config["split_seed"], config["train_ratio"], config["validation_ratio"]
    )
    overlap = audit_split_overlap(splits)
    if any(
        values["shared_unique_keys"]
        for fields in overlap["cross_split"].values()
        for values in fields.values()
    ):
        raise RuntimeError("Grouped splitting left exact cross-split overlap")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, records in {**splits, "rejected": rejected}.items():
        data = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in records
        ).encode("utf-8")
        destination = output_dir / f"{name}.jsonl"
        destination.write_bytes(data)
        files[name] = {
            "file": destination.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "count": len(records),
        }
    by_reason = defaultdict(list)
    for row in rejected:
        by_reason[row["reason"]].append(row)
    rng = random.Random(config["review_seed"])
    review = [
        {**row, "selection": "kept_random"}
        for row in rng.sample(kept, min(100, len(kept)))
    ]
    for reason, records in sorted(by_reason.items()):
        review.extend(
            {**row, "selection": f"rejected_{reason}"}
            for row in rng.sample(records, min(10, len(records)))
        )
    (output_dir / "review_sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review),
        encoding="utf-8",
    )
    manifest = {
        "version": "opus-books-en-fr-lid-grouped-v1",
        "dataset_name": config["dataset_name"],
        "dataset_revision": config["dataset_revision"],
        "language_pair": "en-fr",
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejections": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "config": config,
        "language_model_sha256": model_hash,
        "language_model_source": "https://fasttext.cc/docs/en/language-identification.html",
        "language_model_license": "CC-BY-SA-3.0",
        "versions": {
            name: version(name) for name in ("datasets", "fasttext-wheel", "numpy")
        },
        "source_dataset_fingerprint": dataset._fingerprint,
        "split_policy": "NFKC/whitespace normalization; filter; exact pair dedup; shuffle connected same-source OR same-target groups",
        "files": files,
        "overlap": overlap,
        "limitations": "Language confidence is a screening heuristic, not semantic alignment; no book metadata or blind human audit",
        "review_sample": "review_sample.jsonl; automatic sampling, not student-reviewed gold",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if training_config_output is not None:
        training = json.loads(
            Path(config["base_training_config"]).read_text(encoding="utf-8")
        )
        training.update(
            {
                "prepared_data_dir": str(output_dir),
                "prepared_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "max_examples": len(kept),
            }
        )
        training_config_output.parent.mkdir(parents=True, exist_ok=True)
        training_config_output.write_text(
            json.dumps(training, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "input": len(rows),
                "kept": len(kept),
                "rejections": manifest["rejections"],
                "split_sizes": overlap["split_sizes"],
            },
            ensure_ascii=False,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-config-output", type=Path)
    args = parser.parse_args()
    prepare(args.config, args.output_dir, args.training_config_output)


if __name__ == "__main__":
    main()
