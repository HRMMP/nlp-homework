"""Recompute SacreBLEU from an exported prediction JSONL file."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("source", "reference", "prediction")


def load_prediction_records(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise TypeError(f"line {line_number} must contain a JSON object")
            missing = [
                field
                for field in REQUIRED_FIELDS
                if not isinstance(record.get(field), str)
            ]
            if missing:
                raise TypeError(
                    f"line {line_number} must have string fields: {', '.join(missing)}"
                )
            records.append({field: record[field] for field in REQUIRED_FIELDS})
    if not records:
        raise ValueError("prediction file contains no records")
    return records


def score_records(records: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise ImportError("BLEU scoring requires the 'sacrebleu' package") from exc

    rows = list(records)
    if not rows:
        raise ValueError("cannot score an empty prediction set")
    if any(
        not isinstance(row.get("prediction"), str)
        or not isinstance(row.get("reference"), str)
        for row in rows
    ):
        raise ValueError("each record needs string 'prediction' and 'reference' fields")

    metric = BLEU(
        lowercase=False,
        tokenize="13a",
        smooth_method="exp",
        effective_order=False,
        max_ngram_order=4,
    )
    result = metric.corpus_score(
        [row["prediction"] for row in rows],
        [[row["reference"] for row in rows]],
    )
    return {
        "bleu": result.score,
        "bleu_1_to_4_precisions": result.precisions,
        "brevity_penalty": result.bp,
        "system_length": result.sys_len,
        "reference_length": result.ref_len,
        "num_examples": len(rows),
        "sacrebleu_signature": metric.get_signature().format(),
        "sacrebleu_configuration": {
            "metric": "corpus BLEU",
            "implementation": "SacreBLEU",
            "reference_count": 1,
            "max_ngram_order": 4,
            "tokenizer": "13a",
            "lowercase": False,
            "smoothing": "exp",
            "effective_order": False,
            "brevity_penalty": "SacreBLEU default",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="metrics JSON path (default: beside predictions)"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_prediction_records(args.predictions)
    metrics = score_records(records)
    output = args.output or args.predictions.with_name(
        args.predictions.stem + ".bleu.json"
    )
    _write_json(output, metrics)
    print(json.dumps({"metrics": metrics, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
