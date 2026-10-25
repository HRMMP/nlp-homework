"""Small data audit for the selected math dataset."""

from __future__ import annotations

import argparse
import json

from .answer import extract_final_answer


def audit_dataset(
    dataset_name: str, dataset_config: str, dataset_revision: str
) -> dict[str, object]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, revision=dataset_revision)
    result: dict[str, object] = {
        "dataset_name": dataset_name,
        "config": dataset_config,
        "revision": dataset_revision,
        "splits": {},
    }
    for split_name, rows in dataset.items():
        missing = 0
        for row in rows:
            if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
                raise TypeError(f"{split_name} contains a row without question/answer strings")
            if extract_final_answer(row["answer"]) is None:
                missing += 1
        result["splits"][split_name] = {"rows": len(rows), "missing_final_answer": missing}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument(
        "--dataset-revision", default="740312add88f781978c0658806c59bc2815b9866"
    )
    args = parser.parse_args()
    print(json.dumps(
        audit_dataset(args.dataset_name, args.dataset_config, args.dataset_revision), indent=2
    ))


if __name__ == "__main__":
    main()
