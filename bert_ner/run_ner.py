"""Audit CLUENER splits or fine-tune a BERT token classifier."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path


from ner_data import ID_TO_LABEL, LABELS, encode_example, load_jsonl, to_bio
from metrics import compute_entity_metrics

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cluener2020"


def audit_data() -> None:
    for split in ("train", "dev", "test"):
        path = DATA_DIR / f"{split}.json"
        rows = load_jsonl(path, require_labels=split != "test")
        entity_counts = Counter()
        ignored_overlaps = 0
        for row in rows:
            for entity_type, mentions in row.get("label", {}).items():
                entity_counts[entity_type] += sum(len(spans) for spans in mentions.values())
            if "label" in row:
                _, _, ignored = to_bio(row["text"], row["label"])
                ignored_overlaps += ignored
        print(json.dumps({
            "split": split,
            "sentences": len(rows),
            "labeled": "label" in rows[0],
            "entities_by_type": dict(sorted(entity_counts.items())),
            "nested_entities_ignored_by_flat_bio": ignored_overlaps,
        }, ensure_ascii=False))


def train(config_path: Path) -> None:
    import numpy as np
    from datasets import Dataset

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    set_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    train_rows = load_jsonl(DATA_DIR / "train.json")
    dev_rows = load_jsonl(DATA_DIR / "dev.json")
    overlap_counts = {
        "train": sum(to_bio(row["text"], row["label"])[2] for row in train_rows),
        "dev": sum(to_bio(row["text"], row["label"])[2] for row in dev_rows),
    }
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for word_ids-based label alignment")

    def prepare(rows):
        dataset = Dataset.from_list(rows)
        return dataset.map(
            lambda example: encode_example(example, tokenizer, config["max_length"]),
            remove_columns=dataset.column_names,
            desc="Tokenizing and aligning labels",
        )

    train_dataset = prepare(train_rows)
    dev_dataset = prepare(dev_rows)
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    )
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT.parent / output_dir

    def compute_metrics(eval_prediction):
        logits, gold = eval_prediction
        predicted = np.argmax(logits, axis=-1)
        return compute_entity_metrics(predicted, gold, ID_TO_LABEL)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config["seed"],
        data_seed=config["seed"],
        report_to="none",
        logging_strategy="steps",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    prediction_result = trainer.predict(dev_dataset)
    predicted_ids = np.argmax(prediction_result.predictions, axis=-1)
    gold_ids = prediction_result.label_ids
    saved_predictions = {
        "predictions": [
            [int(pred) for pred, gold in zip(pred_row, gold_row, strict=True) if gold != -100]
            for pred_row, gold_row in zip(predicted_ids, gold_ids, strict=True)
        ],
        "labels": [
            [int(gold) for gold in gold_row if gold != -100]
            for gold_row in gold_ids
        ],
    }
    predictions_path = output_dir / "dev_predictions.json"
    predictions_path.write_text(
        json.dumps(saved_predictions, ensure_ascii=False), encoding="utf-8"
    )
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    summary = {
        "model_name": config["model_name"],
        "seed": config["seed"],
        "train_sentences": len(train_rows),
        "dev_sentences": len(dev_rows),
        "train_metrics": train_result.metrics,
        "dev_metrics": eval_metrics,
        "model_output": str(output_dir / "best_model"),
        "dev_predictions": str(predictions_path),
        "nested_entities_ignored_by_flat_bio": overlap_counts,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import torch
    from importlib.metadata import version

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "seqeval": version("seqeval"),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def reevaluate(predictions_path: Path) -> None:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    metrics = compute_entity_metrics(payload["predictions"], payload["labels"], ID_TO_LABEL)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="validate local data and report split statistics")
    train_parser = subparsers.add_parser("train", help="fine-tune BERT and evaluate on dev")
    train_parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    evaluate_parser = subparsers.add_parser("evaluate", help="recompute entity metrics from saved predictions")
    evaluate_parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit_data()
    elif args.command == "evaluate":
        reevaluate(args.predictions)
    else:
        train(args.config)


if __name__ == "__main__":
    main()
