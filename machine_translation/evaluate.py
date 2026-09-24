"""Generate translations and export predictions for independent scoring."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from machine_translation.train import _write_json


def _load_run(run_dir: Path, device: Any) -> tuple[dict[str, Any], Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Evaluation requires PyTorch; install machine_translation/requirements.txt"
        ) from exc

    from machine_translation.data import SentencePieceTokenizer
    from machine_translation.model import TransformerSeq2Seq

    config_path = run_dir / "run_config.json"
    checkpoint_path = run_dir / "best.pt"
    tokenizer_path = run_dir / "tokenizer.model"
    for path in (config_path, checkpoint_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(f"required run artifact not found: {path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = TransformerSeq2Seq(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return config, checkpoint, model, SentencePieceTokenizer(tokenizer_path)


def _load_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build has no CUDA device"
        )
    return device


def evaluate_run(args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Evaluation requires PyTorch; install machine_translation/requirements.txt"
        ) from exc

    from machine_translation.data import (
        TranslationDataset,
        collate_translation,
        load_opus_books_splits,
    )
    from machine_translation.model import greedy_decode
    from machine_translation.score import score_records

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.save_examples < 1:
        raise ValueError("--save-examples must be positive")
    run_dir = args.run_dir.resolve()
    device = _load_device(torch, args.device)
    config, checkpoint, model, tokenizer = _load_run(run_dir, device)
    if args.split not in ("validation", "test"):
        raise ValueError(
            "--split must be 'validation' or 'test'; training data is never scored"
        )

    splits = load_opus_books_splits(
        dataset_name=config["dataset_name"],
        language_pair=config["language_pair"],
        seed=config["seed"],
        max_examples=config["max_examples"],
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
        dataset_revision=config["dataset_revision"],
    )
    pairs = splits[args.split]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError(f"the {args.split} split is empty")

    dataset = TranslationDataset(pairs, tokenizer, config["max_length"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_translation, pad_id=tokenizer.pad_id),
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, str]] = []
    offset = 0
    for batch in loader:
        src_ids = batch["src_ids"].to(device)
        generated = (
            greedy_decode(
                model,
                src_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                max_new_tokens=config["max_length"] - 1,
            )
            .cpu()
            .tolist()
        )
        batch_pairs = pairs[offset : offset + len(generated)]
        for pair, output_ids in zip(batch_pairs, generated):
            records.append(
                {
                    "source": pair["src"],
                    "reference": pair["tgt"],
                    "prediction": tokenizer.decode(output_ids),
                }
            )
        offset += len(generated)

    prediction_path = run_dir / f"predictions.{args.split}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metric = score_records(records)
    metric.update(
        {
            "dataset_name": config["dataset_name"],
            "dataset_revision": config["dataset_revision"],
            "language_pair": config["language_pair"],
            "split": args.split,
            "seed": config["seed"],
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_validation_loss": checkpoint["validation_loss"],
            "decoding": {
                "strategy": "greedy",
                "max_new_tokens": config["max_length"] - 1,
                "special_tokens_removed_before_scoring": ["BOS", "EOS", "PAD"],
            },
        }
    )
    metrics_path = run_dir / f"metrics.{args.split}.json"
    _write_json(metrics_path, metric)
    example_count = min(args.save_examples, len(records))
    _write_json(run_dir / f"examples.{args.split}.json", records[:example_count])
    print(
        json.dumps(
            {"metrics": metric, "predictions": str(prediction_path)}, ensure_ascii=False
        )
    )
    return prediction_path, metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N examples in the deterministic split",
    )
    parser.add_argument("--save-examples", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_run(args)


if __name__ == "__main__":
    main()
