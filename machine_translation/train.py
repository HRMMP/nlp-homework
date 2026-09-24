"""Train a small Transformer encoder-decoder on OPUS Books."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "dataset_name",
        "dataset_revision",
        "language_pair",
        "seed",
        "max_examples",
        "train_ratio",
        "validation_ratio",
        "max_length",
        "vocab_size",
        "batch_size",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
        "learning_rate",
        "weight_decay",
        "epochs",
        "gradient_clip_norm",
        "early_stopping_patience",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"configuration is missing fields: {', '.join(missing)}")
    if config["d_model"] % config["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")
    if config["max_examples"] < 20:
        raise ValueError("max_examples must be at least 20 for non-empty splits")
    return config


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build has no CUDA device"
        )
    return device


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _run_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    pad_id: int,
    vocab_size: int,
    gradient_clip_norm: float | None = None,
) -> float:
    import torch
    from torch.nn import functional

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(src_ids, decoder_input_ids)
            loss = functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=pad_id,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite cross-entropy loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()
            token_count = int(labels.ne(pad_id).sum().item())
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("epoch contains no non-padding target tokens")
    return total_loss / total_tokens


def _make_smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = dict(config)
    smoke.update(
        {
            "max_examples": min(config["max_examples"], 64),
            "max_length": min(config["max_length"], 32),
            "vocab_size": min(config["vocab_size"], 256),
            "batch_size": 8,
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    return smoke


def run_training(args: argparse.Namespace) -> Path:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Training requires PyTorch; install machine_translation/requirements.txt"
        ) from exc

    from functools import partial

    from machine_translation.data import (
        TranslationDataset,
        collate_translation,
        load_opus_books_splits,
        train_sentencepiece_tokenizer,
    )
    from machine_translation.model import TransformerSeq2Seq

    config = _load_config(args.config)
    if args.smoke:
        config = _make_smoke_config(config)
    if args.max_examples is not None:
        config["max_examples"] = args.max_examples
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    seed = int(config["seed"])
    _seed_everything(seed, torch)
    device = _resolve_device(torch, args.device)

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_kind = "smoke" if args.smoke else "run"
        output_dir = Path(__file__).parent / "outputs" / f"en-fr-{run_kind}-{stamp}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_opus_books_splits(
        dataset_name=config["dataset_name"],
        language_pair=config["language_pair"],
        seed=seed,
        max_examples=config["max_examples"],
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
        dataset_revision=config["dataset_revision"],
    )
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError(
            "dataset split is empty; increase max_examples or adjust split ratios"
        )
    _write_json(output_dir / "run_config.json", config)
    _write_json(
        output_dir / "data_manifest.json",
        {
            "dataset_name": config["dataset_name"],
            "dataset_revision": config["dataset_revision"],
            "dataset_card_url": "https://huggingface.co/datasets/Helsinki-NLP/opus_books",
            "original_source": "https://opus.nlpl.eu/",
            "license_metadata": "other (as stated on the pinned Hugging Face dataset card)",
            "language_pair": config["language_pair"],
            "seed": seed,
            "max_examples_after_shuffle": config["max_examples"],
            "split_ratios": {
                "train": config["train_ratio"],
                "validation": config["validation_ratio"],
                "test": 1.0 - config["train_ratio"] - config["validation_ratio"],
            },
            "split_sizes": {name: len(rows) for name, rows in splits.items()},
            "split_order": "seeded shuffle, then contiguous train/validation/test slices",
        },
    )

    tokenizer = train_sentencepiece_tokenizer(
        splits["train"], output_dir / "tokenizer", vocab_size=config["vocab_size"]
    )
    _write_json(
        output_dir / "runtime.json",
        {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "datasets": _package_version("datasets"),
            "sentencepiece": _package_version("sentencepiece"),
            "sacrebleu": _package_version("sacrebleu"),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "seed": seed,
        },
    )

    collate = partial(collate_translation, pad_id=tokenizer.pad_id)
    train_data = TranslationDataset(splits["train"], tokenizer, config["max_length"])
    validation_data = TranslationDataset(
        splits["validation"], tokenizer, config["max_length"]
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = TransformerSeq2Seq(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_length=config["max_length"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    _write_json(
        output_dir / "model_summary.json",
        {
            "architecture": "torch.nn.Transformer encoder-decoder",
            "batch_first": True,
            "positional_encoding": "fixed sinusoidal",
            "shared_source_target_sentencepiece_vocabulary": True,
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": {
                "pad": tokenizer.pad_id,
                "unk": tokenizer.unk_id,
                "bos": tokenizer.bos_id,
                "eos": tokenizer.eos_id,
            },
            "parameters_total": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameters_trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history: list[dict[str, float | int]] = []
    log_path = output_dir / "training.jsonl"
    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.monotonic()
        epoch_train_loader = train_loader
        if args.max_steps is not None:
            from itertools import islice

            remaining_steps = args.max_steps - global_step
            epoch_train_loader = islice(train_loader, remaining_steps)
            epoch_steps = min(remaining_steps, len(train_loader))
        else:
            epoch_steps = len(train_loader)
        train_loss = _run_epoch(
            model,
            epoch_train_loader,
            optimizer,
            device,
            tokenizer.pad_id,
            tokenizer.vocab_size,
            gradient_clip_norm=config["gradient_clip_norm"],
        )
        global_step += epoch_steps
        validation_loss = _run_epoch(
            model,
            validation_loader,
            None,
            device,
            tokenizer.pad_id,
            tokenizer.vocab_size,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "seconds": time.monotonic() - started,
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _write_json(output_dir / "history.json", history)
        print(json.dumps(record), flush=True)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model_config": {
                        "src_vocab_size": tokenizer.vocab_size,
                        "tgt_vocab_size": tokenizer.vocab_size,
                        "pad_id": tokenizer.pad_id,
                        "d_model": config["d_model"],
                        "nhead": config["nhead"],
                        "num_encoder_layers": config["num_encoder_layers"],
                        "num_decoder_layers": config["num_decoder_layers"],
                        "dim_feedforward": config["dim_feedforward"],
                        "dropout": config["dropout"],
                        "max_length": config["max_length"],
                    },
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if args.max_steps is not None and global_step >= args.max_steps:
            break
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break

    if not (output_dir / "best.pt").is_file():
        raise RuntimeError("training finished without producing a best checkpoint")
    print(f"best_checkpoint={output_dir / 'best.pt'}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0"
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--max-steps", type=int, help="stop after this many optimizer batches"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="use a tiny model and at most 64 pairs"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.config.is_absolute():
        args.config = (Path.cwd() / args.config).resolve()
    run_dir = run_training(args)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
