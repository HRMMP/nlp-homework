"""Train a causal language model with a PEFT LoRA adapter."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
from pathlib import Path
from typing import Any

from .data import CausalLMCollator, load_dataset_splits, map_records
from .model import build_lora_model, load_tokenizer


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def environment_info() -> dict[str, str]:
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda or "unavailable",
    }
    for package in ("datasets", "peft", "accelerate"):
        try:
            module = __import__(package)
            info[package] = module.__version__
        except (ImportError, AttributeError):
            info[package] = "unknown"
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda or "unknown"
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["cuda"] = "unavailable"
        info["gpu"] = "cpu"
    return info


def limit_dataset(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def make_training_args(config: dict[str, Any], output_dir: str, max_steps: int | None):
    import torch
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config["train_batch_size"]),
        "per_device_eval_batch_size": int(config["eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["epochs"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "logging_steps": int(config["logging_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": int(config["seed"]),
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        kwargs["bf16"] = True
    elif torch.cuda.is_available():
        kwargs["fp16"] = True

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def train(config: dict[str, Any], max_train_samples: int | None = None,
          max_eval_samples: int | None = None, max_steps: int | None = None) -> Path:
    from transformers import Trainer, set_seed

    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "environment.json", environment_info())

    tokenizer = load_tokenizer(config["model_name"], config.get("model_revision"))
    train_raw, validation_raw, _test_raw = load_dataset_splits(
        config["dataset_name"],
        config["dataset_config"],
        config["dataset_revision"],
        float(config["validation_ratio"]),
        int(config["seed"]),
    )
    source_train_rows = len(train_raw)
    source_validation_rows = len(validation_raw)
    official_test_rows = len(_test_raw)
    train_raw = limit_dataset(train_raw, max_train_samples)
    validation_raw = limit_dataset(validation_raw, max_eval_samples)
    save_json(output_dir / "data_manifest.json", {
        "dataset_name": config["dataset_name"],
        "dataset_config": config["dataset_config"],
        "dataset_revision": config["dataset_revision"],
        "seed": int(config["seed"]),
        "validation_ratio": float(config["validation_ratio"]),
        "source_train_rows": source_train_rows,
        "source_validation_rows": source_validation_rows,
        "official_test_rows": official_test_rows,
        "train_rows_used": len(train_raw),
        "validation_rows_used": len(validation_raw),
    })
    train_dataset = map_records(
        train_raw,
        config["prompt_template"],
        tokenizer,
        int(config["max_length"]),
        config.get("system_prompt", ""),
    )
    validation_dataset = map_records(
        validation_raw,
        config["prompt_template"],
        tokenizer,
        int(config["max_length"]),
        config.get("system_prompt", ""),
    )

    model, stats = build_lora_model(
        config["model_name"], config["lora"], config.get("model_revision")
    )
    model.config.use_cache = False
    save_json(output_dir / "parameter_stats.json", stats)
    print(json.dumps({key: value for key, value in stats.items() if key != "trainable_parameter_names"}, indent=2))

    trainer = Trainer(
        model=model,
        args=make_training_args(config, str(output_dir), max_steps),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    save_json(adapter_dir / "parameter_stats.json", stats)
    print(f"Saved LoRA adapter to {adapter_dir}")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lora_math_finetuning/config.json")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    train(
        load_config(args.config),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
