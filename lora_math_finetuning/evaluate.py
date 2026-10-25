"""Generate and independently score math-task predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .answer import rescore_jsonl, score_jsonl, score_prediction, summarize_records
from .data import load_dataset_splits, prepare_record
from .model import load_base_or_adapter
from .train import load_config


def generate_predictions(
    config: dict[str, Any],
    adapter_path: str | None,
    split: str,
    output_path: str,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch

    from .model import load_tokenizer

    tokenizer = load_tokenizer(config["model_name"], config.get("model_revision"))
    tokenizer.padding_side = "left"
    model = load_base_or_adapter(
        config["model_name"], adapter_path, config.get("model_revision")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _train_split, validation_split, official_test = load_dataset_splits(
        config["dataset_name"],
        config["dataset_config"],
        config["dataset_revision"],
        float(config["validation_ratio"]),
        int(config["seed"]),
    )
    if split == "test":
        dataset = official_test
    elif split == "validation":
        dataset = validation_split
    else:
        raise ValueError("split must be 'validation' or 'test'")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records = []
    batch_size = int(config["generation_batch_size"])
    if batch_size < 1:
        raise ValueError("generation_batch_size must be at least 1")

    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch_records = [
            prepare_record(
                row["question"],
                row["answer"],
                str(start + offset),
                config["prompt_template"],
                config.get("system_prompt", ""),
            )
            for offset, row in enumerate(rows)
        ]
        conversations = []
        for record in batch_records:
            messages = []
            if record["system_prompt"]:
                messages.append({"role": "system", "content": record["system_prompt"]})
            messages.append({"role": "user", "content": record["prompt"]})
            conversations.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        encoded = tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["max_length"]),
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        for record, output_ids in zip(batch_records, generated[:, prompt_width:]):
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            scored = score_prediction(record["reference_answer"], text)
            scored.update({"question_id": record["question_id"], "question": record["question"]})
            records.append(scored)
        completed = min(start + batch_size, len(dataset))
        if completed == len(dataset) or completed % (batch_size * 25) == 0:
            print(f"Generated {completed}/{len(dataset)} test predictions", flush=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", default="lora_math_finetuning/config.json")
    generate.add_argument("--adapter-path")
    generate.add_argument("--split", default="test")
    generate.add_argument("--output", required=True)
    generate.add_argument("--limit", type=int)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--output")
    score.add_argument("--predictions-output")
    score.add_argument("--analysis-output")

    args = parser.parse_args()
    if args.command == "score":
        if args.predictions_output:
            summary = rescore_jsonl(
                args.predictions, args.predictions_output, args.analysis_output
            )
        else:
            summary = score_jsonl(args.predictions, args.analysis_output)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    generate_predictions(
        load_config(args.config),
        args.adapter_path,
        args.split,
        args.output,
        args.limit,
    )


if __name__ == "__main__":
    main()
