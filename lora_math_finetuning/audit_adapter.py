"""Audit saved LoRA tensors and recorded parameter names without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from safetensors import safe_open

_NAME_RE = re.compile(
    r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight"
)


def check_tensors(shapes, parameter_stats, adapter_config, model_config):
    recorded = {
        name.replace(".default", "")
        for name in parameter_stats["trainable_parameter_names"]
    }
    if recorded != set(shapes):
        raise ValueError(
            "Saved tensor names differ from recorded trainable parameter names"
        )
    if set(adapter_config["target_modules"]) != {"q_proj", "v_proj"}:
        raise ValueError("Unexpected target modules")
    rank = int(adapter_config["r"])
    expected = set()
    for layer in range(model_config["num_hidden_layers"]):
        for module in ("q_proj", "v_proj"):
            for factor in ("A", "B"):
                expected.add((layer, module, factor))
    actual = set()
    for name, shape in shapes.items():
        match = _NAME_RE.fullmatch(name)
        if match is None or len(shape) != 2:
            raise ValueError(f"Unexpected adapter tensor: {name}")
        layer, module, factor = int(match[1]), match[2], match[3]
        actual.add((layer, module, factor))
        hidden = model_config["hidden_size"]
        output = (
            hidden
            if module == "q_proj"
            else model_config["num_key_value_heads"]
            * (hidden // model_config["num_attention_heads"])
        )
        expected_shape = [rank, hidden] if factor == "A" else [output, rank]
        if list(shape) != expected_shape:
            raise ValueError(f"Unexpected shape for {name}: {shape}")
    if actual != expected:
        raise ValueError("Adapter layer/module coverage is incomplete")
    parameters = sum(math.prod(shape) for shape in shapes.values())
    if parameters != parameter_stats["trainable_parameters"]:
        raise ValueError(
            "Saved trainable parameter count differs from recorded statistics"
        )
    return {
        "tensor_count": len(shapes),
        "trainable_parameters": parameters,
        "total_parameters_recorded": parameter_stats["total_parameters"],
        "trainable_ratio": parameters / parameter_stats["total_parameters"],
        "layers": model_config["num_hidden_layers"],
        "rank": rank,
        "target_modules": sorted(adapter_config["target_modules"]),
    }


def audit(run_dir: Path, output: Path):
    from transformers import AutoConfig

    paths = {
        "tensors": run_dir / "final_adapter/adapter_model.safetensors",
        "adapter_config": run_dir / "final_adapter/adapter_config.json",
        "parameter_stats": run_dir / "parameter_stats.json",
        "run_config": run_dir / "config.json",
    }
    if output.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("Audit output must not overwrite an input")
    config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
    model = AutoConfig.from_pretrained(
        config["model_name"], revision=config["model_revision"], local_files_only=True
    )
    if model._commit_hash != config["model_revision"]:
        raise ValueError("Local model configuration revision mismatch")
    with safe_open(paths["tensors"], framework="numpy") as reader:
        tensor_names = reader.keys()
        shapes = {name: reader.get_slice(name).get_shape() for name in tensor_names}
    stats = json.loads(paths["parameter_stats"].read_text(encoding="utf-8"))
    adapter_config = json.loads(paths["adapter_config"].read_text(encoding="utf-8"))
    result = check_tensors(shapes, stats, adapter_config, model.to_dict())
    result.update(
        {
            "scope": "Saved tensor/config audit, not independent proof of historical optimizer behavior or freezing",
            "model_revision": config["model_revision"],
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths.values()
            },
            "shapes": shapes,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"shapes", "source_sha256"}
            }
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("lora_math_finetuning/outputs/qwen2.5-1.5b-gsm8k"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.output)


if __name__ == "__main__":
    main()
