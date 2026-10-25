"""Model loading and LoRA parameter checks."""

from __future__ import annotations

from typing import Any


def load_tokenizer(model_name: str, model_revision: str | None = None):
    from transformers import AutoTokenizer

    options = {"use_fast": True}
    if model_revision:
        options["revision"] = model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_load_options() -> dict[str, Any]:
    import torch
    from transformers import __version__

    version = tuple(int(part) for part in __version__.split(".")[:2])
    dtype_key = "dtype" if version >= (4, 57) else "torch_dtype"
    if not torch.cuda.is_available():
        return {dtype_key: torch.float32}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {dtype_key: dtype}


def build_lora_model(
    model_name: str, lora_options: dict[str, Any], model_revision: str | None = None
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    options = model_load_options()
    if model_revision:
        options["revision"] = model_revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **options)
    config = LoraConfig(
        r=int(lora_options["r"]),
        lora_alpha=int(lora_options["alpha"]),
        lora_dropout=float(lora_options["dropout"]),
        target_modules=list(lora_options["target_modules"]),
        bias=lora_options.get("bias", "none"),
        task_type=lora_options.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, config)
    stats = parameter_stats(model)
    if stats["trainable_parameters"] == 0:
        raise ValueError("LoRA injection produced zero trainable parameters")
    if stats["trainable_ratio"] >= 1.0:
        raise ValueError("All parameters are trainable; this is not parameter-efficient tuning")
    return model, stats


def parameter_stats(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
        "trainable_parameter_names": trainable_names,
    }


def load_base_or_adapter(
    model_name: str,
    adapter_path: str | None = None,
    model_revision: str | None = None,
):
    from transformers import AutoModelForCausalLM

    options = model_load_options()
    if model_revision:
        options["revision"] = model_revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **options)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    if model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return model
