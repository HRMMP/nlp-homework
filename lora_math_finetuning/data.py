"""Dataset formatting and causal-LM label construction for the math task."""

from __future__ import annotations

from typing import Any

from .answer import extract_final_answer


def format_prompt(question: str, template: str) -> str:
    """Fill the configured prompt template without exposing the reference answer."""
    return template.format(question=question)


def prepare_record(
    question: str,
    answer: str,
    question_id: str,
    template: str,
    system_prompt: str = "",
) -> dict[str, str]:
    """Convert one raw dataset row to the fields used by training/evaluation."""
    reference = extract_final_answer(answer)
    if reference is None:
        raise ValueError(f"Could not extract a #### answer for {question_id}")
    return {
        "question_id": question_id,
        "question": question,
        "system_prompt": system_prompt,
        "prompt": format_prompt(question, template),
        "completion": answer.strip(),
        "reference_answer": reference,
    }


def tokenize_record(
    record: dict[str, str], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """Tokenize a chat and ignore prompt tokens in the causal-LM loss."""
    messages = []
    if record.get("system_prompt"):
        messages.append({"role": "system", "content": record["system_prompt"]})
    messages.append({"role": "user", "content": record["prompt"]})
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["completion"]}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt tokens are not a prefix of the training example")
    if len(full_ids) > max_length:
        raise ValueError(
            f"Chat-formatted example is longer than max_length={max_length}; "
            "increase max_length or shorten the example"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class CausalLMCollator:
    """Pad input ids, masks, and labels to the longest example in a batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in features]
        masks = [torch.tensor(row["attention_mask"], dtype=torch.long) for row in features]
        labels = [torch.tensor(row["labels"], dtype=torch.long) for row in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


def load_dataset_splits(
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    validation_ratio: float,
    seed: int,
):
    """Load a pinned official train/test split and split train for validation."""
    from datasets import load_dataset

    raw = load_dataset(dataset_name, dataset_config, revision=dataset_revision)
    if "train" not in raw or "test" not in raw:
        raise ValueError("Expected a dataset with official train and test splits")
    split = raw["train"].train_test_split(test_size=validation_ratio, seed=seed)
    return split["train"], split["test"], raw["test"]


def map_records(
    dataset: Any,
    template: str,
    tokenizer: Any,
    max_length: int,
    system_prompt: str = "",
) -> Any:
    """Format and tokenize a Hugging Face Dataset."""
    def convert(row: dict[str, Any], index: int) -> dict[str, Any]:
        record = prepare_record(
            row["question"], row["answer"], str(index), template, system_prompt
        )
        tokenized = tokenize_record(record, tokenizer, max_length)
        return tokenized

    return dataset.map(
        convert,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Formatting and tokenizing math examples",
    )
