"""Tests for prompt construction and prompt-label masking."""

import unittest

from lora_math_finetuning.data import prepare_record, tokenize_record


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(range(len(text)))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.template = "Question: {question}\nAnswer:\n"

    def test_prompt_does_not_contain_reference_answer(self):
        record = prepare_record(
            "What is 2 + 2?", "2 + 2 = 4\n#### 4", "0", self.template, "Solve carefully."
        )
        self.assertIn("What is 2 + 2?", record["prompt"])
        self.assertNotIn("#### 4", record["prompt"])
        self.assertEqual(record["reference_answer"], "4")
        self.assertEqual(record["system_prompt"], "Solve carefully.")

    def test_only_completion_contributes_to_loss(self):
        record = prepare_record("Q", "work\n#### 4", "0", self.template)
        encoded = tokenize_record(record, TinyTokenizer(), max_length=100)
        prompt_length = encoded["labels"].index(next(v for v in encoded["labels"] if v != -100))
        self.assertGreater(prompt_length, 0)
        self.assertTrue(all(value == -100 for value in encoded["labels"][:prompt_length]))
        self.assertTrue(all(value != -100 for value in encoded["labels"][prompt_length:]))
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))


if __name__ == "__main__":
    unittest.main()
