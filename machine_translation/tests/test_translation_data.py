"""Focused tests for the translation data pipeline."""

import sys
import types
import unittest
from unittest.mock import patch

from machine_translation.data import (
    audit_split_overlap,
    collate_translation,
    load_opus_books_splits,
)


class TranslationSplitTests(unittest.TestCase):
    def test_seeded_shuffle_and_splits_are_stable(self):
        rows = [
            {"translation": {"en": f" English {index} ", "fr": f" Francais {index} "}}
            for index in range(20)
        ]
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda name, config: {"train": rows}

        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            first = load_opus_books_splits("opus_books", "en-fr", 41, 12)
            second = load_opus_books_splits("opus_books", ("en", "fr"), 41, 12)

        self.assertEqual(first, second)
        self.assertEqual(
            {key: len(value) for key, value in first.items()},
            {
                "train": 10,
                "validation": 0,
                "test": 2,
            },
        )
        self.assertTrue(
            all(
                set(pair) == {"src", "tgt"}
                for split in first.values()
                for pair in split
            )
        )
        self.assertTrue(
            all(
                pair["src"].startswith("English ")
                for split in first.values()
                for pair in split
            )
        )

    def test_invalid_split_ratios_are_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            load_opus_books_splits("opus_books", "en-fr", 1, None, 0.9, 0.2)

    def test_dataset_revision_is_forwarded_to_loader(self):
        rows = [{"translation": {"en": "hello", "fr": "bonjour"}}]
        fake_datasets = types.ModuleType("datasets")
        calls = []

        def fake_load_dataset(name, config, **kwargs):
            calls.append((name, config, kwargs))
            return {"train": rows}

        fake_datasets.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            load_opus_books_splits(
                "Helsinki-NLP/opus_books", "en-fr", 9, 1, dataset_revision="abc123"
            )
        self.assertEqual(
            calls, [("Helsinki-NLP/opus_books", "en-fr", {"revision": "abc123"})]
        )


class TranslationOverlapTests(unittest.TestCase):
    def test_distinguishes_unique_pairs_and_duplicate_occurrences(self):
        pair = {"src": "hello", "tgt": "bonjour"}
        splits = {
            "train": [pair, pair, {"src": "hello", "tgt": "salut"}],
            "test": [pair, pair, {"src": "other", "tgt": "bonjour"}],
        }
        result = audit_split_overlap(splits)
        self.assertEqual(
            result["within_split_duplicates_beyond_first"]["train"],
            {
                "pair": 1,
                "source": 2,
                "target": 1,
            },
        )
        self.assertEqual(
            result["cross_split"]["train/test"]["pair"],
            {
                "shared_unique_keys": 1,
                "left_occurrences": 2,
                "right_occurrences": 2,
            },
        )
        self.assertEqual(
            result["cross_split"]["train/test"]["source"]["left_occurrences"], 3
        )
        self.assertEqual(
            result["cross_split"]["train/test"]["target"]["right_occurrences"], 3
        )
        self.assertEqual(len(splits["train"]), 3)

    def test_does_not_silently_normalize_text(self):
        result = audit_split_overlap(
            {
                "train": [{"src": "Hello", "tgt": "Bonjour"}],
                "test": [{"src": "hello", "tgt": "Bonjour "}],
            }
        )
        self.assertEqual(
            result["cross_split"]["train/test"]["pair"]["shared_unique_keys"], 0
        )

    def test_empty_split_and_identical_text(self):
        result = audit_split_overlap(
            {
                "train": [{"src": "42", "tgt": "42"}],
                "test": [],
            }
        )
        self.assertEqual(result["identical_source_target"], {"train": 1, "test": 0})
        self.assertEqual(
            result["cross_split"]["train/test"]["pair"]["right_occurrences"], 0
        )


@unittest.skipUnless(
    __import__("importlib").util.find_spec("sentencepiece"),
    "sentencepiece is not installed",
)
class SentencePieceTokenizerTests(unittest.TestCase):
    def test_fixed_special_ids_truncation_and_decode(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from machine_translation.data import train_sentencepiece_tokenizer

        with TemporaryDirectory() as directory:
            tokenizer = train_sentencepiece_tokenizer(
                [
                    {"src": "hello book", "tgt": "bonjour livre"},
                    {"src": "good morning", "tgt": "bonne matin"},
                ],
                Path(directory) / "tokenizer",
                vocab_size=48,
            )
            self.assertIsInstance(tokenizer.vocab_size, int)
            self.assertGreaterEqual(tokenizer.vocab_size, 4)
            self.assertEqual(
                (
                    tokenizer.pad_id,
                    tokenizer.unk_id,
                    tokenizer.bos_id,
                    tokenizer.eos_id,
                ),
                (0, 1, 2, 3),
            )
            self.assertEqual(len(tokenizer.encode_target("bonjour livre", 3)), 3)
            self.assertEqual(len(tokenizer.encode_source("hello book", 2)), 2)
            decoded = tokenizer.decode(tokenizer.encode_target("bonjour livre", 20))
            self.assertIn("bonjour", decoded)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch"), "torch is not installed"
)
class TranslationCollateTests(unittest.TestCase):
    def test_targets_shift_once_and_padding_masks_mark_pad(self):
        import torch

        batch = collate_translation(
            [
                {"src_ids": [8, 3], "target_ids": [2, 9, 3]},
                {"src_ids": [7, 6, 3], "target_ids": [2, 10, 11, 3]},
            ],
            pad_id=0,
        )

        self.assertTrue(
            torch.equal(
                batch["decoder_input_ids"], torch.tensor([[2, 9, 0], [2, 10, 11]])
            )
        )
        self.assertTrue(
            torch.equal(batch["labels"], torch.tensor([[9, 3, 0], [10, 11, 3]]))
        )
        self.assertTrue(
            torch.equal(
                batch["src_key_padding_mask"],
                torch.tensor([[False, False, True], [False, False, False]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["tgt_key_padding_mask"],
                torch.tensor([[False, False, True], [False, False, False]]),
            )
        )
        self.assertEqual(batch["src_key_padding_mask"].dtype, torch.bool)
        self.assertEqual(batch["tgt_key_padding_mask"].dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
