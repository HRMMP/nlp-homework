"""Tests for independent SacreBLEU input and scoring."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from machine_translation.score import load_prediction_records, score_records


class PredictionFileTests(unittest.TestCase):
    def test_loads_structured_jsonl_records(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {"source": "hello", "reference": "bonjour", "prediction": "salut"}
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_prediction_records(path),
                [{"source": "hello", "reference": "bonjour", "prediction": "salut"}],
            )

    def test_rejects_records_missing_string_fields(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"source":"hello","prediction":3}\n', encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "reference"):
                load_prediction_records(path)

    def test_rejects_empty_prediction_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no records"):
                load_prediction_records(path)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("sacrebleu"), "sacrebleu is not installed"
)
class SacreBleuTests(unittest.TestCase):
    def test_exact_corpus_records_score_and_record_configuration(self):
        records = [
            {
                "source": "a source sentence",
                "reference": "the cat is sitting on the mat",
                "prediction": "the cat is sitting on the mat",
            },
            {
                "source": "another source",
                "reference": "we like machine translation",
                "prediction": "we like machine translation",
            },
        ]
        metrics = score_records(records)
        self.assertAlmostEqual(metrics["bleu"], 100.0)
        self.assertEqual(metrics["num_examples"], 2)
        self.assertEqual(metrics["bleu_1_to_4_precisions"], [100.0] * 4)
        self.assertIn("tok:13a", metrics["sacrebleu_signature"])
        self.assertEqual(metrics["sacrebleu_configuration"]["reference_count"], 1)


if __name__ == "__main__":
    unittest.main()
