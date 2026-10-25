"""Tests for paired post-hoc scoring diagnostics, without model dependencies."""

import copy
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lora_math_finetuning.analyze_scoring import (
    analyze_paired,
    extract_last_number,
    main,
    sample_disagreements,
    validate_output_paths,
)


def record(identifier="0", generation="#### 42", answer="42"):
    return {
        "question_id": identifier,
        "question": f"Question {identifier}",
        "reference_answer": answer,
        "generation": generation,
    }


class ScoringSensitivityTests(unittest.TestCase):
    def test_uses_last_numeric_match(self):
        self.assertEqual(extract_last_number("3 + 8 = 11. Answer: 42"), "42")
        self.assertEqual(extract_last_number("The answer is -1,200.50"), "-1,200.50")

    def test_missing_number(self):
        self.assertIsNone(extract_last_number("Cannot determine the answer"))

    def test_regex_limitations_are_not_silently_repaired(self):
        self.assertEqual(extract_last_number("42 ..."), "...")

    def test_diagnostic_can_gain_or_lose_strict_correctness(self):
        baseline = [record(generation="The answer is 42.")]
        lora = [record(generation="#### 42\nSee step 2")]
        original = copy.deepcopy((baseline, lora))
        result = analyze_paired(baseline, lora)
        self.assertEqual(result["models"]["baseline"]["strict"]["correct"], 0)
        self.assertEqual(
            result["models"]["baseline"]["last_number_diagnostic"]["correct"], 1
        )
        self.assertEqual(result["models"]["lora"]["strict"]["correct"], 1)
        self.assertEqual(
            result["models"]["lora"]["last_number_diagnostic"]["correct"], 0
        )
        self.assertEqual(result["paired_decisions"]["strict"]["lora_only"], 1)
        self.assertEqual(
            result["paired_decisions"]["last_number_diagnostic"]["baseline_only"], 1
        )
        self.assertEqual((baseline, lora), original)

    def test_rescores_instead_of_trusting_saved_correct_flag(self):
        baseline = [{**record(), "correct": False}]
        result = analyze_paired(baseline, [record()])
        self.assertEqual(result["models"]["baseline"]["strict"]["correct"], 1)

    def test_rejects_empty_predictions(self):
        with self.assertRaises(ValueError):
            analyze_paired([], [])

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            analyze_paired([record()], [record(), record("1")])

    def test_rejects_duplicate_questions(self):
        with self.assertRaises(ValueError):
            analyze_paired([record(), record()], [record(), record()])

    def test_rejects_misaligned_questions_or_references(self):
        for field, value in (
            ("question_id", "1"),
            ("question", "different"),
            ("reference_answer", "99"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_paired([record()], [{**record(), field: value}])

    def test_rejects_invalid_row_types(self):
        for row in (None, {**record(), "generation": 42}):
            with self.subTest(row=row), self.assertRaises(TypeError):
                analyze_paired([row], [record()])

    def test_stratified_samples_preserve_both_directions(self):
        baseline = [
            record("0", "The answer is 42."),
            record("1", "#### 42\nSee step 2"),
        ]
        lora = [record("0"), record("1")]
        samples = sample_disagreements(baseline, lora, seed=12, per_stratum=1)
        self.assertEqual(
            samples, sample_disagreements(baseline, lora, seed=12, per_stratum=1)
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(
            {row["stratum"] for row in samples},
            {
                "strict_wrong_diagnostic_correct",
                "strict_correct_diagnostic_wrong",
            },
        )
        self.assertTrue(
            all("not_student_verified" in row["review_status"] for row in samples)
        )
        with self.assertRaises(ValueError):
            sample_disagreements(baseline, lora, per_stratum=0)

    def test_output_paths_cannot_alias_inputs_or_each_other(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.touch()
            with self.assertRaises(ValueError):
                validate_output_paths([source], [source])
            output = Path(directory) / "new.json"
            with self.assertRaises(ValueError):
                validate_output_paths([source], [output, output])
            validate_output_paths([source], [output])

    def test_cli_jsonl_keeps_unicode_line_separator_inside_question(self):
        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "base.jsonl"
            lora = Path(directory) / "lora.jsonl"
            row = {**record(), "question": "Question" + chr(0x2028) + "continues"}
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            baseline.write_text(payload, encoding="utf-8")
            lora.write_text(payload, encoding="utf-8")
            stream = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "analyze_scoring",
                        "--baseline",
                        str(baseline),
                        "--lora",
                        str(lora),
                    ],
                ),
                redirect_stdout(stream),
            ):
                main()
            self.assertEqual(
                json.loads(stream.getvalue())["models"]["baseline"]["strict"][
                    "correct"
                ],
                1,
            )
            self.assertEqual(baseline.read_text(encoding="utf-8"), payload)

    def test_rejects_missing_generation(self):
        row = record()
        del row["generation"]
        with self.assertRaises(ValueError):
            analyze_paired([row], [record()])


if __name__ == "__main__":
    unittest.main()
