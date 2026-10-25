"""Tests for conservative math answer extraction and scoring."""

import json
import tempfile
import unittest
from pathlib import Path

from lora_math_finetuning.answer import (
    answers_equal,
    extract_final_answer,
    normalize_answer,
    rescore_jsonl,
    score_prediction,
    summarize_records,
)


class AnswerTests(unittest.TestCase):
    def test_extracts_last_explicit_marker(self):
        text = "The intermediate values are 3 and 8.\n#### 42\n#### 43"
        self.assertEqual(extract_final_answer(text), "43")

    def test_boxed_answer_fallback_and_nested_braces(self):
        self.assertEqual(extract_final_answer(r"work \boxed{18}"), "18")
        self.assertEqual(
            extract_final_answer(r"first \boxed{1}, final \boxed{\frac{3}{4}}"),
            r"\frac{3}{4}",
        )
        self.assertEqual(extract_final_answer("#### 12\n\\boxed{13}"), "12")

    def test_missing_marker_is_failure(self):
        self.assertIsNone(extract_final_answer("The answer is 42."))

    def test_numeric_and_fraction_normalization(self):
        self.assertEqual(normalize_answer("1,000"), "1000")
        self.assertTrue(answers_equal("1/2", "2/4"))
        self.assertTrue(answers_equal("1", "1.0"))
        self.assertTrue(answers_equal("18", "`18`"))
        self.assertTrue(answers_equal("3", "**3**"))
        self.assertTrue(answers_equal("\\boxed{42}", "42"))

    def test_leading_decimal_point_is_preserved(self):
        self.assertEqual(normalize_answer(".25"), ".25")
        self.assertTrue(answers_equal("0.25", ".25"))
        self.assertTrue(answers_equal("0.25", "**.25**."))
        self.assertTrue(answers_equal("-0.5", "-.5"))
        self.assertFalse(answers_equal("25", ".25"))
        self.assertFalse(score_prediction("25", "#### .25")["correct"])

    def test_units_are_not_deleted(self):
        self.assertFalse(answers_equal("12 kg", "12"))

    def test_scoring_keeps_extraction_failure(self):
        result = score_prediction("42", "I cannot determine it.")
        self.assertFalse(result["correct"])
        self.assertFalse(result["format_compliant"])
        self.assertEqual(result["error_reason"], "extraction_failed")

    def test_summary_reports_format_and_writes_error_cases(self):
        records = [
            {**score_prediction("42", "#### 42"), "question_id": "1", "question": "Q1"},
            {**score_prediction("42", "#### 41"), "question_id": "2", "question": "Q2"},
            {
                **score_prediction("42", "no marker"),
                "question_id": "3",
                "question": "Q3",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            analysis_path = Path(directory) / "errors.jsonl"
            summary = summarize_records(records, analysis_path)
            errors = [
                json.loads(line)
                for line in analysis_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(summary["format_compliant"], 2)
        self.assertEqual(summary["format_compliance_rate"], 2 / 3)
        self.assertEqual(summary["accuracy_when_extracted"], 1 / 2)
        self.assertEqual(summary["answer_mismatch"], 1)
        self.assertEqual(
            [row["category"] for row in errors],
            ["answer_mismatch", "format_noncompliant"],
        )

    def test_rescore_jsonl_updates_existing_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "reference_answer": "18",
                        "generation": "#### 18`",
                        "correct": False,
                        "error_reason": "answer_mismatch",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = rescore_jsonl(path, path)
            scored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["correct"], 1)
        self.assertTrue(scored["correct"])
        self.assertIsNone(scored["error_reason"])

    def test_summary(self):
        records = [
            score_prediction("42", "#### 42"),
            score_prediction("42", "#### 41"),
            score_prediction("42", "no marker"),
        ]
        self.assertEqual(
            summarize_records(records),
            {
                "total": 3,
                "correct": 1,
                "accuracy": 1 / 3,
                "format_compliant": 2,
                "format_compliance_rate": 2 / 3,
                "accuracy_when_extracted": 1 / 2,
                "answer_mismatch": 1,
                "extraction_failed": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
