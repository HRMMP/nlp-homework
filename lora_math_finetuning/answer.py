"""Answer extraction and auditable scoring for GSM8K-style outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

SCORING_VERSION = "numeric-v2-leading-decimal"

_FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_FRACTION_RE = re.compile(r"^[+-]?\d+\s*/\s*[+-]?\d+$")


def _boxed_answers(text: str) -> list[str]:
    answers = []
    for match in _BOXED_START_RE.finditer(text):
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[match.end() : index].strip())
                    break
    return answers


def extract_final_answer(text: str) -> str | None:
    """Prefer GSM8K's ``####`` marker, falling back to the last boxed answer."""
    matches = _FINAL_ANSWER_RE.findall(text)
    if matches:
        answer = matches[-1].strip()
        return answer or None
    boxed = _boxed_answers(text)
    return boxed[-1] if boxed else None


def normalize_answer(answer: str | None) -> str | None:
    """Apply conservative normalization before exact comparison.

    Numeric integers, decimals, and fractions are compared mathematically.
    Units and other text are retained, so the normalizer cannot silently turn
    an answer with the wrong unit into a correct one.
    """
    if answer is None:
        return None
    # A leading decimal point is numeric content, not sentence punctuation.
    value = answer.strip().lstrip("`*_$,;:。 \t").rstrip("`*_.$,;:。 \t")
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", value)
    if boxed:
        value = boxed.group(1).strip()
    value = value.replace("$", "").strip()
    value = re.sub(r"\s+", " ", value)
    if _NUMBER_RE.fullmatch(value.replace(",", "")):
        return value.replace(",", "")
    if _FRACTION_RE.fullmatch(value):
        numerator, denominator = re.split(r"\s*/\s*", value)
        denominator_int = int(denominator)
        if denominator_int == 0:
            return value.replace(" ", "")
        return str(Fraction(int(numerator), denominator_int))
    return value.casefold()


def _numeric_value(value: str | None) -> Fraction | None:
    if value is None:
        return None
    if _FRACTION_RE.fullmatch(value):
        numerator, denominator = re.split(r"\s*/\s*", value)
        denominator_int = int(denominator)
        if denominator_int == 0:
            return None
        return Fraction(int(numerator), denominator_int)
    if _NUMBER_RE.fullmatch(value):
        try:
            return Fraction(Decimal(value))
        except InvalidOperation:
            return None
    return None


def answers_equal(reference: str | None, prediction: str | None) -> bool:
    """Compare answers, using exact arithmetic for plain numeric values."""
    reference_normalized = normalize_answer(reference)
    prediction_normalized = normalize_answer(prediction)
    reference_number = _numeric_value(reference_normalized)
    prediction_number = _numeric_value(prediction_normalized)
    if reference_number is not None or prediction_number is not None:
        return reference_number is not None and reference_number == prediction_number
    return reference_normalized == prediction_normalized


def score_prediction(reference_answer: str, generation: str) -> dict[str, Any]:
    """Return one auditable prediction record."""
    extracted = extract_final_answer(generation)
    correct = extracted is not None and answers_equal(reference_answer, extracted)
    if extracted is None:
        reason = "extraction_failed"
    elif correct:
        reason = None
    else:
        reason = "answer_mismatch"
    return {
        "reference_answer": reference_answer,
        "generation": generation,
        "extracted_answer": extracted,
        "format_compliant": extracted is not None,
        "correct": correct,
        "error_reason": reason,
    }


def summarize_records(
    records: Iterable[dict[str, Any]], error_analysis_path: str | Path | None = None
) -> dict[str, Any]:
    records = list(records)
    correct = sum(bool(row.get("correct")) for row in records)
    extraction_failed = sum(
        row.get("error_reason") == "extraction_failed" for row in records
    )
    answer_mismatch = sum(
        row.get("error_reason") == "answer_mismatch" for row in records
    )
    total = len(records)
    extractable = total - extraction_failed
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "format_compliant": extractable,
        "format_compliance_rate": extractable / total if total else 0.0,
        "accuracy_when_extracted": correct / extractable if extractable else 0.0,
        "answer_mismatch": answer_mismatch,
        "extraction_failed": extraction_failed,
    }
    if error_analysis_path is not None:
        destination = Path(error_analysis_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in records:
                if row.get("correct"):
                    continue
                analysis_row = {
                    key: row.get(key)
                    for key in (
                        "question_id",
                        "question",
                        "reference_answer",
                        "generation",
                        "extracted_answer",
                        "error_reason",
                    )
                }
                analysis_row["category"] = (
                    "format_noncompliant"
                    if row.get("error_reason") == "extraction_failed"
                    else "answer_mismatch"
                )
                stream.write(json.dumps(analysis_row, ensure_ascii=False) + "\n")
    return summary


def rescore_jsonl(
    path: str | Path,
    output_path: str | Path,
    error_analysis_path: str | Path | None = None,
) -> dict[str, Any]:
    """Re-score saved generations and write synchronized per-example decisions."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "reference_answer" not in row or "generation" not in row:
                raise ValueError(f"{path}:{line_number}: missing prediction fields")
            records.append(
                {
                    **row,
                    **score_prediction(row["reference_answer"], row["generation"]),
                }
            )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summarize_records(records, error_analysis_path)


def score_jsonl(
    path: str | Path, error_analysis_path: str | Path | None = None
) -> dict[str, Any]:
    """Recompute metrics from a saved prediction JSONL file."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "reference_answer" not in row or "generation" not in row:
                raise ValueError(f"{path}:{line_number}: missing prediction fields")
            records.append(score_prediction(row["reference_answer"], row["generation"]))
    return summarize_records(records, error_analysis_path)
