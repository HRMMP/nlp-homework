"""Post-hoc answer-extraction sensitivity on saved, paired GSM8K predictions.

No model is loaded and the original predictions and strict scores are preserved.
The diagnostic is NOT an official lm-evaluation-harness benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .answer import SCORING_VERSION, answers_equal, score_prediction

HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
HARNESS_URL = (
    "https://github.com/EleutherAI/lm-evaluation-harness/blob/"
    f"{HARNESS_REVISION}/lm_eval/tasks/gsm8k/gsm8k.yaml"
)
# The upstream flexible-extract filter takes the last match and nonempty group.
DIAGNOSTIC_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
_NUMBER_RE = re.compile(DIAGNOSTIC_PATTERN)


def extract_last_number(text: str) -> str | None:
    """Use the upstream regex as-is, including its punctuation limitations."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return next((part.strip() for part in matches[-1] if part), None)


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Prediction set must not be empty")
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Each prediction must be a JSON object")
        for field in ("question_id", "question", "reference_answer", "generation"):
            if field not in row:
                raise ValueError(f"Missing field: {field}")
            if not isinstance(row[field], str):
                raise TypeError(f"Non-string field: {field}")
        if row["question_id"] in identifiers:
            raise ValueError(f"Duplicate question_id: {row['question_id']}")
        identifiers.add(row["question_id"])


def _decisions(row: dict[str, Any]) -> dict[str, bool]:
    answer = extract_last_number(row["generation"])
    return {
        "strict": score_prediction(row["reference_answer"], row["generation"])[
            "correct"
        ],
        "last_number_diagnostic": (
            answer is not None and answers_equal(row["reference_answer"], answer)
        ),
    }


def analyze_paired(
    baseline: list[dict[str, Any]],
    lora: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare protocols only after checking paired question identities."""
    _validate_rows(baseline)
    _validate_rows(lora)
    if len(baseline) != len(lora):
        raise ValueError("Baseline and LoRA must contain the same number of questions")
    for left, right in zip(baseline, lora, strict=True):
        if any(
            left[key] != right[key]
            for key in (
                "question_id",
                "question",
                "reference_answer",
            )
        ):
            raise ValueError("Question order, text, or reference answers differ")

    summaries = {}
    decisions = {}
    for name, rows in (("baseline", baseline), ("lora", lora)):
        decisions[name] = [_decisions(row) for row in rows]
        summaries[name] = {
            protocol: {
                "correct": sum(row[protocol] for row in decisions[name]),
                "total": len(rows),
                "accuracy": sum(row[protocol] for row in decisions[name]) / len(rows),
            }
            for protocol in ("strict", "last_number_diagnostic")
        }
        summaries[name]["protocol_changes"] = {
            "strict_wrong_diagnostic_correct": sum(
                not row["strict"] and row["last_number_diagnostic"]
                for row in decisions[name]
            ),
            "strict_correct_diagnostic_wrong": sum(
                row["strict"] and not row["last_number_diagnostic"]
                for row in decisions[name]
            ),
        }

    paired = {}
    for protocol in ("strict", "last_number_diagnostic"):
        counts = {
            "both_correct": 0,
            "baseline_only": 0,
            "lora_only": 0,
            "both_wrong": 0,
        }
        for left, right in zip(decisions["baseline"], decisions["lora"], strict=True):
            key = {
                (True, True): "both_correct",
                (True, False): "baseline_only",
                (False, True): "lora_only",
                (False, False): "both_wrong",
            }[(left[protocol], right[protocol])]
            counts[key] += 1
        paired[protocol] = counts

    return {
        "scope": "Post-hoc parser sensitivity; not official harness scores or causal reasoning assessment",
        "scoring_version": SCORING_VERSION,
        "extraction_reference": {
            "url": HARNESS_URL,
            "revision": HARNESS_REVISION,
            "regex": DIAGNOSTIC_PATTERN,
            "match_selection": "last match, first nonempty capture group",
            "comparison": "Project answers_equal, not harness exact_match",
            "generation": "Existing project chat prompt and greedy outputs; not harness 5-shot generation",
        },
        "models": summaries,
        "paired_decisions": paired,
    }


def validate_output_paths(inputs: list[Path], outputs: list[Path]) -> None:
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Analysis output paths must differ")
    for output in outputs:
        if output.resolve() in {path.resolve() for path in inputs} or (
            output.exists() and any(output.samefile(path) for path in inputs)
        ):
            raise ValueError("Analysis must not overwrite either input prediction")


def sample_disagreements(baseline, lora, seed=42, per_stratum=5):
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    analyze_paired(baseline, lora)
    rng = random.Random(seed)
    samples = []
    for model, rows in (("baseline", baseline), ("lora", lora)):
        strata = {
            "strict_wrong_diagnostic_correct": [],
            "strict_correct_diagnostic_wrong": [],
        }
        for row in rows:
            strict = score_prediction(row["reference_answer"], row["generation"])
            diagnostic = extract_last_number(row["generation"])
            correct = diagnostic is not None and answers_equal(
                row["reference_answer"], diagnostic
            )
            if strict["correct"] == correct:
                continue
            stratum = (
                "strict_correct_diagnostic_wrong"
                if strict["correct"]
                else "strict_wrong_diagnostic_correct"
            )
            strata[stratum].append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "question_id",
                            "question",
                            "reference_answer",
                            "generation",
                        )
                    },
                    "legacy_saved_correct": row.get("correct"),
                    "model": model,
                    "stratum": stratum,
                    "strict_extracted_answer": strict["extracted_answer"],
                    "diagnostic_extracted_answer": diagnostic,
                    "strict_correct": strict["correct"],
                    "diagnostic_correct": correct,
                    "scoring_version": SCORING_VERSION,
                    "review_status": "automatic_stratified_candidate_not_student_verified",
                }
            )
        for stratum in sorted(strata):
            candidates = strata[stratum]
            samples.extend(rng.sample(candidates, min(per_stratum, len(candidates))))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disagreements-output", type=Path)
    parser.add_argument("--sample-per-stratum", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()
    paths = (args.baseline, args.lora)
    try:
        validate_output_paths(
            list(paths),
            [
                path
                for path in (args.output, args.disagreements_output)
                if path is not None
            ],
        )
    except ValueError as error:
        parser.error(str(error))
    payloads = [path.read_bytes() for path in paths]
    rows = [
        [
            json.loads(line)
            for line in payload.decode("utf-8").split("\n")
            if line.strip()
        ]
        for payload in payloads
    ]
    result = analyze_paired(*rows)
    result["source_sha256"] = {
        str(path): hashlib.sha256(payload).hexdigest()
        for path, payload in zip(paths, payloads, strict=True)
    }
    if args.disagreements_output is not None:
        samples = sample_disagreements(*rows, args.sample_seed, args.sample_per_stratum)
        args.disagreements_output.parent.mkdir(parents=True, exist_ok=True)
        args.disagreements_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
            encoding="utf-8",
        )
        result["disagreement_sample"] = {
            "file": str(args.disagreements_output),
            "seed": args.sample_seed,
            "per_stratum_limit": args.sample_per_stratum,
            "count": len(samples),
            "review_status": "automatic_candidates_not_student_verified",
        }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
