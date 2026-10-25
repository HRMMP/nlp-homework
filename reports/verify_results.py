"""Recompute report tables from saved predictions; no model training or generation.

Run from the repository root with the existing experiment dependencies:
    python reports/verify_results.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bert_ner"))


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    from ner_data import ID_TO_LABEL
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2

    from lora_math_finetuning.answer import (
        SCORING_VERSION,
        score_jsonl,
        score_prediction,
    )
    from machine_translation.score import load_prediction_records, score_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-mt-run",
        type=Path,
        help="Also verify a completed prepared-data translation run",
    )
    parser.add_argument(
        "--audit-cached-mt",
        action="store_true",
        help="Check saved MT pairs against the pinned local cache; offline only",
    )
    args = parser.parse_args()
    mt_dir = "machine_translation/outputs/en-fr-full-seed42"
    ner_dir = "bert_ner/outputs/bert-base-chinese"
    lora_dir = "lora_math_finetuning/outputs"
    sources = [
        f"{mt_dir}/predictions.test.jsonl",
        f"{mt_dir}/history.json",
        f"{ner_dir}/dev_predictions.json",
        f"{ner_dir}/checkpoint-2016/trainer_state.json",
    ]
    mt = score_records(load_prediction_records(ROOT / sources[0]))
    assert mt == load(f"{mt_dir}/metrics.test.independent.json")
    payload = load(sources[2])
    assert len(payload["labels"]) == len(payload["predictions"]) == 1343
    gold, pred = [], []
    for a, b in zip(payload["labels"], payload["predictions"], strict=True):
        assert len(a) == len(b)
        gold.append([ID_TO_LABEL[i] for i in a])
        pred.append([ID_TO_LABEL[i] for i in b])
    ner = classification_report(
        gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )
    ner = {
        name: {
            key: int(value) if key == "support" else float(value)
            for key, value in scores.items()
        }
        for name, scores in ner.items()
    }
    expected = load(f"{ner_dir}/experiment_summary.json")["dev_metrics"]
    for field, key in [
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1-score", "f1"),
    ]:
        assert abs(ner["micro avg"][field] - expected[f"eval_{key}"]) < 1e-12
    lora = {}
    lora_legacy = {}
    corrections = {}
    for name in [
        "baseline_test",
        "lora_test",
        "sweep-baseline/validation",
        "sweep-lr2e-4/validation",
        "sweep-lr1e-4/validation",
        "sweep-lr5e-5/validation",
    ]:
        path = f"{lora_dir}/{name}.jsonl"
        sources.append(path)
        lora[name] = score_jsonl(ROOT / path)
        assert lora[name] == load(f"{lora_dir}/{name}.metrics.v2.json")
        lora_legacy[name] = load(f"{lora_dir}/{name}.metrics.json")
        corrections[name] = []
        for line in (ROOT / path).read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            current = score_prediction(row["reference_answer"], row["generation"])
            if row.get("correct") != current["correct"]:
                corrections[name].append(
                    {
                        "question_id": row["question_id"],
                        "reference_answer": row["reference_answer"],
                        "extracted_answer": current["extracted_answer"],
                        "legacy_correct": row.get("correct"),
                        "current_correct": current["correct"],
                    }
                )
    lora_loss = {}
    for name in ["qwen2.5-1.5b-gsm8k", "sweep-lr1e-4", "sweep-lr5e-5"]:
        path = f"{lora_dir}/{name}/checkpoint-421/trainer_state.json"
        sources.append(path)
        state = load(path)
        lora_loss[name] = [row for row in state["log_history"] if "eval_loss" in row][
            -1
        ]["eval_loss"]
    paired = {"both_correct": 0, "baseline_only": 0, "lora_only": 0, "both_wrong": 0}
    base_rows = [
        json.loads(line) for line in (ROOT / lora_dir / "baseline_test.jsonl").open()
    ]
    tuned_rows = [
        json.loads(line) for line in (ROOT / lora_dir / "lora_test.jsonl").open()
    ]
    for a, b in zip(base_rows, tuned_rows, strict=True):
        assert all(
            a[key] == b[key] for key in ("question_id", "question", "reference_answer")
        )
        ca = score_prediction(a["reference_answer"], a["generation"])["correct"]
        cb = score_prediction(b["reference_answer"], b["generation"])["correct"]
        key = {
            (True, True): "both_correct",
            (True, False): "baseline_only",
            (False, True): "lora_only",
            (False, False): "both_wrong",
        }[(ca, cb)]
        paired[key] += 1
    result = {
        "lora_paired_test_decisions": paired,
        "scope": "Saved-prediction rescoring only; NER scores are dev scores used for selection.",
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in sources
        },
        "translation": mt,
        "translation_history": load(sources[1]),
        "ner_classification": ner,
        "ner_history": [
            row for row in load(sources[3])["log_history"] if "eval_f1" in row
        ],
        "lora": lora,
        "lora_legacy_v1": lora_legacy,
        "scoring_version": SCORING_VERSION,
        "scoring_corrections_from_saved_decisions": corrections,
        "lora_validation_loss": lora_loss,
    }
    if args.audit_cached_mt:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from machine_translation.data import audit_split_overlap, load_opus_books_splits

        config = load(f"{mt_dir}/run_config.json")
        splits = load_opus_books_splits(
            config["dataset_name"],
            config["language_pair"],
            config["seed"],
            config["max_examples"],
            config["train_ratio"],
            config["validation_ratio"],
            config["dataset_revision"],
        )
        exported = load_prediction_records(ROOT / sources[0])
        assert len(splits["test"]) == len(exported)
        assert all(
            a["src"] == b["source"] and a["tgt"] == b["reference"]
            for a, b in zip(splits["test"], exported, strict=True)
        )
        result["cached_translation_pair_audit"] = {
            "revision": config["dataset_revision"],
            "matched_pairs": len(exported),
            "scope": "Exact source/reference correspondence and exact duplicates; not a language or semantic-quality audit",
            "overlap": audit_split_overlap(splits),
            "examples": {str(i + 1): splits["test"][i] for i in (1, 13)},
        }
    if args.clean_mt_run is not None:
        import sentencepiece as spm
        from sacrebleu.metrics import CHRF

        from machine_translation.data import (
            audit_split_overlap,
            load_translation_splits,
        )

        run = args.clean_mt_run
        config = load(run / "run_config.json")
        splits = load_translation_splits(config)
        records = load_prediction_records(ROOT / run / "predictions.test.jsonl")
        if len(records) != len(splits["test"]) or any(
            row["src"] != pred["source"] or row["tgt"] != pred["reference"]
            for row, pred in zip(splits["test"], records, strict=True)
        ):
            raise ValueError(
                "Clean translation predictions do not match the pinned test split"
            )
        metric = score_records(records)
        saved_metric = load(run / "metrics.test.json")
        if any(saved_metric[key] != value for key, value in metric.items()):
            raise ValueError(
                "Clean translation metric disagrees with independent scoring"
            )
        chrf = CHRF(word_order=2)
        chrf_score = chrf.corpus_score(
            [row["prediction"] for row in records],
            [[row["reference"] for row in records]],
        )
        tokenizer = spm.SentencePieceProcessor(
            model_file=str(ROOT / run / "tokenizer.model")
        )
        truncation = {
            name: {
                side: sum(
                    len(tokenizer.encode(row[side])) + extra > config["max_length"]
                    for row in rows
                )
                for side, extra in (("src", 1), ("tgt", 2))
            }
            for name, rows in splits.items()
        }
        history = load(run / "history.json")
        prepared_manifest = load(Path(config["prepared_data_dir"]) / "manifest.json")
        result["translation_clean"] = {
            "run_dir": str(run),
            "metric": metric,
            "chrf_plus_plus": {
                "score": chrf_score.score,
                "signature": str(chrf.get_signature()),
            },
            "history": history,
            "overlap": audit_split_overlap(splits),
            "would_truncate_by_split": truncation,
            "manifest_sha256": config["prepared_manifest_sha256"],
            "cleaning": {
                key: prepared_manifest[key]
                for key in (
                    "input_rows",
                    "kept_rows",
                    "rejections",
                    "split_policy",
                    "limitations",
                )
            },
            "candidate_examples": [
                {
                    "test_line": index + 1,
                    "row_id": splits["test"][index]["row_id"],
                    **row,
                    "review_status": "systematic_first_10_not_student_verified",
                }
                for index, row in enumerate(records[:10])
            ],
            "comparison_limit": "Different cleaned/grouped test set; not a controlled improvement over the legacy BLEU",
        }
        for filename in (
            "run_config.json",
            "data_manifest.json",
            "runtime.json",
            "code_snapshot.json",
            "best.pt",
            "tokenizer.model",
            "history.json",
            "predictions.test.jsonl",
            "metrics.test.json",
        ):
            path = run / filename
            result["source_sha256"][str(path)] = hashlib.sha256(
                (ROOT / path).read_bytes()
            ).hexdigest()
        (ROOT / "reports/evidence/translation_clean_loss.csv").write_text(
            "epoch,train_loss,validation_loss\n"
            + "".join(
                f"{row['epoch']},{row['train_loss']},{row['validation_loss']}\n"
                for row in history
            ),
            encoding="utf-8",
        )
    destination = ROOT / "reports/evidence/results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    curve = destination.with_name("translation_loss.csv")
    curve.write_text(
        "epoch,train_loss,validation_loss\n"
        + "".join(
            f"{row['epoch']},{row['train_loss']},{row['validation_loss']}\n"
            for row in result["translation_history"]
        ),
        encoding="utf-8",
    )
    print(
        "All current metric files agree with independent recomputation; legacy v1 retained separately."
    )
    print("LoRA paired decisions:", paired)
    print(destination)


if __name__ == "__main__":
    main()
