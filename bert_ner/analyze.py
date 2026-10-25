"""Audit saved NER dev predictions and export reproducible entity error evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ner_data import ID_TO_LABEL, encode_example, load_jsonl, to_bio
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2, Entities


def strict_spans(tags, positions=None):
    """Use seqeval strict IOB2; return character spans with exclusive ends."""
    if positions is None:
        positions = list(range(len(tags)))
    if len(positions) != len(tags):
        raise ValueError("Tag and character position lengths differ")
    return {
        (positions[entity.start], positions[entity.end - 1] + 1, entity.tag)
        for entity in Entities([tags], IOB2).entities[0]
    }


def classify_error(entity, others):
    start, end, tag = entity
    if any(a == start and b == end and kind != tag for a, b, kind in others):
        return "same_span_wrong_type"
    overlapping = [(a, b, kind) for a, b, kind in others if a < end and start < b]
    if any(kind == tag for _, _, kind in overlapping):
        return "same_type_wrong_boundary"
    return "overlap_wrong_type_and_boundary" if overlapping else "no_overlapping_entity"


def entity_summary(texts, gold, predicted, known):
    """Partition FN and FP separately by existence of overlapping entities."""
    fn, fp, confusions = Counter(), Counter(), Counter()
    novelty = {name: Counter() for name in ("seen", "unseen")}
    cases = defaultdict(list)
    true_positive = 0
    for index, (text, reference, prediction) in enumerate(
        zip(texts, gold, predicted, strict=True)
    ):
        matched = reference & prediction
        true_positive += len(matched)
        for name, entities in (("gold", reference), ("predicted", prediction)):
            for start, end, tag in entities:
                group = "seen" if (text[start:end], tag) in known else "unseen"
                novelty[group][name] += 1
        for start, end, tag in matched:
            group = "seen" if (text[start:end], tag) in known else "unseen"
            novelty[group]["correct"] += 1
        for side, errors, others, counts in (
            ("false_negative", reference - matched, prediction, fn),
            ("false_positive", prediction - matched, reference, fp),
        ):
            for entity in sorted(errors):
                category = classify_error(entity, others)
                counts[category] += 1
                cases[f"{side}/{category}"].append(
                    {
                        "dev_index": index,
                        "side": side,
                        "category": category,
                        "entity": list(entity),
                        "text": text,
                        "gold_spans": [list(span) for span in sorted(reference)],
                        "predicted_spans": [list(span) for span in sorted(prediction)],
                        "review_status": "automatic_candidate_not_student_verified",
                    }
                )
                if side == "false_negative" and category == "same_span_wrong_type":
                    for start, end, tag in others:
                        if (start, end) == entity[:2] and tag != entity[2]:
                            confusions[f"{entity[2]}->{tag}"] += 1
    for counts in novelty.values():
        correct = counts["correct"]
        counts["precision"] = (
            correct / counts["predicted"] if counts["predicted"] else 0.0
        )
        counts["recall"] = correct / counts["gold"] if counts["gold"] else 0.0
        counts["f1"] = (
            2 * correct / (counts["gold"] + counts["predicted"])
            if counts["gold"] + counts["predicted"]
            else 0.0
        )
    return {
        "true_positive": true_positive,
        "false_negative": sum(fn.values()),
        "false_positive": sum(fp.values()),
        "false_negative_categories": dict(sorted(fn.items())),
        "false_positive_categories": dict(sorted(fp.items())),
        "same_span_type_confusions": dict(sorted(confusions.items())),
        "surface_type_seen_in_training": novelty,
        "category_definition": "Each unmatched gold/pred entity assigned once: same-span wrong type, else overlapping same-type boundary, else overlapping other-type, else no overlap. FN/FP partitions are independent existence rules, not one-to-one error matches.",
    }, cases


def run_analysis(run_dir: Path, data_dir: Path, output: Path, seed=42):
    from transformers import AutoTokenizer

    prediction_path = run_dir / "dev_predictions.json"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    train_rows, dev_rows = (
        load_jsonl(data_dir / f"{name}.json") for name in ("train", "dev")
    )
    if len(dev_rows) != len(payload["labels"]) or len(dev_rows) != len(
        payload["predictions"]
    ):
        raise ValueError("Saved sequence count differs from local dev")
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "best_model", local_files_only=True, use_fast=True
    )
    known = set()
    for row in train_rows:
        _, tags, _ = to_bio(row["text"], row["label"])
        known.update((row["text"][a:b], tag) for a, b, tag in strict_spans(tags))
    gold, predicted, gold_tags, pred_tags = [], [], [], []
    missing_chars = 0
    for row, saved_gold, saved_pred in zip(
        dev_rows, payload["labels"], payload["predictions"], strict=True
    ):
        encoded = encode_example(row, tokenizer, config["max_length"])
        expected = [int(tag) for tag in encoded["labels"] if tag != -100]
        if saved_gold != expected or len(saved_gold) != len(saved_pred):
            raise ValueError(
                "Saved gold or prediction alignment differs from local encoding"
            )
        positions = [
            word
            for word, tag in zip(encoded.word_ids(), encoded["labels"], strict=True)
            if tag != -100
        ]
        missing_chars += len(row["text"]) - len(positions)
        g, p = (
            [ID_TO_LABEL[tag] for tag in sequence]
            for sequence in (saved_gold, saved_pred)
        )
        gold_tags.append(g)
        pred_tags.append(p)
        gold.append(strict_spans(g, positions))
        predicted.append(strict_spans(p, positions))
    report = classification_report(
        gold_tags,
        pred_tags,
        scheme=IOB2,
        mode="strict",
        output_dict=True,
        zero_division=0,
    )
    report = {
        name: {
            key: int(value) if key == "support" else float(value)
            for key, value in scores.items()
        }
        for name, scores in report.items()
    }
    statistics, cases = entity_summary(
        [row["text"] for row in dev_rows], gold, predicted, known
    )
    expected = json.loads(
        (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
    )["dev_metrics"]
    for key, metric in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1-score", "f1"),
    ):
        if abs(report["micro avg"][key] - expected[f"eval_{metric}"]) > 1e-12:
            raise ValueError("Recomputed strict metrics differ from saved results")
    tp = statistics["true_positive"]
    if (
        abs(tp / (tp + statistics["false_negative"]) - report["micro avg"]["recall"])
        > 1e-12
    ):
        raise ValueError("Character span counts differ from strict sequence metrics")
    if (
        abs(tp / (tp + statistics["false_positive"]) - report["micro avg"]["precision"])
        > 1e-12
    ):
        raise ValueError("Predicted character span counts differ from strict metrics")
    rng = random.Random(seed)
    samples = [
        candidate
        for _, records in sorted(cases.items())
        for candidate in rng.sample(records, min(3, len(records)))
    ]
    sources = [
        prediction_path,
        run_dir / "config.json",
        run_dir / "experiment_summary.json",
        data_dir / "train.json",
        data_dir / "dev.json",
    ]
    sources += sorted(
        path
        for path in (run_dir / "best_model").iterdir()
        if path.suffix in {".json", ".txt"}
    )
    result = {
        "scope": "Saved selected-dev diagnostics, not independent test performance",
        "span_convention": "Zero-based character positions, exclusive end",
        "dev_examples": len(dev_rows),
        "characters_without_supervised_token": missing_chars,
        "train_dev_exact_text_overlap": len(
            {row["text"] for row in train_rows} & {row["text"] for row in dev_rows}
        ),
        "metrics": report,
        "entity_errors": statistics,
        "sampling_seed": seed,
        "candidate_examples": samples,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }
    if output.resolve() in {path.resolve() for path in sources}:
        raise ValueError("Analysis must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "dev_examples",
                    "characters_without_supervised_token",
                    "entity_errors",
                )
            },
            ensure_ascii=False,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("bert_ner/outputs/bert-base-chinese")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("bert_ner/data/cluener2020")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_analysis(args.run_dir, args.data_dir, args.output)


if __name__ == "__main__":
    main()
