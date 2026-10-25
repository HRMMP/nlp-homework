"""Entity-level metrics for BIO predictions."""

from __future__ import annotations

from collections.abc import Sequence


def compute_entity_metrics(
    predictions: Sequence[Sequence[int]],
    labels: Sequence[Sequence[int]],
    id_to_label: dict[int, str],
) -> dict[str, float]:
    """Compute strict IOB2 entity-level scores with seqeval."""
    from seqeval.metrics import f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2

    true_sequences = []
    predicted_sequences = []
    for predicted, gold in zip(predictions, labels, strict=True):
        true_tags = []
        predicted_tags = []
        for predicted_id, gold_id in zip(predicted, gold, strict=True):
            if gold_id == -100:
                continue
            true_tags.append(id_to_label[int(gold_id)])
            predicted_tags.append(id_to_label[int(predicted_id)])
        true_sequences.append(true_tags)
        predicted_sequences.append(predicted_tags)
    return {
        "precision": float(
            precision_score(
                true_sequences,
                predicted_sequences,
                average="micro",
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_sequences,
                predicted_sequences,
                average="micro",
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_sequences,
                predicted_sequences,
                average="micro",
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        ),
    }
