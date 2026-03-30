"""Metrics for evaluating the null/unanswerable classifier."""

from __future__ import annotations


def null_confusion_matrix(
    pred_answerable: list[bool],
    true_answerable: list[bool],
) -> dict:
    """Compute confusion matrix and derived metrics for answerability prediction.

    Parameters
    ----------
    pred_answerable : list[bool]
        Model predictions — True means predicted answerable.
    true_answerable : list[bool]
        Ground truth labels — True means actually answerable.

    Returns
    -------
    dict
        Keys: tp, fp, tn, fn, precision, recall, f1, total.
    """
    tp = fp = tn = fn = 0

    for pred, true in zip(pred_answerable, true_answerable):
        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and not true:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
    }
