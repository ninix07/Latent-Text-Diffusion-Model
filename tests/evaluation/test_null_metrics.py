"""Tests for null prediction metrics."""

from src.evaluation.null_metrics import null_confusion_matrix


def test_confusion_matrix_sums():
    pred = [True, False, True, False, True]
    true = [True, True, False, False, True]
    r = null_confusion_matrix(pred, true)
    assert r["tp"] + r["fp"] + r["tn"] + r["fn"] == r["total"] == 5


def test_perfect_predictions():
    pred = [True, True, False, False]
    true = [True, True, False, False]
    r = null_confusion_matrix(pred, true)
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0


def test_all_wrong():
    pred = [True, True, False, False]
    true = [False, False, True, True]
    r = null_confusion_matrix(pred, true)
    assert r["f1"] == 0.0


def test_all_true_predictions():
    pred = [True, True, True, True]
    true = [True, False, True, False]
    r = null_confusion_matrix(pred, true)
    assert r["tp"] == 2
    assert r["fp"] == 2
    assert r["tn"] == 0
    assert r["fn"] == 0
