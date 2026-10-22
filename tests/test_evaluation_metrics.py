"""Tests for the metric definitions used by the evaluation.

The evaluation report, the JSON artefact and these tests must agree on
what "accuracy" and "F1" mean, so the numbers are checked by hand here
rather than compared against the report.
"""

import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    format_confusion_matrix,
    write_confusion_matrix,
    write_predictions,
)


def test_perfect_predictions_score_one():
    y_true = ["none", "none", "unknown_tool", "unknown_tool"]
    y_pred = ["none", "none", "unknown_tool", "unknown_tool"]

    metrics = compute_metrics(y_true, y_pred, labels=["none", "unknown_tool"])

    assert metrics["accuracy"] == 1.0
    assert metrics["n_samples"] == 4

    for label in ("none", "unknown_tool"):
        scores = metrics["per_class"][label]
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1_score"] == 1.0

    assert metrics["macro_avg"]["f1_score"] == 1.0
    assert metrics["weighted_avg"]["f1_score"] == 1.0


def test_confusion_matrix_and_per_class_scores_are_hand_checked():
    # Truth:  a a b b      Prediction:  a b b b
    #         ^ one a predicted as b, so:
    #   precision(a) = 1 / (1 + 0) = 1.0
    #   recall(a)    = 1 / (1 + 1) = 0.5
    #   f1(a)        = 2 * 1 * 0.5 / 1.5 = 0.666667
    #   precision(b) = 2 / (1 + 2) = 0.666667
    #   recall(b)    = 2 / 2 = 1.0
    #   f1(b)        = 2 * 0.666667 * 1 / 1.666667 = 0.8
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]

    metrics = compute_metrics(y_true, y_pred, labels=["a", "b"])

    assert metrics["accuracy"] == pytest.approx(0.75)

    assert metrics["confusion_matrix"]["labels"] == ["a", "b"]
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]

    scores_a = metrics["per_class"]["a"]
    assert scores_a["precision"] == pytest.approx(1.0)
    assert scores_a["recall"] == pytest.approx(0.5)
    assert scores_a["f1_score"] == pytest.approx(0.6666667, rel=1e-5)
    assert scores_a["support"] == 2

    scores_b = metrics["per_class"]["b"]
    assert scores_b["precision"] == pytest.approx(0.6666667, rel=1e-5)
    assert scores_b["recall"] == pytest.approx(1.0)
    assert scores_b["f1_score"] == pytest.approx(0.8)
    assert scores_b["support"] == 2


def test_macro_average_is_the_mean_of_the_per_class_scores():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]

    metrics = compute_metrics(y_true, y_pred, labels=["a", "b", "c"])

    per_class = list(metrics["per_class"].values())
    expected_macro = sum(s["f1_score"] for s in per_class) / len(per_class)

    assert metrics["macro_avg"]["f1_score"] == pytest.approx(expected_macro)

    total_support = sum(s["support"] for s in per_class)
    assert total_support == metrics["n_samples"]


def test_support_counts_the_test_rows_of_each_class():
    y_true = ["a", "b", "b", "b"]
    y_pred = ["a", "a", "b", "b"]

    metrics = compute_metrics(y_true, y_pred, labels=["a", "b"])

    assert metrics["per_class"]["a"]["support"] == 1
    assert metrics["per_class"]["b"]["support"] == 3


def test_a_label_absent_from_the_truth_is_still_reported():
    """A label the model predicted but never saw must be reported as zero."""
    y_true = ["a", "a"]
    y_pred = ["a", "b"]

    metrics = compute_metrics(y_true, y_pred, labels=["a", "b"])

    assert metrics["per_class"]["b"]["support"] == 0
    assert metrics["per_class"]["b"]["recall"] == 0.0
    assert metrics["per_class"]["b"]["f1_score"] == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        compute_metrics(["a", "b"], ["a"])


def test_an_empty_test_set_is_rejected():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_confusion_matrix_frame_is_labelled():
    metrics = compute_metrics(["a", "b"], ["a", "b"], labels=["a", "b"])

    frame = format_confusion_matrix(metrics)

    assert list(frame.index) == ["a", "b"]
    assert list(frame.columns) == ["a", "b"]
    assert frame.loc["a", "a"] == 1
    assert frame.loc["b", "b"] == 1


def test_confusion_matrix_is_written_to_disk(tmp_path):
    metrics = compute_metrics(["a", "b"], ["a", "a"], labels=["a", "b"])

    path = write_confusion_matrix(metrics, tmp_path / "matrix.csv")

    assert path.exists()

    frame = pd.read_csv(path, index_col=0)
    assert frame.loc["a", "a"] == 1
    assert frame.loc["b", "a"] == 1
    assert frame.loc["b", "b"] == 0


def test_predictions_are_written_to_disk(tmp_path):
    rows = [
        {"task_id": "t-1", "expected_label": "a", "predicted_label": "a",
         "correct": True},
        {"task_id": "t-2", "expected_label": "b", "predicted_label": "a",
         "correct": False},
    ]

    path = write_predictions(rows, tmp_path / "predictions.csv")

    frame = pd.read_csv(path)
    assert len(frame) == 2
    assert bool(frame.loc[0, "correct"]) is True
    assert bool(frame.loc[1, "correct"]) is False
