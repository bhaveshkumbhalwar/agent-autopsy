"""Metric computation for the diagnosis evaluation.

Every number the evaluation reports comes from here, so that the report,
the JSON artefact and the tests all agree on the definitions.

The metric set is deliberately complete: accuracy alone hides what a model
actually does with the failure classes that matter.
"""

from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """Compute accuracy, per-class scores, support and a confusion matrix.

    Args:
        y_true: The ground-truth labels.
        y_pred: The predicted labels.
        labels: The label order to use. Defaults to the sorted union of the
            true and predicted labels.

    Returns:
        A dictionary with the overall scores, the per-class breakdown and
        the confusion matrix.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Got {len(y_true)} true labels and {len(y_pred)} predictions."
        )

    if not y_true:
        raise ValueError("Cannot compute metrics for an empty test set.")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: Dict[str, dict] = {}
    for label in labels:
        scores = report.get(label, {})
        per_class[label] = {
            "precision": float(scores.get("precision", 0.0)),
            "recall": float(scores.get("recall", 0.0)),
            "f1_score": float(scores.get("f1-score", 0.0)),
            "support": int(scores.get("support", 0)),
        }

    def _average(name: str) -> dict:
        scores = report.get(name, {})
        return {
            "precision": float(scores.get("precision", 0.0)),
            "recall": float(scores.get("recall", 0.0)),
            "f1_score": float(scores.get("f1-score", 0.0)),
            "support": int(scores.get("support", 0)),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": len(y_true),
        "labels": labels,
        "per_class": per_class,
        "macro_avg": _average("macro avg"),
        "weighted_avg": _average("weighted avg"),
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[int(value) for value in row] for row in matrix],
        },
    }


def format_confusion_matrix(metrics: dict) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame."""
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]

    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "actual\\predicted"
    frame.columns.name = "predicted"

    return frame


def write_confusion_matrix(metrics: dict, path: Path) -> Path:
    """Write the confusion matrix to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    format_confusion_matrix(metrics).to_csv(path)

    return path


def write_predictions(
    rows: List[dict],
    path: Path,
) -> Path:
    """Write per-example predictions to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)

    return path


def bootstrap_accuracy_interval(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    seed: int = 42,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """Return a bootstrap confidence interval for accuracy.

    The interval is built over *cases*: it describes how far the accuracy
    could move if the evaluation set had been drawn differently. On a
    small set it is wide, and quoting an accuracy without it overstates
    what the measurement can support.

    Args:
        y_true: The ground-truth labels.
        y_pred: The predicted labels.
        seed: Random seed, so the interval is reproducible.
        n_resamples: Number of bootstrap resamples.
        confidence: The confidence level.

    Returns:
        The accuracy, its interval and the sample size.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Got {len(y_true)} true labels and {len(y_pred)} predictions."
        )

    if not y_true:
        raise ValueError("Cannot compute an interval for an empty test set.")

    correct = np.array(
        [1.0 if true == predicted else 0.0
         for true, predicted in zip(y_true, y_pred)]
    )

    rng = np.random.default_rng(seed)
    n_cases = len(correct)

    means = np.empty(n_resamples)
    for index in range(n_resamples):
        sample = rng.integers(0, n_cases, n_cases)
        means[index] = correct[sample].mean()

    tail = (1.0 - confidence) / 2.0

    return {
        "accuracy": float(correct.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_cases": n_cases,
    }


def mcnemar_exact_test(
    y_true: Sequence[str],
    first: Sequence[str],
    second: Sequence[str],
) -> dict:
    """Compare two classifiers with an exact McNemar test.

    Only the cases the two models disagree on carry information about
    which one is better. When they disagree on a handful of cases the
    test cannot distinguish them, and the result says so.

    Args:
        y_true: The ground-truth labels.
        first: The first model's predictions.
        second: The second model's predictions.

    Returns:
        The discordant counts and the two-sided exact p-value.
    """
    y_true = list(y_true)
    first = list(first)
    second = list(second)

    if not (len(y_true) == len(first) == len(second)):
        raise ValueError("All three sequences must have the same length.")

    first_only = sum(
        1 for true, a, b in zip(y_true, first, second)
        if a == true and b != true
    )
    second_only = sum(
        1 for true, a, b in zip(y_true, first, second)
        if b == true and a != true
    )

    discordant = first_only + second_only

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(comb(discordant, k) for k in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))

    return {
        "only_first_correct": int(first_only),
        "only_second_correct": int(second_only),
        "discordant": int(discordant),
        "exact_p_value": float(p_value),
    }
