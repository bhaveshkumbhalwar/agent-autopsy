"""Running a classifier pipeline over the evaluation splits.

The functions here are shared by the single-model evaluation
(``src/evaluation/evaluate_model.py``) and by the baseline/combined
comparison (``src/diagnosis/compare_models.py``), so that both use
exactly the same splits and exactly the same metric definitions.

Every function takes a *pipeline factory* -- a callable that returns an
unfitted pipeline -- and fits it inside, on the training rows only.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.splitting import (
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    TARGET_COLUMN,
    group_folds,
    label_distribution,
    split_frame,
)


PipelineFactory = Callable[[], object]


def as_traces(frame: pd.DataFrame) -> List[dict]:
    """Return the rows of a frame as the traces the pipelines consume."""
    return frame.to_dict("records")


def fit_predict(
    pipeline_factory: PipelineFactory,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> List[str]:
    """Fit a pipeline on ``train_frame`` and predict ``test_frame``.

    The pipeline is fitted here, on the training rows only, so the
    vectorizer and the scaler never see a test row.
    """
    pipeline = pipeline_factory()

    pipeline.fit(
        as_traces(train_frame),
        train_frame[TARGET_COLUMN].to_numpy(),
    )

    return [str(value) for value in pipeline.predict(as_traces(test_frame))]


def prediction_rows(
    frame: pd.DataFrame,
    predictions: Sequence[str],
    extra_columns: Optional[List[str]] = None,
) -> List[dict]:
    """Turn predictions into the rows written to a prediction report."""
    rows = []

    for (_, row), prediction in zip(frame.iterrows(), predictions):
        expected = str(row[TARGET_COLUMN])
        entry = {
            "task_id": row.get("task_id", ""),
            "expected_label": expected,
            "predicted_label": str(prediction),
            "correct": bool(prediction == expected),
        }

        for column in extra_columns or []:
            if column in frame.columns:
                entry[column] = row[column]

        rows.append(entry)

    return rows


def cross_validate(
    pipeline_factory: PipelineFactory,
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
    target_column: str = TARGET_COLUMN,
) -> Dict[str, object]:
    """Score every row by a model that never saw its template.

    Each fold's test rows are predicted by a pipeline fitted on that
    fold's training rows. The reported metrics are computed over the
    pooled out-of-fold predictions, which keeps the per-class support
    balanced even when the individual folds are not.

    Returns:
        The metrics, the per-fold summaries and the per-row predictions.
    """
    labels = df[target_column].to_numpy()

    folds = group_folds(
        df, seed=seed, n_splits=n_splits, target_column=target_column
    )

    out_of_fold = np.empty(len(df), dtype=object)
    fold_summaries = []

    for number, (train_index, test_index) in enumerate(folds):
        train_frame = df.iloc[train_index]
        test_frame = df.iloc[test_index]

        predictions = fit_predict(pipeline_factory, train_frame, test_frame)

        out_of_fold[test_index] = predictions

        fold_summaries.append({
            "fold": number,
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "test_label_distribution": label_distribution(
                df, test_index, target_column
            ),
            "test_accuracy": float(
                np.mean(predictions == labels[test_index])
            ),
        })

    predictions = [str(value) for value in out_of_fold]

    metrics = compute_metrics(
        df[target_column].tolist(),
        predictions,
        labels=sorted(df[target_column].unique()),
    )

    return {
        "split_method": (
            f"stratified_group_kfold(n_splits={n_splits}) with pooled "
            "out-of-fold predictions"
        ),
        "seed": seed,
        "n_splits": n_splits,
        "n_scored": int(len(df)),
        "label_distribution": label_distribution(
            df, np.arange(len(df)), target_column
        ),
        "folds": fold_summaries,
        "metrics": metrics,
        "predictions": prediction_rows(df, predictions),
    }


def evaluate_holdout(
    pipeline_factory: PipelineFactory,
    df: pd.DataFrame,
    split,
) -> Dict[str, object]:
    """Train on the split's training rows and score its test rows."""
    train_frame, test_frame = split_frame(df, split)

    predictions = fit_predict(pipeline_factory, train_frame, test_frame)

    metrics = compute_metrics(
        test_frame[TARGET_COLUMN].tolist(),
        predictions,
        labels=sorted(df[TARGET_COLUMN].unique()),
    )

    return {
        "split_method": split.method,
        "seed": split.seed,
        "n_train": split.n_train,
        "n_test": split.n_test,
        "train_label_distribution": label_distribution(
            df, split.train_index, TARGET_COLUMN
        ),
        "test_label_distribution": label_distribution(
            df, split.test_index, TARGET_COLUMN
        ),
        "metrics": metrics,
        "predictions": prediction_rows(test_frame, predictions),
    }


def evaluate_dataset(
    pipeline_factory: PipelineFactory,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    extra_columns: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Train on one dataset and score another.

    The two datasets must be disjoint: the caller checks that before
    getting here, so that a final test set is never trained on.
    """
    predictions = fit_predict(pipeline_factory, train_frame, test_frame)

    metrics = compute_metrics(
        test_frame[TARGET_COLUMN].tolist(),
        predictions,
        labels=sorted(train_frame[TARGET_COLUMN].unique()),
    )

    return {
        "metrics": metrics,
        "predictions": prediction_rows(test_frame, predictions, extra_columns),
    }


def by_case_kind(
    frame: pd.DataFrame,
    predictions: Sequence[str],
    *,
    target_column: str = TARGET_COLUMN,
) -> List[dict]:
    """Break results down by the stress each case applies."""
    rows = []

    working = frame.copy()
    working["predicted_label"] = predictions

    for kind, group in working.groupby("case_kind"):
        correct = sum(
            1 for _, row in group.iterrows()
            if str(row["predicted_label"]) == str(row[target_column])
        )
        rows.append({
            "case_kind": kind,
            "n_cases": int(len(group)),
            "correct": int(correct),
            "accuracy": float(correct / len(group)) if len(group) else 0.0,
        })

    return sorted(rows, key=lambda row: row["case_kind"])
