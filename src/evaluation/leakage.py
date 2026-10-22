"""Leakage detection for the diagnosis dataset.

The dataset is synthetic, and that has consequences for what a score
means. This module measures them instead of assuming them:

* how many *distinct* model inputs the rows actually contain;
* how many rows are copies of one another;
* whether a row's model input spans more than one label (ambiguity);
* whether identifiers such as the task id predict the label on their own;
* how many rows of a test split are literal copies of a training row.

The audit is the evidence behind the split method used for reporting.
"""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.diagnosis.features import IDENTIFIER_COLUMNS, TEXT_COLUMNS, _as_text
from src.evaluation.splitting import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    case_group,
    group_folds,
    stratified_split,
)


TARGET_COLUMN = "failure_label"


def template_family(row: object) -> str:
    """Return the generation template a row was produced from.

    Tool names and numbers are replaced by placeholders, so the ten
    different "unknown tool" rows collapse into one family. The number of
    families tells us how many distinct situations the dataset contains.
    """
    get = row.get  # type: ignore[union-attr]

    tool = _as_text(get("tool_name"))
    error = _as_text(get("error"))

    if tool:
        error = error.replace(tool, "<tool>")

    error = re.sub(r"\d+", "<n>", error)

    return "|".join([
        _as_text(get("agent_type")),
        _as_text(get("status")),
        _as_text(get("error_type")),
        error,
    ])


def signature_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return how many rows share each distinct model input."""
    counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        signature = case_group(row)
        counts[signature] = counts.get(signature, 0) + 1

    return counts


def split_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Describe how much of the test set is copied from the training set."""
    train_signatures = set(signature_counts(train_df))
    test_signatures = signature_counts(test_df)

    overlapping = {
        signature for signature in test_signatures
        if signature in train_signatures
    }

    copied_rows = sum(
        count for signature, count in test_signatures.items()
        if signature in train_signatures
    )

    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "distinct_train_signatures": len(train_signatures),
        "distinct_test_signatures": len(test_signatures),
        "signatures_in_both_splits": len(overlapping),
        "test_rows_copied_from_train": int(copied_rows),
        "test_rows_copied_fraction": (
            float(copied_rows) / float(len(test_df))
            if len(test_df) else 0.0
        ),
    }


def probe_single_feature(
    df: pd.DataFrame,
    column: str,
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    target_column: str = TARGET_COLUMN,
) -> dict:
    """Measure how well one column alone predicts the label.

    A column that predicts the label on its own is a shortcut: a model
    given it would look accurate without diagnosing anything.

    Args:
        df: The dataset.
        column: The column to probe.
        seed: Random seed for the probe's own split.
        test_size: Fraction held out for the probe.
        target_column: The label column.

    Returns:
        The probe's held-out accuracy and the label distribution.
    """
    if column not in df.columns:
        raise ValueError(f"Column not in the dataset: {column}")

    values = df[column]

    if pd.api.types.is_numeric_dtype(values):
        X = values.to_numpy(dtype=float).reshape(-1, 1)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
    else:
        X = values.astype(str).to_numpy()
        model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])

    y = df[target_column].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "column": column,
        "held_out_accuracy": float(
            np.mean(predictions == y_test)
        ),
        "n_test": int(len(y_test)),
    }


def probe_ordinal_neighbours(
    df: pd.DataFrame,
    column: str,
    *,
    target_column: str = TARGET_COLUMN,
) -> dict:
    """Measure whether neighbouring values of a column share a label.

    A column whose values are issued in blocks -- task ids handed out one
    class at a time, for example -- lets a model that sees the column
    recover the label by looking at its neighbours. The accuracy below is
    what such a model would achieve: every row is predicted from the label
    of the row immediately before it in the column's order.

    Args:
        df: The dataset.
        column: The column to probe.
        target_column: The label column.

    Returns:
        The neighbour accuracy and the number of label changes along the
        column's order.
    """
    if column not in df.columns:
        raise ValueError(f"Column not in the dataset: {column}")

    ordered = df.sort_values(column, kind="stable")
    labels = ordered[target_column].to_numpy()

    if len(labels) < 2:
        return {
            "column": column,
            "neighbour_accuracy": 0.0,
            "label_transitions": 0,
        }

    predictions = labels[:-1]
    actual = labels[1:]

    transitions = int(sum(
        1 for index in range(len(labels) - 1)
        if labels[index] != labels[index + 1]
    ))

    return {
        "column": column,
        "neighbour_accuracy": float(np.mean(predictions == actual)),
        "label_transitions": transitions,
    }


def audit_dataset(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    target_column: str = TARGET_COLUMN,
) -> dict:
    """Run the full leakage audit and return its findings.

    Args:
        df: The dataset to audit.
        seed: Random seed for the probes and the splits.
        target_column: The label column.

    Returns:
        A dictionary describing duplication, template families, identifier
        shortcuts and how much each split method leaks.
    """
    signatures = signature_counts(df)

    families: Dict[str, int] = {}
    for _, row in df.iterrows():
        family = template_family(row)
        families[family] = families.get(family, 0) + 1

    label_counts = df[target_column].value_counts().to_dict()

    # A model input that appears with more than one label is genuinely
    # ambiguous; everything else is perfectly learnable.
    signature_labels: Dict[str, set] = {}
    for _, row in df.iterrows():
        signature_labels.setdefault(case_group(row), set()).add(
            str(row[target_column])
        )

    ambiguous = sorted(
        signature for signature, labels in signature_labels.items()
        if len(labels) > 1
    )

    naive = stratified_split(df, seed=seed, target_column=target_column)

    # The grouped split is measured on the first grouped fold. The folds
    # themselves carry no label-coverage requirement -- that check belongs
    # to ``group_split``, and this audit must also work on small frames.
    folds = group_folds(df, seed=seed, target_column=target_column)
    grouped_train, grouped_test = folds[0]

    naive_overlap = split_overlap(
        df.iloc[naive.train_index], df.iloc[naive.test_index]
    )
    grouped_overlap = split_overlap(
        df.iloc[grouped_train], df.iloc[grouped_test]
    )

    shortcuts = {}
    for column in IDENTIFIER_COLUMNS:
        if column in df.columns:
            shortcuts[column] = probe_single_feature(
                df, column, seed=seed, target_column=target_column
            )
            shortcuts[column]["ordinal_neighbour_probe"] = (
                probe_ordinal_neighbours(
                    df, column, target_column=target_column
                )
            )

    return {
        "n_rows": int(len(df)),
        "label_distribution": {
            str(key): int(value) for key, value in label_counts.items()
        },
        "distinct_model_inputs": len(signatures),
        "duplicate_rows": int(len(df) - len(signatures)),
        "largest_group_size": int(max(signatures.values())) if signatures else 0,
        "distinct_template_families": len(families),
        "template_families": dict(sorted(families.items())),
        "ambiguous_model_inputs": len(ambiguous),
        "ambiguous_examples": ambiguous[:5],
        "identifier_shortcuts": shortcuts,
        "text_columns": list(TEXT_COLUMNS),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "split_overlap": {
            "stratified_random": naive_overlap,
            "stratified_group_kfold": grouped_overlap,
        },
    }
