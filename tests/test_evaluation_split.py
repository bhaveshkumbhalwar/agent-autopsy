"""Tests for the reproducible, leakage-aware splits.

The point of these tests is that the reported numbers mean what they claim
to mean: the same seed always produces the same split, and a grouped split
never lets a row's model input appear on both sides.
"""

import pandas as pd
import pytest

from src.diagnosis.features import build_text
from src.evaluation.splitting import (
    case_group,
    group_folds,
    group_labels,
    group_split,
    label_distribution,
    split_frame,
    stratified_split,
)


def _make_frame() -> pd.DataFrame:
    """Build a frame with the same shape as the generated dataset.

    Eight template groups, two per label, four rows each. The rows of a
    group differ only in values the model never sees, exactly like the
    real dataset.
    """
    templates = [
        ("add", "success", "", "", "none"),
        ("multiply", "success", "", "", "none"),
        ("divide", "failure", "ValueError", "Unknown tool: divide",
         "unknown_tool"),
        ("subtract", "failure", "ValueError", "Unknown tool: subtract",
         "unknown_tool"),
        ("add", "failure", "KeyError", "'b'", "missing_argument"),
        ("multiply", "failure", "KeyError", "'a'", "missing_argument"),
        ("add", "failure", "ValueError",
         "Arguments a and b must be numbers", "invalid_input"),
        ("multiply", "failure", "ValueError",
         "Arguments a and b must be numbers", "invalid_input"),
    ]

    rows = []
    number = 0

    for tool, status, error_type, error, label in templates:
        for _ in range(4):
            number += 1
            rows.append({
                "task_id": f"t-{number:03d}",
                "agent_type": "calculator",
                "tool_name": tool,
                "status": status,
                "error_type": error_type,
                "error": error,
                "failure_label": label,
            })

    return pd.DataFrame(rows)


def test_case_group_is_the_model_input():
    df = _make_frame()

    assert case_group(df.iloc[0]) == build_text(df.iloc[0])
    assert len(set(group_labels(df))) == 8


def test_the_same_seed_produces_the_same_split():
    df = _make_frame()

    first = group_split(df, seed=7, n_splits=2)
    second = group_split(df, seed=7, n_splits=2)

    assert list(first.train_index) == list(second.train_index)
    assert list(first.test_index) == list(second.test_index)


def test_different_seeds_produce_different_splits():
    df = _make_frame()

    first = group_split(df, seed=7, n_splits=2)
    second = group_split(df, seed=8, n_splits=2)

    assert list(first.test_index) != list(second.test_index)


def test_stratified_split_is_also_reproducible():
    df = _make_frame()

    first = stratified_split(df, seed=42)
    second = stratified_split(df, seed=42)

    assert list(first.train_index) == list(second.train_index)
    assert list(first.test_index) == list(second.test_index)


def test_group_split_never_shares_a_template_group():
    df = _make_frame()

    split = group_split(df, seed=42, n_splits=2)

    train_groups = set(group_labels(df)[split.train_index])
    test_groups = set(group_labels(df)[split.test_index])

    assert train_groups & test_groups == set()


def test_group_split_copies_no_test_row_into_the_training_set():
    df = _make_frame()

    train_df, test_df = split_frame(df, group_split(df, seed=42, n_splits=2))

    train_inputs = {build_text(row) for _, row in train_df.iterrows()}
    test_inputs = {build_text(row) for _, row in test_df.iterrows()}

    assert train_inputs & test_inputs == set()


def test_group_split_keeps_every_label_on_both_sides():
    df = _make_frame()

    split = group_split(df, seed=42, n_splits=2)

    train_labels = set(df["failure_label"].to_numpy()[split.train_index])
    test_labels = set(df["failure_label"].to_numpy()[split.test_index])

    assert train_labels == test_labels
    assert len(train_labels) == 4


def test_group_split_rejects_a_label_it_cannot_measure():
    """A label whose groups all land on one side must not pass silently."""
    df = _make_frame()

    # Collapse one label onto a single template group: with two folds it
    # can only ever appear on one side of the split.
    df = df[~((df["failure_label"] == "invalid_input")
              & (df["tool_name"] == "multiply"))]

    with pytest.raises(ValueError):
        group_split(df, seed=42, n_splits=2)


def test_group_folds_cover_every_row_exactly_once():
    df = _make_frame()

    folds = group_folds(df, seed=42, n_splits=4)

    covered = []
    for _, test_index in folds:
        covered.extend(test_index.tolist())

    assert sorted(covered) == list(range(len(df)))


def test_group_folds_keep_their_templates_apart():
    df = _make_frame()

    groups = group_labels(df)

    for train_index, test_index in group_folds(df, seed=42, n_splits=4):
        assert set(groups[train_index]) & set(groups[test_index]) == set()


def test_group_folds_are_reproducible():
    df = _make_frame()

    first = group_folds(df, seed=42, n_splits=4)
    second = group_folds(df, seed=42, n_splits=4)

    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        assert list(train_a) == list(train_b)
        assert list(test_a) == list(test_b)


def test_label_distribution_counts_the_rows_it_is_given():
    df = _make_frame()

    distribution = label_distribution(df, df.index, "failure_label")

    assert distribution == {
        "invalid_input": 8,
        "missing_argument": 8,
        "none": 8,
        "unknown_tool": 8,
    }


def test_split_frame_returns_the_rows_of_each_side():
    df = _make_frame()

    split = group_split(df, seed=42, n_splits=2)
    train_df, test_df = split_frame(df, split)

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["task_id"]) & set(test_df["task_id"]) == set()
