"""Tests for the independent evaluation dataset.

The independent dataset is the project's final test set. It is only
meaningful if its labels are valid, its cases are distinct, and it does
not overlap the training data.
"""

import csv
from pathlib import Path

import pytest

from src.evaluation.build_eval_dataset import (
    CASE_KINDS,
    COLUMNS,
    EVALUATION_CASES,
    OUTPUT_FILE,
    PROVENANCE,
    validate_cases,
)
from src.tracing.labels import SUPPORTED_LABELS


def _read_rows() -> list:
    if not OUTPUT_FILE.exists():
        pytest.skip(
            f"{OUTPUT_FILE} has not been generated; run "
            "python -m src.evaluation.build_eval_dataset"
        )

    with OUTPUT_FILE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_dataset_has_been_generated():
    rows = _read_rows()

    assert len(rows) == len(EVALUATION_CASES)


def test_the_dataset_has_the_documented_columns():
    rows = _read_rows()

    assert list(rows[0].keys()) == COLUMNS


def test_every_label_is_inside_the_taxonomy():
    rows = _read_rows()

    for row in rows:
        assert row["failure_label"] in SUPPORTED_LABELS


def test_the_label_agrees_with_the_status():
    rows = _read_rows()

    for row in rows:
        if row["failure_label"] == "none":
            assert row["status"] == "success"
        else:
            assert row["status"] == "failure"


def test_the_dataset_is_balanced_across_the_labels():
    rows = _read_rows()

    counts = {}
    for row in rows:
        counts[row["failure_label"]] = counts.get(row["failure_label"], 0) + 1

    assert len(set(counts.values())) == 1
    assert set(counts) == set(SUPPORTED_LABELS)


def test_every_case_states_its_stress_and_its_provenance():
    rows = _read_rows()

    for row in rows:
        assert row["case_kind"] in CASE_KINDS
        assert row["provenance"] == PROVENANCE
        assert row["label_rationale"].strip()


def test_the_task_ids_are_unique():
    rows = _read_rows()

    task_ids = [row["task_id"] for row in rows]

    assert len(set(task_ids)) == len(task_ids)


def test_the_dataset_does_not_overlap_the_training_data():
    training_file = Path("data/processed/ml_dataset.csv")

    if not training_file.exists():
        pytest.skip(
            "The training dataset has not been generated; run "
            "python -m src.evaluation.prepare_dataset"
        )

    import pandas as pd

    from src.diagnosis.features import build_text

    training = pd.read_csv(training_file)
    independent = pd.read_csv(OUTPUT_FILE)

    assert set(independent["task_id"]) & set(training["task_id"]) == set()

    # The task ids alone would not catch a case that repeats a training
    # template, so the model inputs are compared as well.
    training_inputs = {
        build_text(row) for _, row in training.iterrows()
    }
    independent_inputs = [
        build_text(row) for _, row in independent.iterrows()
    ]

    shared = [text for text in independent_inputs if text in training_inputs]

    # Only the deliberate "seen template" controls may repeat a training
    # input; every other case must be new.
    for text in shared:
        row = independent[
            independent.apply(
                lambda r: build_text(r) == text, axis=1
            )
        ].iloc[0]
        assert row["case_kind"] == "seen_template"


def test_validate_cases_rejects_a_label_outside_the_taxonomy():
    cases = [dict(EVALUATION_CASES[0])]
    cases[0]["failure_label"] = "not_a_label"

    with pytest.raises(ValueError):
        validate_cases(cases)


def test_validate_cases_rejects_a_label_that_contradicts_the_status():
    cases = [dict(EVALUATION_CASES[0])]
    cases[0]["status"] = "failure"

    with pytest.raises(ValueError):
        validate_cases(cases)


def test_validate_cases_rejects_a_duplicate_task_id():
    cases = [dict(EVALUATION_CASES[0]), dict(EVALUATION_CASES[0])]

    with pytest.raises(ValueError):
        validate_cases(cases)


def test_validate_cases_rejects_an_unknown_case_kind():
    cases = [dict(EVALUATION_CASES[0])]
    cases[0]["case_kind"] = "invented"

    with pytest.raises(ValueError):
        validate_cases(cases)
