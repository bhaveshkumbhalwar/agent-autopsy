"""Tests for the dataset contract enforced when preparing the data."""

import json

import pandas as pd
import pytest

from src.evaluation import prepare_dataset
from src.tracing.labels import (
    ALLOWED_STATUSES,
    SUPPORTED_LABELS,
    describe,
    is_supported,
    normalize_label,
    validate_labels,
)


def make_row(task_id="t-001", **overrides):
    row = {
        "task_id": task_id,
        "agent_type": "calculator",
        "tool_name": "add",
        "arguments": json.dumps({"a": 1, "b": 2}),
        "status": "success",
        "error_type": "",
        "error": "",
        "duration_ms": 0.1,
        "failure_label": "none",
    }
    row.update(overrides)
    return row


def write_dataset(tmp_path, rows):
    source = tmp_path / "failure_dataset.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source


def prepare(tmp_path, rows, monkeypatch):
    source = write_dataset(tmp_path, rows)
    target = tmp_path / "ml_dataset.csv"

    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(prepare_dataset, "OUTPUT_FILE", target)

    prepare_dataset.prepare_dataset()

    return pd.read_csv(target)


# --- vocabulary ------------------------------------------------------------


def test_supported_labels_have_definitions():
    for label in SUPPORTED_LABELS:
        assert label in SUPPORTED_LABELS
        assert describe(label)


def test_allowed_statuses():
    assert ALLOWED_STATUSES == ("success", "failure")


def test_normalize_label_treats_blank_as_not_determined():
    assert normalize_label(None) is None
    assert normalize_label("") is None
    assert normalize_label("  ") is None
    assert normalize_label(" none ") == "none"


def test_normalize_label_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="Invalid failure label"):
        normalize_label("http_error")


def test_validate_labels_rejects_a_missing_label():
    with pytest.raises(ValueError, match="missing failure label"):
        validate_labels(["none", ""], ["t-001", "t-002"])


def test_validate_labels_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="invalid failure label"):
        validate_labels(["none", "http_error"], ["t-001", "t-002"])


def test_validate_labels_names_the_offending_row():
    with pytest.raises(ValueError, match="row t-002"):
        validate_labels(["none", "nope"], ["t-001", "t-002"])


def test_validate_labels_accepts_a_valid_set():
    validate_labels(["none", "unknown_tool"], ["t-001", "t-002"])


# --- preparation -----------------------------------------------------------


def test_preparation_preserves_the_important_columns(tmp_path, monkeypatch):
    result = prepare(
        tmp_path,
        [make_row("t-001"), make_row("t-002", tool_name="multiply")],
        monkeypatch,
    )

    for column in prepare_dataset.OUTPUT_COLUMNS:
        assert column in result.columns

    assert list(result["task_id"]) == ["t-001", "t-002"]
    assert list(result["tool_name"]) == ["add", "multiply"]
    assert json.loads(result["arguments"][0]) == {"a": 1, "b": 2}


def test_preparation_keeps_the_target_label(tmp_path, monkeypatch):
    result = prepare(
        tmp_path,
        [
            make_row("t-001"),
            make_row(
                "t-002",
                status="failure",
                error="Unknown tool: divide",
                error_type="ValueError",
                failure_label="unknown_tool",
            ),
        ],
        monkeypatch,
    )

    assert list(result["failure_label"]) == ["none", "unknown_tool"]


def test_preparation_rejects_an_invalid_label(tmp_path, monkeypatch):
    source = write_dataset(
        tmp_path, [make_row("t-001", failure_label="http_error")]
    )
    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(
        prepare_dataset, "OUTPUT_FILE", tmp_path / "ml_dataset.csv"
    )

    with pytest.raises(ValueError, match="invalid failure_label"):
        prepare_dataset.prepare_dataset()


def test_preparation_rejects_a_missing_label(tmp_path, monkeypatch):
    source = write_dataset(tmp_path, [make_row("t-001", failure_label="")])
    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(
        prepare_dataset, "OUTPUT_FILE", tmp_path / "ml_dataset.csv"
    )

    with pytest.raises(ValueError, match="missing failure_label"):
        prepare_dataset.prepare_dataset()


def test_preparation_does_not_silently_drop_an_invalid_row(
    tmp_path, monkeypatch
):
    source = write_dataset(
        tmp_path,
        [make_row("t-001"), make_row("t-002", failure_label="http_error")],
    )
    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(
        prepare_dataset, "OUTPUT_FILE", tmp_path / "ml_dataset.csv"
    )

    with pytest.raises(ValueError):
        prepare_dataset.prepare_dataset()

    # The bad row must not have been quietly kept or dropped.
    assert not (tmp_path / "ml_dataset.csv").exists()


def test_preparation_rejects_a_label_that_contradicts_the_status(
    tmp_path, monkeypatch
):
    source = write_dataset(
        tmp_path,
        [make_row("t-001", status="success", failure_label="unknown_tool")],
    )
    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(
        prepare_dataset, "OUTPUT_FILE", tmp_path / "ml_dataset.csv"
    )

    with pytest.raises(ValueError, match="disagrees with status"):
        prepare_dataset.prepare_dataset()


def test_preparation_rejects_an_invalid_status(tmp_path, monkeypatch):
    source = write_dataset(tmp_path, [make_row("t-001", status="finished")])
    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(
        prepare_dataset, "OUTPUT_FILE", tmp_path / "ml_dataset.csv"
    )

    with pytest.raises(ValueError, match="invalid status"):
        prepare_dataset.prepare_dataset()


def test_preparation_rejects_a_missing_required_column(tmp_path, monkeypatch):
    source = tmp_path / "failure_dataset.csv"
    pd.DataFrame([make_row()]).drop(columns=["arguments"]).to_csv(
        source, index=False
    )
    monkeypatch.setattr(prepare_dataset, "INPUT_FILE", source)
    monkeypatch.setattr(
        prepare_dataset, "OUTPUT_FILE", tmp_path / "ml_dataset.csv"
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_dataset.prepare_dataset()


def test_preparation_reports_a_missing_source_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare_dataset, "INPUT_FILE", tmp_path / "does_not_exist.csv"
    )

    with pytest.raises(FileNotFoundError):
        prepare_dataset.prepare_dataset()
