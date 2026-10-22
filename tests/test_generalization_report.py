"""Tests for the generalization report and the taxonomy contract.

These tests do not need a trained model: they exercise the scoring and
reporting logic that decides which cases count as model errors.
"""

import pandas as pd
import pytest

from src.evaluation import analyze_errors
from src.evaluation.test_generalization import (
    TEST_CASES,
    build_result_row,
    summarize,
)
from src.tracing.labels import (
    LABEL_DEFINITIONS,
    SUPPORTED_LABELS,
    describe,
    is_supported,
)


def test_supported_labels_have_definitions():
    for label in SUPPORTED_LABELS:
        assert label in LABEL_DEFINITIONS
        assert describe(label)


def test_is_supported_rejects_unknown_label():
    assert is_supported("unknown_tool") is True
    assert is_supported("http_error") is False


def test_describe_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown label"):
        describe("http_error")


def test_in_taxonomy_case_is_scored():
    case = {
        "task_id": "case-001",
        "expected_label": "missing_argument",
    }
    prediction = {"predicted_label": "missing_argument"}

    row = build_result_row(case, prediction)

    assert row["scored"] is True
    assert row["correct"] is True
    assert row["expected_label"] == "missing_argument"


def test_in_taxonomy_misprediction_is_scored_and_incorrect():
    case = {
        "task_id": "case-002",
        "expected_label": "invalid_input",
    }
    prediction = {"predicted_label": "unknown_tool"}

    row = build_result_row(case, prediction)

    assert row["scored"] is True
    assert row["correct"] is False


def test_out_of_taxonomy_case_is_not_scored():
    case = {
        "task_id": "case-003",
        "expected_label": None,
        "out_of_taxonomy_reason": "runtime error, no class in the taxonomy",
    }
    prediction = {"predicted_label": "unknown_tool"}

    row = build_result_row(case, prediction)

    assert row["scored"] is False
    assert row["correct"] == ""
    assert row["expected_label"] == ""
    assert row["out_of_taxonomy_reason"].startswith("runtime error")


def test_build_result_row_rejects_label_outside_taxonomy():
    case = {
        "task_id": "case-004",
        "expected_label": "http_error",
    }
    prediction = {"predicted_label": "unknown_tool"}

    with pytest.raises(ValueError, match="not in the trained taxonomy"):
        build_result_row(case, prediction)


def test_summary_excludes_out_of_taxonomy_cases_from_accuracy():
    rows = [
        {"scored": True, "correct": True},
        {"scored": True, "correct": False},
        {"scored": False, "correct": ""},
        {"scored": False, "correct": ""},
    ]

    summary = summarize(rows)

    assert summary["total"] == 4
    assert summary["scored"] == 2
    assert summary["unscored"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["coverage"] == pytest.approx(0.5)


def test_summary_reports_none_when_nothing_is_scored():
    summary = summarize([{"scored": False, "correct": ""}])

    assert summary["accuracy"] is None
    assert summary["coverage"] == pytest.approx(0.0)


def test_test_cases_declare_consistent_taxonomy_metadata():
    for case in TEST_CASES:
        expected = case.get("expected_label")
        reason = case.get("out_of_taxonomy_reason")

        if expected is None:
            assert reason, (
                f"Case {case['task_id']} has no expected label and must "
                "explain why it is outside the trained taxonomy."
            )
        else:
            assert is_supported(expected), (
                f"Case {case['task_id']} expects {expected!r}, which is not "
                "in the trained taxonomy."
            )
            assert not reason


def test_analyze_errors_ignores_out_of_taxonomy_rows(tmp_path, monkeypatch):
    results = tmp_path / "generalization_results.csv"
    errors = tmp_path / "error_analysis.csv"

    pd.DataFrame([
        {
            "task_id": "ok-001",
            "expected_label": "none",
            "predicted_label": "none",
            "correct": True,
            "scored": True,
            "out_of_taxonomy_reason": "",
        },
        {
            "task_id": "bad-001",
            "expected_label": "invalid_input",
            "predicted_label": "unknown_tool",
            "correct": False,
            "scored": True,
            "out_of_taxonomy_reason": "",
        },
        {
            "task_id": "out-001",
            "expected_label": "",
            "predicted_label": "unknown_tool",
            "correct": "",
            "scored": False,
            "out_of_taxonomy_reason": "no label applies",
        },
    ]).to_csv(results, index=False)

    monkeypatch.setattr(analyze_errors, "INPUT_FILE", results)
    monkeypatch.setattr(analyze_errors, "OUTPUT_FILE", errors)

    analyze_errors.main()

    written = pd.read_csv(errors)

    assert list(written["task_id"]) == ["bad-001"]
    assert "error_analysis" in written.columns
