"""Tests for the model comparison and its uncertainty reporting.

The comparison is only useful if it can say "no difference". These tests
pin down the statistics that let it do so: the exact McNemar test on the
cases the two models disagree about, and the bootstrap interval that says
how much a small evaluation set can support.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.diagnosis.compare_models import (
    _changed_task_ids,
    _delta,
    structured_feature_diagnostics,
)
from src.evaluation.metrics import (
    bootstrap_accuracy_interval,
    compute_metrics,
    mcnemar_exact_test,
)


REPORT_FILE = Path("reports/model_comparison.json")


def _result(truth, predictions):
    """Build the part of a comparison result the delta needs."""
    return {
        "metrics": compute_metrics(
            truth, predictions, labels=sorted(set(truth))
        ),
        "predictions": [
            {
                "task_id": f"t-{index}",
                "expected_label": true,
                "predicted_label": predicted,
                "correct": true == predicted,
            }
            for index, (true, predicted) in enumerate(
                zip(truth, predictions)
            )
        ],
    }


def test_mcnemar_ignores_the_cases_both_models_get_right():
    truth = ["a", "a", "b", "b"]
    first = ["a", "a", "b", "b"]
    second = ["a", "a", "b", "b"]

    result = mcnemar_exact_test(truth, first, second)

    assert result["discordant"] == 0
    assert result["exact_p_value"] == 1.0


def test_mcnemar_counts_only_the_disagreements():
    truth = ["a", "a", "b", "b"]
    first = ["a", "b", "b", "b"]
    second = ["a", "a", "b", "a"]

    result = mcnemar_exact_test(truth, first, second)

    # t-1 is fixed by the second model; t-3 is broken by it.
    assert result["only_first_correct"] == 1
    assert result["only_second_correct"] == 1
    assert result["discordant"] == 2


def test_mcnemar_cannot_show_a_difference_from_two_flips():
    """Two flipped cases is not evidence, and the p-value must say so."""
    truth = ["a", "a"] + ["b"] * 10
    first = ["b", "b"] + ["b"] * 10
    second = ["a", "a"] + ["b"] * 10

    result = mcnemar_exact_test(truth, first, second)

    assert result["only_second_correct"] == 2
    assert result["only_first_correct"] == 0
    assert result["exact_p_value"] == pytest.approx(0.5)


def test_mcnemar_reports_a_small_p_value_for_a_lopsided_split():
    truth = ["a"] * 10
    first = ["b"] * 10
    second = ["a"] * 10

    result = mcnemar_exact_test(truth, first, second)

    assert result["discordant"] == 10
    assert result["exact_p_value"] == pytest.approx(2 / 1024)


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact_test(["a", "b"], ["a"], ["a", "b"])


def test_the_bootstrap_interval_brackets_the_accuracy():
    truth = ["a"] * 30 + ["b"] * 18
    predicted = ["a"] * 25 + ["b"] * 5 + ["b"] * 12 + ["a"] * 6

    interval = bootstrap_accuracy_interval(truth, predicted)

    assert interval["n_cases"] == 48
    assert interval["ci_low"] <= interval["accuracy"] <= interval["ci_high"]
    assert interval["accuracy"] == pytest.approx(37 / 48)


def test_the_bootstrap_interval_is_reproducible():
    truth = ["a", "b", "a", "b"] * 10
    predicted = ["a", "b", "b", "a"] * 10

    first = bootstrap_accuracy_interval(truth, predicted, seed=7)
    second = bootstrap_accuracy_interval(truth, predicted, seed=7)

    assert first == second


def test_the_bootstrap_interval_is_wide_on_a_small_sample():
    """A 48-case set cannot pin accuracy down to a percentage point."""
    truth = ["a"] * 24 + ["b"] * 24
    predicted = ["a"] * 24 + ["a"] * 24

    interval = bootstrap_accuracy_interval(truth, predicted)

    assert interval["ci_high"] - interval["ci_low"] > 0.2


def test_the_bootstrap_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        bootstrap_accuracy_interval([], [])


def test_structured_diagnostics_flag_a_constant_feature():
    df = pd.DataFrame({
        "agent_type": ["calculator"] * 4,
        "tool_name": ["add"] * 4,
        "status": ["success"] * 4,
        "error_type": [""] * 4,
        "error": [""] * 4,
        "arguments": ['{"a": 1}'] * 4,
        "failure_label": ["none"] * 4,
    })

    diagnostics = {
        row["feature"]: row for row in structured_feature_diagnostics(df)
    }

    assert diagnostics["has_arguments"]["always_fires"] is True
    assert diagnostics["has_error"]["never_fires"] is True


def test_structured_diagnostics_flag_a_label_copy():
    """A feature that fires for one label only is a copy of the label."""
    df = pd.DataFrame({
        "agent_type": ["calculator"] * 4,
        "tool_name": ["add", "add", "divide", "divide"],
        "status": ["failure"] * 4,
        "error_type": ["KeyError", "KeyError", "ValueError", "ValueError"],
        "error": ["'b'", "'b'", "Unknown tool: x", "Unknown tool: y"],
        "arguments": ['{"a": 1}'] * 4,
        "failure_label": [
            "missing_argument", "missing_argument",
            "unknown_tool", "unknown_tool",
        ],
    })

    diagnostics = {
        row["feature"]: row for row in structured_feature_diagnostics(df)
    }

    missing = diagnostics["has_missing_argument_error"]

    assert missing["n_true"] == 2
    assert missing["labels_when_true"] == {"missing_argument": 2}
    assert missing["distinct_labels_when_true"] == 1


def test_changed_task_ids_lists_only_the_flipped_rows():
    truth = ["a", "a", "b"]
    baseline = _result(truth, ["a", "b", "b"])
    combined = _result(truth, ["a", "a", "b"])

    assert _changed_task_ids(baseline, combined) == ["t-1"]


def test_changed_task_ids_rejects_results_from_different_rows():
    truth = ["a", "b"]
    baseline = _result(truth, ["a", "b"])
    combined = _result(truth, ["a", "b"])

    combined["predictions"][0]["task_id"] = "other-row"

    with pytest.raises(ValueError, match="different rows"):
        _changed_task_ids(baseline, combined)


def test_the_delta_reports_the_improvement_and_its_uncertainty():
    truth = ["a"] * 8 + ["b"] * 2
    baseline = _result(truth, ["b"] * 10)
    combined = _result(truth, ["a"] * 8 + ["b"] * 2)

    changed = [f"t-{index}" for index in range(8)]

    delta = _delta(baseline, combined, changed, truth)

    assert delta["baseline_accuracy"] == pytest.approx(0.2)
    assert delta["combined_accuracy"] == 1.0
    assert delta["delta_accuracy"] == pytest.approx(0.8)
    assert delta["n_rows_where_prediction_changed"] == 8

    uncertainty = delta["uncertainty"]
    assert uncertainty["mcnemar"]["only_second_correct"] == 8
    assert uncertainty["mcnemar"]["only_first_correct"] == 0
    assert uncertainty["combined"]["n_cases"] == 10


def test_the_delta_reports_no_change_honestly():
    truth = ["a", "b", "a", "b"]
    same = _result(truth, truth)

    delta = _delta(same, same, [], truth)

    assert delta["delta_accuracy"] == 0.0
    assert delta["delta_macro_f1"] == 0.0
    assert delta["n_rows_where_prediction_changed"] == 0
    assert delta["uncertainty"]["mcnemar"]["exact_p_value"] == 1.0


@pytest.mark.skipif(
    not REPORT_FILE.exists(),
    reason="reports/model_comparison.json has not been generated",
)
def test_the_written_report_has_the_documented_shape():
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    for key in (
        "generated_at",
        "seed",
        "dataset",
        "feature_configuration",
        "structured_feature_diagnostics",
        "structured_feature_coefficients",
        "models",
        "comparison",
    ):
        assert key in report

    assert set(report["models"]) == {"text", "combined"}

    for split in (
        "group_cross_validation",
        "stratified_random_split",
        "independent_evaluation_set",
    ):
        assert split in report["comparison"]
        summary = report["comparison"][split]
        assert "uncertainty" in summary
        assert summary["uncertainty"]["text"]["n_cases"] == summary[
            "uncertainty"]["combined"
        ]["n_cases"]
