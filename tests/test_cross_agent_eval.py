"""Tests for the labelling rules and the cross-agent evaluation.

The rules are the part that must not drift: a label is derived from where
the call failed, never from the exception class. These tests pin the
rules down, check the dataset against them, and run the evaluation
workflow end to end on a small in-memory dataset so the scoring and
reporting can be checked without a trained model.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.diagnosis import training
from src.evaluation import evaluate_cross_agent as cross_agent
from src.evaluation.build_cross_agent_dataset import (
    COLUMNS,
    DOMAINS,
    EVALUATION_CASES,
    OUTPUT_FILE as EVAL_FILE,
    PROVENANCE,
    validate_cases,
)
from src.evaluation.labelling_rules import (
    DETAIL_KINDS,
    FAILURE_STAGES,
    RULE_TABLE,
    audit_cases,
    decide_label,
    describe_rule,
    naive_label_for,
)
from src.evaluation.test_generalization import TEST_CASES
from src.tracing.labels import SUPPORTED_LABELS, is_supported


def _write_dataset(path: Path) -> None:
    """Write a small cross-agent dataset with two unlabelled rows."""
    rows = [
        {
            "task_id": "t-001",
            "agent_type": "calculator",
            "tool_name": "add",
            "status": "success",
            "error_type": "",
            "error": "",
            "failure_stage": "success",
            "detail_kind": "success",
            "failure_label": "none",
            "in_taxonomy": True,
            "out_of_taxonomy_reason": "",
            "domain": "calculator",
            "case_kind": "in_domain_transfer",
            "provenance": PROVENANCE,
        },
        {
            "task_id": "t-002",
            "agent_type": "calculator",
            "tool_name": "power",
            "status": "failure",
            "error_type": "ValueError",
            "error": "Unknown tool: power",
            "failure_stage": "call_validation",
            "detail_kind": "tool_not_registered",
            "failure_label": "unknown_tool",
            "in_taxonomy": True,
            "out_of_taxonomy_reason": "",
            "domain": "calculator",
            "case_kind": "in_domain_transfer",
            "provenance": PROVENANCE,
        },
        {
            "task_id": "t-003",
            "agent_type": "calculator",
            "tool_name": "add",
            "status": "failure",
            "error_type": "KeyError",
            "error": "'b'",
            "failure_stage": "call_validation",
            "detail_kind": "argument_absent",
            "failure_label": "missing_argument",
            "in_taxonomy": True,
            "out_of_taxonomy_reason": "",
            "domain": "calculator",
            "case_kind": "in_domain_transfer",
            "provenance": PROVENANCE,
        },
        {
            "task_id": "t-004",
            "agent_type": "calculator",
            "tool_name": "multiply",
            "status": "failure",
            "error_type": "ValueError",
            "error": "Arguments a and b must be numbers",
            "failure_stage": "call_validation",
            "detail_kind": "argument_unusable",
            "failure_label": "invalid_input",
            "in_taxonomy": True,
            "out_of_taxonomy_reason": "",
            "domain": "calculator",
            "case_kind": "new_agent",
            "provenance": PROVENANCE,
        },
        {
            "task_id": "t-005",
            "agent_type": "web",
            "tool_name": "fetch_page",
            "status": "failure",
            "error_type": "HTTPError",
            "error": "404 Client Error",
            "failure_stage": "tool_execution",
            "detail_kind": "execution_error",
            "failure_label": "",
            "in_taxonomy": False,
            "out_of_taxonomy_reason": "the tool ran and the page was missing",
            "domain": "web",
            "case_kind": "out_of_taxonomy",
            "provenance": PROVENANCE,
        },
        {
            "task_id": "t-006",
            "agent_type": "python",
            "tool_name": "execute_code",
            "status": "failure",
            "error_type": "NameError",
            "error": "name 'total' is not defined",
            "failure_stage": "tool_execution",
            "detail_kind": "execution_error",
            "failure_label": "",
            "in_taxonomy": False,
            "out_of_taxonomy_reason": "generated code raised while running",
            "domain": "python_execution",
            "case_kind": "out_of_taxonomy",
            "provenance": PROVENANCE,
        },
    ]

    pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(path, index=False)


def _write_training_frame(path: Path) -> None:
    rows = [
        {
            "task_id": "train-001",
            "agent_type": "calculator",
            "tool_name": "add",
            "status": "success",
            "error_type": "",
            "error": "",
            "failure_label": "none",
        },
        {
            "task_id": "train-002",
            "agent_type": "calculator",
            "tool_name": "power",
            "status": "failure",
            "error_type": "ValueError",
            "error": "Unknown tool: power",
            "failure_label": "unknown_tool",
        },
        {
            "task_id": "train-003",
            "agent_type": "calculator",
            "tool_name": "add",
            "status": "failure",
            "error_type": "KeyError",
            "error": "'b'",
            "failure_label": "missing_argument",
        },
        {
            "task_id": "train-004",
            "agent_type": "calculator",
            "tool_name": "multiply",
            "status": "failure",
            "error_type": "ValueError",
            "error": "Arguments a and b must be numbers",
            "failure_label": "invalid_input",
        },
    ]

    pd.DataFrame(rows).to_csv(path, index=False)


def test_every_rule_pair_has_a_statement_and_a_label():
    for pair, label in RULE_TABLE.items():
        stage, kind = pair
        assert stage in FAILURE_STAGES
        assert kind in DETAIL_KINDS
        assert label is None or is_supported(label)
        assert describe_rule(stage, kind)


def test_an_illegal_stage_kind_pair_is_rejected():
    with pytest.raises(ValueError, match="No labelling rule"):
        decide_label("call_validation", "execution_error")

    with pytest.raises(ValueError, match="No labelling rule"):
        decide_label("tool_execution", "argument_absent")


def test_execution_errors_have_no_label():
    assert decide_label("tool_execution", "execution_error") is None


def test_the_stage_decides_the_label_not_the_exception_type():
    """The same exception can be in or out of the taxonomy."""
    assert naive_label_for("NameError", "failure") == "invalid_input"
    assert decide_label("tool_execution", "execution_error") is None
    assert decide_label("call_validation", "argument_unusable") == (
        "invalid_input"
    )


def test_the_dataset_labels_are_derived_from_the_rules():
    for case in EVALUATION_CASES:
        expected = decide_label(case["failure_stage"], case["detail_kind"])
        declared = case["failure_label"] or None

        assert declared == expected, case["task_id"]


def test_the_dataset_covers_every_domain_and_label():
    assert {case["domain"] for case in EVALUATION_CASES} == set(DOMAINS)

    labelled = {
        case["failure_label"] for case in EVALUATION_CASES if case["failure_label"]
    }

    assert labelled == set(SUPPORTED_LABELS)


def test_the_dataset_is_hand_authored_simulation():
    for case in EVALUATION_CASES:
        assert case["provenance"] == PROVENANCE
        assert case["provenance"] != "captured"


def test_unlabelled_cases_explain_themselves():
    for case in EVALUATION_CASES:
        if case["failure_label"]:
            assert case["out_of_taxonomy_reason"] == ""
        else:
            assert case["out_of_taxonomy_reason"]


def test_validate_cases_rejects_a_hand_written_label():
    broken = dict(EVALUATION_CASES[0])
    broken["failure_label"] = "unknown_tool"

    with pytest.raises(ValueError, match="contradicts the labelling rule"):
        validate_cases([broken])


def test_validate_cases_rejects_a_duplicate_task_id():
    duplicate = [EVALUATION_CASES[0], dict(EVALUATION_CASES[1])]

    duplicate[1]["task_id"] = duplicate[0]["task_id"]

    with pytest.raises(ValueError, match="Duplicate task id"):
        validate_cases(duplicate)


def test_validate_cases_rejects_a_status_label_mismatch():
    broken = dict(EVALUATION_CASES[0])
    broken["status"] = "failure"

    with pytest.raises(ValueError, match="contradicts status"):
        validate_cases([broken])


def test_the_written_dataset_matches_the_builder():
    """A stale CSV would make the report describe cases that moved."""
    if not EVAL_FILE.exists():
        pytest.skip("dataset has not been generated")

    written = pd.read_csv(EVAL_FILE, keep_default_na=False)

    assert list(written.columns) == list(COLUMNS)
    assert list(written["task_id"]) == [
        case["task_id"] for case in EVALUATION_CASES
    ]
    assert list(written["failure_label"]) == [
        case["failure_label"] for case in EVALUATION_CASES
    ]


def test_the_generalization_cases_agree_with_the_rules():
    for row in audit_cases(TEST_CASES):
        assert row["rules_agree_with_declared"], row
        assert row["failure_stage"] in FAILURE_STAGES


def test_the_exception_type_mapping_mislabels_the_two_known_cases():
    """This is the audit's finding, kept as a regression test."""
    rows = {row["task_id"]: row for row in audit_cases(TEST_CASES)}

    assert rows["gen-005"]["exception_type_would_say"] == "invalid_input"
    assert rows["gen-005"]["rules_say"] == ""
    assert rows["gen-006"]["exception_type_would_say"] == "unknown_tool"
    assert rows["gen-006"]["rules_say"] == ""


def test_the_training_path_never_reads_the_cross_agent_dataset():
    """The evaluation set must stay out of training."""
    source = Path(training.__file__).read_text(encoding="utf-8")

    assert "cross_agent" not in source
    assert str(training.DATA_FILE) != str(EVAL_FILE)


def test_the_evaluation_workflow_writes_every_report(tmp_path, monkeypatch):
    eval_file = tmp_path / "eval.csv"
    train_file = tmp_path / "train.csv"
    reports = tmp_path / "reports"

    _write_dataset(eval_file)
    _write_training_frame(train_file)

    def fake_diagnose(trace, model="text"):
        # Deliberately wrong for invalid_input, so there is an error to
        # analyse.
        label = {
            "ValueError": "unknown_tool",
            "KeyError": "missing_argument",
            "": "none",
        }.get(trace["error_type"], "none")

        return {
            "task_id": trace["task_id"],
            "predicted_label": label,
            "class_scores": {name: 0.25 for name in SUPPORTED_LABELS},
        }

    monkeypatch.setattr(cross_agent, "EVAL_FILE", eval_file)
    monkeypatch.setattr(cross_agent, "TRAIN_FILE", train_file)
    monkeypatch.setattr(cross_agent, "REPORTS_DIR", reports)
    monkeypatch.setattr(cross_agent, "REPORT_FILE", reports / "report.json")
    monkeypatch.setattr(
        cross_agent, "PREDICTIONS_FILE", reports / "predictions.csv"
    )
    monkeypatch.setattr(cross_agent, "ERRORS_FILE", reports / "errors.csv")
    monkeypatch.setattr(cross_agent, "COVERAGE_FILE", reports / "coverage.csv")
    monkeypatch.setattr(
        cross_agent, "CONFUSION_FILE", reports / "confusion.csv"
    )
    monkeypatch.setattr(cross_agent, "AUDIT_FILE", reports / "audit.csv")
    monkeypatch.setattr(cross_agent, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(cross_agent, "diagnose_trace", fake_diagnose)

    report = cross_agent.evaluate()

    assert report["dataset"]["n_cases"] == 6
    assert report["dataset"]["scored"] == 4
    assert report["dataset"]["unscored"] == 2

    # Accuracy is over the four labelled cases only: three are right.
    assert report["metrics"]["accuracy"] == pytest.approx(0.75)
    assert report["uncertainty"]["n_cases"] == 4

    assert len(report["error_analysis"]) == 1
    assert report["error_analysis"][0]["task_id"] == "t-004"

    assert report["out_of_taxonomy"]["n_cases"] == 2
    assert report["model"]["retrained_for_this_evaluation"] is False
    assert report["dataset"]["is_captured_agent_behaviour"] is False

    # The fresh dataset is not used for training, so no training row is a
    # copy: only the rows that literally match the training frame count.
    coverage = report["coverage"]["totals"]
    assert coverage["exact_input_seen_in_training"] == 4

    for path in (
        reports / "report.json",
        reports / "predictions.csv",
        reports / "errors.csv",
        reports / "coverage.csv",
        reports / "confusion.csv",
        reports / "audit.csv",
    ):
        assert path.exists(), path

    written = json.loads((reports / "report.json").read_text())
    assert written["metrics"]["accuracy"] == pytest.approx(0.75)


def test_unlabelled_rows_are_never_counted_as_errors(tmp_path, monkeypatch):
    eval_file = tmp_path / "eval.csv"
    train_file = tmp_path / "train.csv"
    reports = tmp_path / "reports"

    _write_dataset(eval_file)
    _write_training_frame(train_file)

    def always_wrong(trace, model="text"):
        return {
            "task_id": trace["task_id"],
            "predicted_label": "unknown_tool",
            "class_scores": {name: 0.25 for name in SUPPORTED_LABELS},
        }

    monkeypatch.setattr(cross_agent, "EVAL_FILE", eval_file)
    monkeypatch.setattr(cross_agent, "TRAIN_FILE", train_file)
    monkeypatch.setattr(cross_agent, "REPORTS_DIR", reports)
    monkeypatch.setattr(cross_agent, "REPORT_FILE", reports / "report.json")
    monkeypatch.setattr(
        cross_agent, "PREDICTIONS_FILE", reports / "predictions.csv"
    )
    monkeypatch.setattr(cross_agent, "ERRORS_FILE", reports / "errors.csv")
    monkeypatch.setattr(cross_agent, "COVERAGE_FILE", reports / "coverage.csv")
    monkeypatch.setattr(
        cross_agent, "CONFUSION_FILE", reports / "confusion.csv"
    )
    monkeypatch.setattr(cross_agent, "AUDIT_FILE", reports / "audit.csv")
    monkeypatch.setattr(cross_agent, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(cross_agent, "diagnose_trace", always_wrong)

    report = cross_agent.evaluate()

    # The two unlabelled rows were predicted wrong too, but they must not
    # appear as errors: there is no correct label to be wrong about.
    assert len(report["error_analysis"]) == 3
    assert {
        row["task_id"] for row in report["error_analysis"]
    } == {"t-001", "t-003", "t-004"}

    # t-002 happened to match the constant prediction.
    assert report["metrics"]["accuracy"] == pytest.approx(0.25)


def test_a_dataset_without_labels_is_rejected(tmp_path, monkeypatch):
    if not EVAL_FILE.exists():
        pytest.skip("dataset has not been generated")

    frame = pd.read_csv(EVAL_FILE, keep_default_na=False)
    frame["failure_label"] = ""

    eval_file = tmp_path / "eval.csv"
    train_file = tmp_path / "train.csv"
    frame.to_csv(eval_file, index=False)
    _write_training_frame(train_file)

    monkeypatch.setattr(cross_agent, "EVAL_FILE", eval_file)
    monkeypatch.setattr(cross_agent, "TRAIN_FILE", train_file)
    monkeypatch.setattr(cross_agent, "load_model", lambda *a, **k: None)

    with pytest.raises(ValueError, match="no labelled cases"):
        cross_agent.evaluate()
