"""Tests for the agent's execution contract and its error reporting."""

import pytest

from src.agents.basic_agent import run_agent


def test_successful_execution(tmp_path):
    trace = run_agent(
        {
            "task_id": "t-001",
            "tool": "add",
            "arguments": {"a": 10, "b": 5},
        },
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.status == "success"
    assert trace.result == 15
    assert trace.error is None
    assert trace.error_type is None
    assert trace.failure_label == "none"


def test_unknown_tool(tmp_path):
    trace = run_agent(
        {"task_id": "t-002", "tool": "divide", "arguments": {"a": 10, "b": 2}},
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.status == "failure"
    assert trace.result is None
    assert trace.error == "Unknown tool: divide"
    assert trace.error_type == "ValueError"
    assert trace.failure_label == "unknown_tool"


def test_missing_argument_keeps_the_historical_error_text(tmp_path):
    trace = run_agent(
        {"task_id": "t-003", "tool": "add", "arguments": {"a": 10}},
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.status == "failure"
    assert trace.error == "'b'"
    assert trace.error_type == "KeyError"
    assert trace.failure_label == "missing_argument"


def test_invalid_input_keeps_the_historical_error_text(tmp_path):
    trace = run_agent(
        {
            "task_id": "t-004",
            "tool": "add",
            "arguments": {"a": "hello", "b": 5},
        },
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.status == "failure"
    assert trace.error == "Arguments a and b must be numbers"
    assert trace.error_type == "ValueError"
    assert trace.failure_label == "invalid_input"


def test_boolean_is_not_a_number(tmp_path):
    trace = run_agent(
        {"task_id": "t-005", "tool": "add", "arguments": {"a": True, "b": 5}},
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.failure_label == "invalid_input"


@pytest.mark.parametrize(
    "arguments",
    [None, ["a", "b"], "notadict", 42, 3.5],
)
def test_malformed_arguments_produce_a_failure_not_a_crash(
    arguments, tmp_path
):
    trace = run_agent(
        {"task_id": "t-006", "tool": "add", "arguments": arguments},
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.status == "failure"
    assert trace.result is None
    assert trace.error_type == "TypeError"
    assert trace.failure_label == "invalid_input"
    assert "must be a mapping" in trace.error
    assert type(arguments).__name__ in trace.error


def test_missing_arguments_key_is_treated_as_an_empty_mapping(tmp_path):
    trace = run_agent(
        {"task_id": "t-007", "tool": "add"},
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.status == "failure"
    assert trace.failure_label == "missing_argument"
    assert trace.error == "'a'"


def test_non_mapping_task_is_a_caller_error(tmp_path):
    with pytest.raises(TypeError, match="Task must be a mapping"):
        run_agent(["not", "a", "dict"], trace_file=tmp_path / "traces.jsonl")


def test_missing_identity_fields_use_defaults(tmp_path):
    trace = run_agent(
        {"tool": "add", "arguments": {"a": 1, "b": 2}},
        trace_file=tmp_path / "traces.jsonl",
    )

    assert trace.task_id == "unknown"
    assert trace.agent_type == "calculator"


def test_trace_is_written_to_the_given_file(tmp_path):
    path = tmp_path / "traces.jsonl"

    run_agent(
        {"task_id": "t-008", "tool": "add", "arguments": {"a": 1, "b": 2}},
        trace_file=path,
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 1
    assert '"failure_label":"none"' in lines[0]
