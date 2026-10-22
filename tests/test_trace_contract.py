"""Tests for the trace contract enforced by ``AgentTrace``."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.tracing.trace_schema import AgentTrace


def make_trace(**overrides) -> AgentTrace:
    """Build a valid trace, with the given fields overridden."""
    now = datetime.now(timezone.utc)

    data = {
        "task_id": "t-001",
        "agent_type": "calculator",
        "tool_name": "add",
        "arguments": {"a": 1, "b": 2},
        "status": "success",
        "result": 3,
        "error": None,
        "error_type": None,
        "failure_label": "none",
        "started_at": now,
        "finished_at": now,
        "duration_ms": 0.0,
    }
    data.update(overrides)

    return AgentTrace(**data)


def test_valid_success_trace():
    trace = make_trace()

    assert trace.status == "success"
    assert trace.result == 3
    assert trace.error is None
    assert trace.error_type is None
    assert trace.failure_label == "none"


def test_valid_failure_trace():
    trace = make_trace(
        status="failure",
        result=None,
        error="Unknown tool: divide",
        error_type="ValueError",
        failure_label="unknown_tool",
    )

    assert trace.status == "failure"
    assert trace.result is None
    assert trace.error == "Unknown tool: divide"
    assert trace.failure_label == "unknown_tool"


@pytest.mark.parametrize(
    "missing",
    ["task_id", "agent_type", "tool_name", "status", "started_at",
     "finished_at", "duration_ms"],
)
def test_missing_required_field_is_rejected(missing):
    now = datetime.now(timezone.utc)
    data = {
        "task_id": "t-001",
        "agent_type": "calculator",
        "tool_name": "add",
        "arguments": {},
        "status": "success",
        "result": 1,
        "error": None,
        "error_type": None,
        "failure_label": "none",
        "started_at": now,
        "finished_at": now,
        "duration_ms": 0.0,
    }
    data.pop(missing)

    with pytest.raises(ValidationError):
        AgentTrace(**data)


def test_invalid_status_is_rejected():
    with pytest.raises(ValidationError):
        make_trace(status="ok")


def test_success_with_error_is_rejected():
    with pytest.raises(ValidationError, match="must not carry error"):
        make_trace(status="success", error="boom", error_type="ValueError")


def test_success_with_error_type_is_rejected():
    with pytest.raises(ValidationError, match="must not carry error"):
        make_trace(status="success", error_type="ValueError")


def test_failure_without_error_message_is_rejected():
    with pytest.raises(ValidationError, match="non-empty error message"):
        make_trace(
            status="failure",
            result=None,
            error=None,
            error_type="ValueError",
            failure_label="unknown_tool",
        )


def test_failure_without_error_type_is_rejected():
    with pytest.raises(ValidationError, match="non-empty error_type"):
        make_trace(
            status="failure",
            result=None,
            error="boom",
            error_type=None,
            failure_label="unknown_tool",
        )


def test_failure_with_result_is_rejected():
    with pytest.raises(ValidationError, match="must not carry a result"):
        make_trace(
            status="failure",
            result=5,
            error="boom",
            error_type="ValueError",
            failure_label="unknown_tool",
        )


def test_invalid_failure_label_is_rejected():
    with pytest.raises(ValidationError, match="Invalid failure label"):
        make_trace(
            status="failure",
            result=None,
            error="boom",
            error_type="ValueError",
            failure_label="http_error",
        )


def test_success_cannot_carry_a_failure_label():
    with pytest.raises(ValidationError, match="cannot carry the failure label"):
        make_trace(failure_label="unknown_tool")


def test_failure_cannot_carry_the_none_label():
    with pytest.raises(ValidationError, match="cannot carry the failure label"):
        make_trace(
            status="failure",
            result=None,
            error="boom",
            error_type="ValueError",
            failure_label="none",
        )


def test_blank_failure_label_means_not_determined():
    trace = make_trace(failure_label="   ")

    assert trace.failure_label is None


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        make_trace(unexpected_field="value")


def test_duration_ms_must_be_numeric():
    with pytest.raises(ValidationError):
        make_trace(duration_ms="not a number")


def test_arguments_must_be_a_mapping():
    with pytest.raises(ValidationError):
        make_trace(arguments=None)


def test_arguments_default_to_an_empty_mapping():
    now = datetime.now(timezone.utc)

    trace = AgentTrace(
        task_id="t-001",
        agent_type="calculator",
        tool_name="add",
        status="success",
        result=1,
        started_at=now,
        finished_at=now,
        duration_ms=0.0,
    )

    assert trace.arguments == {}


def test_create_derives_duration_from_timestamps():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)

    trace = AgentTrace.create(
        task_id="t-001",
        agent_type="calculator",
        tool_name="add",
        arguments={"a": 1, "b": 2},
        status="success",
        started_at=start,
        finished_at=end,
        result=3,
        failure_label="none",
    )

    assert trace.duration_ms == pytest.approx(2000.0)
