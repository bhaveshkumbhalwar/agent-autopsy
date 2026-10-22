"""Tests for the JSONL trace format and its explicit legacy migration."""

import json
from datetime import datetime, timezone

import pytest

from src.tracing.trace_io import (
    TraceFormatError,
    append_trace,
    load_traces,
    write_traces,
)
from src.tracing.trace_schema import AgentTrace


def build_trace(task_id: str = "t-001") -> AgentTrace:
    now = datetime.now(timezone.utc)

    return AgentTrace.create(
        task_id=task_id,
        agent_type="calculator",
        tool_name="add",
        arguments={"a": 1, "b": 2},
        status="success",
        started_at=now,
        finished_at=now,
        result=3,
        failure_label="none",
    )


def test_append_then_load_round_trip(tmp_path):
    path = tmp_path / "traces.jsonl"

    append_trace(build_trace("t-001"), path)
    append_trace(build_trace("t-002"), path)

    result = load_traces(path)

    assert len(result.traces) == 2
    assert [trace.task_id for trace in result.traces] == ["t-001", "t-002"]
    assert result.migrated == 0


def test_write_traces_replaces_previous_content(tmp_path):
    path = tmp_path / "traces.jsonl"

    write_traces([build_trace("t-001"), build_trace("t-002")], path)
    write_traces([build_trace("t-003")], path)

    result = load_traces(path)

    assert [trace.task_id for trace in result.traces] == ["t-003"]


def test_legacy_record_without_duration_is_migrated(tmp_path):
    path = tmp_path / "legacy.jsonl"
    record = {
        "task_id": "legacy-001",
        "agent_type": "calculator",
        "tool_name": "add",
        "arguments": {"a": 10, "b": 5},
        "status": "success",
        "result": 15,
        "error": None,
        "error_type": None,
        "started_at": "2026-10-18T18:19:41.192837+00:00",
        "finished_at": "2026-10-18T18:19:41.193011+00:00",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = load_traces(path)

    assert result.migrated == 1
    assert len(result.traces) == 1
    assert result.traces[0].duration_ms == pytest.approx(0.174, abs=0.01)
    # A legacy record cannot state its label, so it stays undetermined.
    assert result.traces[0].failure_label is None


def test_invalid_json_reports_the_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(build_trace().model_dump(mode="json")) + "\n"
        + "{not json}\n",
        encoding="utf-8",
    )

    with pytest.raises(TraceFormatError, match=r":2 is not valid JSON"):
        load_traces(path)


def test_non_object_line_is_rejected(tmp_path):
    path = tmp_path / "array.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(TraceFormatError, match="not a JSON object"):
        load_traces(path)


def test_record_that_breaks_the_contract_is_rejected(tmp_path):
    path = tmp_path / "bad-contract.jsonl"
    record = build_trace().model_dump(mode="json")
    record["status"] = "finished"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(TraceFormatError, match="does not match the trace"):
        load_traces(path)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "spaced.jsonl"
    path.write_text(
        "\n" + json.dumps(build_trace().model_dump(mode="json")) + "\n\n",
        encoding="utf-8",
    )

    result = load_traces(path)

    assert len(result.traces) == 1
