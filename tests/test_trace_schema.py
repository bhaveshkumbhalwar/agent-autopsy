
from datetime import datetime, timezone

from src.tracing.trace_schema import AgentTrace


def test_create_failure_trace():
    start = datetime.now(timezone.utc)
    end = start.replace(
        microsecond=start.microsecond
    )

    trace = AgentTrace.create(
        task_id="task-101",
        agent_type="python",
        tool_name="execute_code",
        arguments={"code": "print(x)"},
        status="failure",
        started_at=start,
        finished_at=end,
        error="name 'x' is not defined",
        error_type="NameError",
    )

    assert trace.status == "failure"
    assert trace.error_type == "NameError"
    assert trace.duration_ms >= 0
