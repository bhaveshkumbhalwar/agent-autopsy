"""A minimal calculator agent used to generate trace data.

The agent executes one tool call per task and records an
:class:`~src.tracing.trace_schema.AgentTrace` for it. The recorded trace
carries a taxonomy ``failure_label`` so downstream dataset builders do not
have to re-derive the reason for a failure from free text.
"""

import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.tracing.trace_io import append_trace
from src.tracing.trace_schema import AgentTrace


TRACE_DIR = Path("data/traces")
TRACE_FILE = TRACE_DIR / "standardized_agent_traces.jsonl"

TOOLS = {
    "add": operator.add,
    "multiply": operator.mul,
}

REQUIRED_ARGUMENTS = ("a", "b")

MAX_ARGUMENT_REPR_LENGTH = 200


def _short_repr(value: Any) -> str:
    """Return a bounded representation of a value for error messages."""
    text = repr(value)
    if len(text) > MAX_ARGUMENT_REPR_LENGTH:
        return text[:MAX_ARGUMENT_REPR_LENGTH] + "..."
    return text


def run_agent(
    task: Any,
    trace_file: Optional[Path] = None,
) -> AgentTrace:
    """Execute a tool request and record a standardized trace.

    Args:
        task: A mapping with ``task_id``, ``agent_type``, ``tool`` and
            ``arguments``. Missing keys are filled with sensible defaults.
        trace_file: Where the trace is appended. Defaults to
            :data:`TRACE_FILE`.

    Returns:
        The recorded trace.

    Raises:
        TypeError: If ``task`` is not a mapping. That is a caller error
            rather than a failed execution, so it is not recorded as a trace.
    """
    if not isinstance(task, Mapping):
        raise TypeError(
            f"Task must be a mapping, got {type(task).__name__}."
        )

    started_at = datetime.now(timezone.utc)

    task_id = str(task.get("task_id") or "unknown")
    agent_type = str(task.get("agent_type") or "calculator")
    tool_name = str(task.get("tool") or "unknown")
    raw_arguments = task.get("arguments", {})

    arguments: Dict[str, Any] = {}
    failure_label: Optional[str] = None

    try:
        if not isinstance(raw_arguments, Mapping):
            failure_label = "invalid_input"
            raise TypeError(
                "Task arguments must be a mapping of argument names to "
                f"values, got {type(raw_arguments).__name__}: "
                f"{_short_repr(raw_arguments)}"
            )

        arguments = {str(key): value for key, value in raw_arguments.items()}

        if tool_name not in TOOLS:
            failure_label = "unknown_tool"
            raise ValueError(f"Unknown tool: {tool_name}")

        missing = [
            name for name in REQUIRED_ARGUMENTS
            if name not in arguments
        ]
        if missing:
            failure_label = "missing_argument"
            raise KeyError(missing[0])

        a = arguments["a"]
        b = arguments["b"]

        if (
            isinstance(a, bool) or not isinstance(a, (int, float))
            or isinstance(b, bool) or not isinstance(b, (int, float))
        ):
            failure_label = "invalid_input"
            raise ValueError("Arguments a and b must be numbers")

        result = TOOLS[tool_name](a, b)
        status = "success"
        error = None
        error_type = None
        failure_label = "none"

    except Exception as exc:
        result = None
        status = "failure"
        error = str(exc)
        error_type = type(exc).__name__

    finished_at = datetime.now(timezone.utc)

    trace = AgentTrace.create(
        task_id=task_id,
        agent_type=agent_type,
        tool_name=tool_name,
        arguments=arguments,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        result=result,
        error=error,
        error_type=error_type,
        failure_label=failure_label,
    )

    append_trace(trace, trace_file or TRACE_FILE)

    return trace


if __name__ == "__main__":
    tasks = [
        {
            "task_id": "task-101",
            "tool": "add",
            "arguments": {"a": 10, "b": 5},
        },
        {
            "task_id": "task-102",
            "tool": "multiply",
            "arguments": {"a": 4, "b": 3},
        },
        {
            "task_id": "task-103",
            "tool": "divide",
            "arguments": {"a": 10, "b": 2},
        },
    ]

    for task in tasks:
        trace = run_agent(task)
        print(trace.model_dump_json(indent=2))
