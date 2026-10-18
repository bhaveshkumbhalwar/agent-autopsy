
import operator
from datetime import datetime, timezone
from pathlib import Path

from src.tracing.trace_schema import AgentTrace


TRACE_DIR = Path("data/traces")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

TRACE_FILE = TRACE_DIR / "standardized_agent_traces.jsonl"

TOOLS = {
    "add": operator.add,
    "multiply": operator.mul,
}


def run_agent(task: dict) -> AgentTrace:
    """Execute a tool request and save a standardized trace."""

    started_at = datetime.now(timezone.utc)

    task_id = task.get("task_id", "demo-001")
    agent_type = task.get("agent_type", "calculator")
    tool_name = task.get("tool", "unknown")
    arguments = task.get("arguments", {})

    try:
        if tool_name not in TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")

        a = arguments["a"]
        b = arguments["b"]

        if (
            isinstance(a, bool) or not isinstance(a, (int, float))
            or isinstance(b, bool) or not isinstance(b, (int, float))
        ):
            raise ValueError("Arguments a and b must be numbers")

        result = TOOLS[tool_name](a, b)
        status = "success"
        error = None
        error_type = None

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
    )

    with TRACE_FILE.open("a") as file:
        file.write(trace.model_dump_json() + "\n")

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
