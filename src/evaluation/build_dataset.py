"""Build the small labelled dataset from executed agent tasks."""

import json
from pathlib import Path

import pandas as pd

from src.agents.basic_agent import run_agent
from src.tracing.labels import SUPPORTED_LABELS


OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "task_id": "d5-001",
        "tool": "add",
        "arguments": {"a": 10, "b": 5},
        "failure_label": "none",
    },
    {
        "task_id": "d5-002",
        "tool": "multiply",
        "arguments": {"a": 4, "b": 3},
        "failure_label": "none",
    },
    {
        "task_id": "d5-003",
        "tool": "divide",
        "arguments": {"a": 10, "b": 2},
        "failure_label": "unknown_tool",
    },
    {
        "task_id": "d5-004",
        "tool": "add",
        "arguments": {"a": 10},
        "failure_label": "missing_argument",
    },
    {
        "task_id": "d5-005",
        "tool": "add",
        "arguments": {"a": "hello", "b": 5},
        "failure_label": "invalid_input",
    },
]


def build_dataset():
    """Execute every case and write the labelled dataset.

    Raises:
        ValueError: If the label recorded by the trace disagrees with the
            label the case expects, or is outside the trained taxonomy.
    """
    rows = []

    for case in CASES:
        trace = run_agent(case)

        label = trace.failure_label
        expected = case["failure_label"]

        if label not in SUPPORTED_LABELS:
            raise ValueError(
                f"Trace for {case['task_id']} carries a label outside the "
                f"trained taxonomy: {label!r}. "
                f"Supported labels: {', '.join(SUPPORTED_LABELS)}"
            )

        if label != expected:
            raise ValueError(
                f"Label mismatch for {case['task_id']}: the trace was labelled "
                f"{label!r} but the case expects {expected!r}."
            )

        rows.append({
            "task_id": case["task_id"],
            "agent_type": trace.agent_type,
            "tool_name": trace.tool_name,
            "arguments": json.dumps(trace.arguments, sort_keys=True),
            "status": trace.status,
            "error_type": trace.error_type,
            "error": trace.error,
            "duration_ms": trace.duration_ms,
            "failure_label": label,
        })

    df = pd.DataFrame(rows)

    output_file = OUTPUT_DIR / "failure_dataset.csv"
    df.to_csv(output_file, index=False)

    print("\nDataset created successfully!")
    print(f"Saved to: {output_file}")
    print("\nDataset preview:")
    print(df[[
        "task_id",
        "status",
        "error_type",
        "failure_label",
    ]].to_string(index=False))

    print("\nLabel counts:")
    print(df["failure_label"].value_counts())


if __name__ == "__main__":
    build_dataset()
