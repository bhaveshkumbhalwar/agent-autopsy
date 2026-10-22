
import json
import random
from pathlib import Path

import pandas as pd

from src.agents.basic_agent import run_agent
from src.tracing.labels import SUPPORTED_LABELS


OUTPUT_FILE = Path("data/processed/failure_dataset.csv")
random.seed(42)


def generate_cases():
    cases = []
    task_number = 1

    # Class 1: Successful executions
    for _ in range(50):
        tool = random.choice(["add", "multiply"])
        arguments = {
            "a": random.randint(1, 100),
            "b": random.randint(1, 100),
        }

        cases.append({
            "task_id": f"d7-{task_number:03}",
            "tool": tool,
            "arguments": arguments,
            "failure_label": "none",
        })
        task_number += 1

    # Class 2: Unknown tools
    unknown_tools = [
        "divide", "subtract", "power", "sqrt", "search",
        "save", "delete", "send", "fetch", "unknown",
    ]

    for i in range(50):
        cases.append({
            "task_id": f"d7-{task_number:03}",
            "tool": unknown_tools[i % len(unknown_tools)],
            "arguments": {
                "a": random.randint(1, 100),
                "b": random.randint(1, 100),
            },
            "failure_label": "unknown_tool",
        })
        task_number += 1

    # Class 3: Missing required arguments
    for _ in range(50):
        cases.append({
            "task_id": f"d7-{task_number:03}",
            "tool": random.choice(["add", "multiply"]),
            "arguments": {"a": random.randint(1, 100)},
            "failure_label": "missing_argument",
        })
        task_number += 1

    # Class 4: Invalid input types
    invalid_values = [
        "hello", None, True, [], "25", {"value": 10}
    ]

    for i in range(50):
        cases.append({
            "task_id": f"d7-{task_number:03}",
            "tool": random.choice(["add", "multiply"]),
            "arguments": {
                "a": invalid_values[i % len(invalid_values)],
                "b": random.randint(1, 100),
            },
            "failure_label": "invalid_input",
        })
        task_number += 1

    random.shuffle(cases)

    # The generated labels must stay inside the trained taxonomy, otherwise
    # the classifier would be trained on a class it cannot be evaluated on.
    unsupported = {
        case["failure_label"] for case in cases
    } - set(SUPPORTED_LABELS)
    if unsupported:
        raise ValueError(
            f"Generated labels outside the trained taxonomy: "
            f"{sorted(unsupported)}. "
            f"Supported labels: {', '.join(SUPPORTED_LABELS)}"
        )

    return cases


def build_dataset():
    """Execute every generated case and write the labelled dataset.

    Raises:
        ValueError: If the label recorded by the trace disagrees with the
            label the case expects, or is outside the trained taxonomy.
    """
    rows = []

    for case in generate_cases():
        trace = run_agent(case)

        # The trace decides the label; the case states what was expected.
        # Disagreement means the agent or the generator is wrong.
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
                f"{label!r} but the case expects {expected!r} "
                f"(status={trace.status})."
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
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nExpanded dataset created!")
    print("Dataset shape:", df.shape)
    print("\nLabel counts:")
    print(df["failure_label"].value_counts())
    print("\nActual execution status:")
    print(df["status"].value_counts())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()
