
import random
from pathlib import Path

import pandas as pd

from src.agents.basic_agent import run_agent
from src.diagnosis.taxonomy import SUPPORTED_LABELS


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
    rows = []

    for case in generate_cases():
        trace = run_agent(case)

        # A label should match the actual outcome.
        expected_failure = case["failure_label"] != "none"
        actual_failure = trace.status == "failure"

        if expected_failure != actual_failure:
            raise ValueError(
                f"Label mismatch for {case['task_id']}: "
                f"expected failure={expected_failure}, "
                f"actual status={trace.status}"
            )

        rows.append({
            "task_id": case["task_id"],
            "agent_type": trace.agent_type,
            "tool_name": trace.tool_name,
            "arguments": str(trace.arguments),
            "status": trace.status,
            "error_type": trace.error_type,
            "error": trace.error,
            "duration_ms": trace.duration_ms,
            "failure_label": case["failure_label"],
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
