
from pathlib import Path

import pandas as pd

from src.diagnosis.predict_failure import diagnose_trace


OUTPUT_FILE = Path("reports/generalization_results.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# These are new test cases, not copied from the training CSV.
TEST_CASES = [
    {
        "task_id": "gen-001",
        "agent_type": "calculator",
        "tool_name": "power",
        "status": "failure",
        "error_type": "ValueError",
        "error": "Unknown tool: power",
        "expected_label": "unknown_tool",
    },
    {
        "task_id": "gen-002",
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "expected_label": "missing_argument",
    },
    {
        "task_id": "gen-003",
        "agent_type": "calculator",
        "tool_name": "multiply",
        "status": "failure",
        "error_type": "ValueError",
        "error": "Arguments a and b must be numbers",
        "expected_label": "invalid_input",
    },
    {
        "task_id": "gen-004",
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "success",
        "error_type": "",
        "error": "",
        "expected_label": "none",
    },
    {
        "task_id": "gen-005",
        "agent_type": "python",
        "tool_name": "execute_code",
        "status": "failure",
        "error_type": "NameError",
        "error": "name 'total' is not defined",
        "expected_label": "invalid_input",
    },
    {
        "task_id": "gen-006",
        "agent_type": "web",
        "tool_name": "fetch_page",
        "status": "failure",
        "error_type": "HTTPError",
        "error": "The requested page could not be found",
        "expected_label": "unknown_tool",
    },
]


def main():
    results = []

    for case in TEST_CASES:
        prediction = diagnose_trace(case)

        expected = case["expected_label"]
        predicted = prediction["predicted_label"]

        results.append({
            "task_id": case["task_id"],
            "expected_label": expected,
            "predicted_label": predicted,
            "correct": expected == predicted,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nGeneralization test results:")
    print(df.to_string(index=False))

    accuracy = df["correct"].mean()
    print(f"\nAccuracy on these {len(df)} cases: {accuracy:.1%}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
