
from pathlib import Path

import joblib
import pandas as pd


MODEL_FILE = Path("models/failure_classifier.joblib")

TEXT_COLUMNS = [
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
]


def diagnose_trace(trace: dict) -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}. "
            "Train the classifier first."
        )

    model = joblib.load(MODEL_FILE)

    # Build input in the same format used during training.
    row = pd.DataFrame(
        [{column: str(trace.get(column) or "")
          for column in TEXT_COLUMNS}]
    )

    text = row[TEXT_COLUMNS].agg(" ".join, axis=1)

    prediction = model.predict(text)[0]
    probabilities = model.predict_proba(text)[0]

    classifier = model.named_steps["classifier"]
    confidence_scores = dict(
        zip(classifier.classes_, probabilities)
    )

    return {
        "task_id": trace.get("task_id", "unknown"),
        "predicted_label": prediction,
        "confidence_scores": {
            label: round(float(score), 3)
            for label, score in confidence_scores.items()
        },
    }


if __name__ == "__main__":
    examples = [
        {
            "task_id": "new-001",
            "agent_type": "calculator",
            "tool_name": "divide",
            "status": "failure",
            "error_type": "ValueError",
            "error": "Unknown tool: divide",
        },
        {
            "task_id": "new-002",
            "agent_type": "calculator",
            "tool_name": "add",
            "status": "failure",
            "error_type": "KeyError",
            "error": "'b'",
        },
        {
            "task_id": "new-003",
            "agent_type": "calculator",
            "tool_name": "add",
            "status": "failure",
            "error_type": "ValueError",
            "error": "Arguments a and b must be numbers",
        },
        {
            "task_id": "new-004",
            "agent_type": "calculator",
            "tool_name": "multiply",
            "status": "success",
            "error_type": "",
            "error": "",
        },
    ]

    for example in examples:
        diagnosis = diagnose_trace(example)
        print("\nDiagnosis:")
        for key, value in diagnosis.items():
            print(f"{key}: {value}")
