
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/failure_dataset.csv")
OUTPUT_FILE = Path("data/processed/ml_dataset.csv")

TARGET = "failure_label"

FEATURE_COLUMNS = [
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
]


def prepare_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. "
            "Run the Day 5 dataset generator first."
        )

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Remove duplicate task records.
    df = df.drop_duplicates(subset=["task_id"])

    # Check required columns.
    required_columns = FEATURE_COLUMNS + [TARGET, "task_id"]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Fill missing feature values with empty strings.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna("")

    # Rows without a target cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET])

    # Keep only features and the target for this first experiment.
    ml_df = df[FEATURE_COLUMNS + [TARGET]].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(OUTPUT_FILE, index=False)

    X = ml_df[FEATURE_COLUMNS]
    y = ml_df[TARGET]

    print("\nPrepared dataset shape:", ml_df.shape)
    print("Feature matrix X shape:", X.shape)
    print("Target vector y shape:", y.shape)

    print("\nLabel distribution:")
    print(y.value_counts())

    print("\nSaved prepared dataset to:", OUTPUT_FILE)
    print("\nPrepared dataset preview:")
    print(ml_df.to_string(index=False))


if __name__ == "__main__":
    prepare_dataset()
