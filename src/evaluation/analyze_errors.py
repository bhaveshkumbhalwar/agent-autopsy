
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("reports/generalization_results.csv")
OUTPUT_FILE = Path("reports/error_analysis.csv")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Run the Day 10 generalization test first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "task_id",
        "expected_label",
        "predicted_label",
        "correct",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns: {sorted(required)}"
        )

    errors = df[df["correct"] == False].copy()

    if errors.empty:
        print("No incorrect predictions found.")
    else:
        errors["error_analysis"] = errors.apply(
            lambda row: (
                f"Expected {row['expected_label']}, "
                f"but predicted {row['predicted_label']}. "
                "Inspect the input features and label definition."
            ),
            axis=1,
        )

        print("\nIncorrect predictions:")
        print(
            errors[
                [
                    "task_id",
                    "expected_label",
                    "predicted_label",
                    "error_analysis",
                ]
            ].to_string(index=False)
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved error analysis to: {OUTPUT_FILE}")
    print(f"Incorrect predictions: {len(errors)} / {len(df)}")


if __name__ == "__main__":
    main()
