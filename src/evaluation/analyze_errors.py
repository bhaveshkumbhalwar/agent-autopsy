"""Report the cases the classifier actually got wrong.

Only scored cases can be model errors. Cases whose ground truth falls
outside the trained taxonomy have no correct label to predict, so they are
reported separately as a coverage gap instead of being counted as
misclassifications.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("reports/generalization_results.csv")
OUTPUT_FILE = Path("reports/error_analysis.csv")


def true_mask(series: pd.Series) -> pd.Series:
    """Interpret a column of ``True``/``False`` values as a boolean mask.

    The column may be read back as booleans or as strings depending on the
    blanks present in the CSV, so both spellings are accepted.
    """
    return series.astype(str).str.strip().str.lower() == "true"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Run src/evaluation/test_generalization.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "task_id",
        "expected_label",
        "predicted_label",
        "correct",
        "scored",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns: {sorted(required)}. "
            "Re-run src/evaluation/test_generalization.py to regenerate "
            f"{INPUT_FILE}."
        )

    scored = df[true_mask(df["scored"])].copy()
    unscored = df[~true_mask(df["scored"])].copy()
    errors = scored[~true_mask(scored["correct"])].copy()

    errors["error_analysis"] = errors.apply(
        lambda row: (
            f"Expected {row['expected_label']}, "
            f"but predicted {row['predicted_label']}. "
            "Inspect the input features and label definition."
        ),
        axis=1,
    )

    if errors.empty:
        print("No incorrect predictions among the scored cases.")
    else:
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

    if not unscored.empty:
        print(
            "\nOutside the trained taxonomy (not counted as model errors):"
        )
        print(
            unscored[
                [
                    "task_id",
                    "predicted_label",
                    "out_of_taxonomy_reason",
                ]
            ].to_string(index=False)
            if "out_of_taxonomy_reason" in unscored.columns
            else unscored[["task_id", "predicted_label"]].to_string(index=False)
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved error analysis to: {OUTPUT_FILE}")
    print(
        f"Model errors: {len(errors)} / {len(scored)} scored cases "
        f"({len(unscored)} cases outside the trained taxonomy)"
    )


if __name__ == "__main__":
    main()
