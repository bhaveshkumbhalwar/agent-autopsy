"""Turn the raw labelled dataset into the dataset used for modelling.

This step is also the dataset's data-quality gate. Rows whose target label
is missing or outside the trained taxonomy are **not** silently dropped or
replaced with a valid label: the script fails and names the offending rows
so the source data can be fixed and regenerated.
"""

from pathlib import Path
from typing import List

import pandas as pd

from src.tracing.labels import ALLOWED_STATUSES, SUPPORTED_LABELS


INPUT_FILE = Path("data/processed/failure_dataset.csv")
OUTPUT_FILE = Path("data/processed/ml_dataset.csv")

TARGET = "failure_label"

# Columns carried through from the raw dataset into the modelling dataset.
OUTPUT_COLUMNS = [
    "task_id",
    "agent_type",
    "tool_name",
    "arguments",
    "status",
    "error_type",
    "error",
    "duration_ms",
    "failure_label",
]

# Columns whose empty value means "this did not happen" rather than "unknown".
NULLABLE_COLUMNS = ["error", "error_type"]

MAX_REPORTED_PROBLEMS = 10


def _clean(value) -> str:
    """Return a trimmed string for a cell, treating blanks as empty."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def collect_problems(df: pd.DataFrame) -> List[str]:
    """Return a description of every data-quality problem in ``df``.

    The rules enforced here are:

    * ``status`` is one of the allowed execution statuses.
    * ``agent_type`` and ``tool_name`` are present.
    * ``failure_label`` is present and inside the trained taxonomy.
    * the label agrees with the status: ``none`` means ``success``.
    """
    problems: List[str] = []

    for _, row in df.iterrows():
        task_id = _clean(row.get("task_id")) or "<unknown task>"

        for column in ("agent_type", "tool_name"):
            if not _clean(row.get(column)):
                problems.append(f"{task_id}: missing {column}")

        status = _clean(row.get("status"))
        if status not in ALLOWED_STATUSES:
            problems.append(
                f"{task_id}: invalid status {status!r}; allowed: "
                f"{', '.join(ALLOWED_STATUSES)}"
            )

        label = _clean(row.get(TARGET))
        if not label:
            problems.append(
                f"{task_id}: missing {TARGET}; a row without a target cannot "
                "be used for supervised learning"
            )
        elif label not in SUPPORTED_LABELS:
            problems.append(
                f"{task_id}: invalid {TARGET} {label!r}; supported: "
                f"{', '.join(SUPPORTED_LABELS)}"
            )
        elif (label == "none") != (status == "success"):
            problems.append(
                f"{task_id}: {TARGET} {label!r} disagrees with status "
                f"{status!r}"
            )

    return problems


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` when ``df`` violates the dataset contract.

    Raises:
        ValueError: Listing every problem found.
    """
    problems = collect_problems(df)
    if not problems:
        return

    detail = "\n  - ".join(problems[:MAX_REPORTED_PROBLEMS])
    remaining = len(problems) - MAX_REPORTED_PROBLEMS
    suffix = f"\n  ... and {remaining} more" if remaining > 0 else ""

    raise ValueError(
        f"{len(problems)} problem(s) found in the dataset:\n"
        f"  - {detail}{suffix}\n"
        "Labels are never guessed or dropped automatically. Fix the source "
        "data and regenerate it."
    )


def prepare_dataset():
    """Validate the raw dataset and write the modelling dataset.

    Raises:
        FileNotFoundError: If the raw dataset has not been generated.
        ValueError: If the raw dataset violates the dataset contract.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. "
            "Run python -m src.evaluation.build_larger_dataset first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Regenerate the dataset with "
            "python -m src.evaluation.build_larger_dataset."
        )

    validate_dataset(df)

    before = len(df)
    df = df.drop_duplicates(subset=["task_id"])
    duplicates = before - len(df)

    # Only the error columns may be empty; a blank there means "no error".
    df[NULLABLE_COLUMNS] = df[NULLABLE_COLUMNS].fillna("")

    ml_df = df[OUTPUT_COLUMNS].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(OUTPUT_FILE, index=False)

    print("Original dataset shape:", (before, len(df.columns)))
    print(f"Duplicate task_id rows removed: {duplicates}")
    print("Prepared dataset shape:", ml_df.shape)
    print("Preserved columns:", ", ".join(OUTPUT_COLUMNS))

    print("\nLabel distribution:")
    print(ml_df[TARGET].value_counts())

    print("\nSaved prepared dataset to:", OUTPUT_FILE)
    print("\nPrepared dataset preview:")
    print(ml_df.head(10).to_string(index=False))


if __name__ == "__main__":
    prepare_dataset()
