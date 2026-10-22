"""Shared training routine for the two diagnosis models.

Both models take the same input and differ only in which pipeline is
built, so the loading, validation and reporting around them lives here
once. The two entry points are thin wrappers:

``python -m src.diagnosis.train_classifier``
    Trains the text-only baseline.
``python -m src.diagnosis.train_combined_model``
    Trains the text-plus-structured model.

Neither script prints a score. The split they used to print a report for
was a plain random split, and the leakage audit showed that every test row
of such a split was a literal copy of a training row -- the number was a
measure of memorisation. The leakage-free numbers live in

    python -m src.evaluation.evaluate_model
    python -m src.diagnosis.compare_models
"""

from pathlib import Path
from typing import Callable, List

import joblib
import pandas as pd

from src.diagnosis.features import TEXT_COLUMNS
from src.diagnosis.modeling import MODEL_FILES, feature_configuration
from src.tracing.labels import validate_labels


DATA_FILE = Path("data/processed/ml_dataset.csv")


def load_training_frame() -> pd.DataFrame:
    """Load the prepared dataset and validate it can be trained on.

    Returns:
        The dataset, unchanged.

    Raises:
        FileNotFoundError: If the dataset has not been generated.
        ValueError: If a required column is missing, or if a label is
            missing or outside the taxonomy.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run python -m src.evaluation.build_larger_dataset and "
            "python -m src.evaluation.prepare_dataset first."
        )

    df = pd.read_csv(DATA_FILE)

    required = list(TEXT_COLUMNS) + ["failure_label"]
    missing_columns = [
        column for column in required if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            "Regenerate it with python -m src.evaluation.prepare_dataset."
        )

    # A missing or unknown label is never trained on, and never replaced
    # with a valid one.
    identifiers = df["task_id"] if "task_id" in df.columns else None
    validate_labels(df["failure_label"], identifiers, source=str(DATA_FILE))

    if df["failure_label"].nunique() < 2:
        raise ValueError("At least two label categories are required.")

    return df


def train_and_save(name: str, pipeline_factory: Callable[[], object]) -> Path:
    """Fit ``name``'s pipeline on every training row and save it.

    The saved object is a single scikit-learn ``Pipeline`` that contains
    the vectorizer, the scaler and the classifier, so it is everything
    inference needs.

    Args:
        name: The model name in :data:`src.diagnosis.modeling.MODEL_FILES`.
        pipeline_factory: A callable that returns an unfitted pipeline.

    Returns:
        The path the model was written to.
    """
    df = load_training_frame()

    # The model input is built by the pipeline itself, from the trace
    # mappings, so training and inference cannot drift apart. Identifiers
    # such as the task id are deliberately not part of it.
    traces: List[dict] = df.to_dict("records")
    labels = df["failure_label"].to_numpy()

    model = pipeline_factory()
    model.fit(traces, labels)

    model_file = MODEL_FILES[name]
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)

    print("Model trained.")
    print(f"Model: {name}")
    print(f"Dataset: {DATA_FILE} ({len(df)} rows)")
    print("Label distribution:")
    print(df["failure_label"].value_counts().to_string())

    configuration = feature_configuration()
    print("\nFeature configuration:")
    print(f"  text columns: {', '.join(configuration['text_columns'])}")
    print(
        "  excluded identifiers: "
        f"{', '.join(configuration['identifier_columns_excluded'])}"
    )
    print(
        "  structured features: "
        f"{', '.join(configuration['structured_feature_names'])}"
    )
    print(f"\nModel saved to: {model_file}")
    print(
        "\nScores are not printed here: the leakage-free evaluation is "
        "run with\n    python -m src.evaluation.evaluate_model\nand the "
        "baseline/combined comparison with\n    "
        "python -m src.diagnosis.compare_models"
    )

    return model_file


__all__ = [
    "DATA_FILE",
    "load_training_frame",
    "train_and_save",
]
