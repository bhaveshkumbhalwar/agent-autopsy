
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_FILE = Path("data/processed/ml_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "failure_classifier.joblib"

TEXT_COLUMNS = [
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run the Day 7 dataset generation and Day 6 "
            "preparation scripts first."
        )

    df = pd.read_csv(DATA_FILE)

    required = TEXT_COLUMNS + ["failure_label"]
    if df[required].isna().any().any():
        df[required] = df[required].fillna("")

    if df["failure_label"].nunique() < 2:
        raise ValueError("At least two label categories are required.")

    # Combine trace fields into one text input.
    X = df[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
    y = df["failure_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
