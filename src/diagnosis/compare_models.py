
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.diagnosis.features import extract_features


DATA_FILE = Path("data/processed/ml_dataset.csv")
MODEL_DIR = Path("models")


def main():
    df = pd.read_csv(DATA_FILE).fillna("")

    text_columns = [
        "agent_type",
        "tool_name",
        "status",
        "error_type",
        "error",
    ]

    texts = df[text_columns].astype(str).agg(" ".join, axis=1).tolist()
    labels = df["failure_label"].astype(str).to_numpy()

    # Extract structured features from the fields available in the dataset.
    traces = df[text_columns].to_dict(orient="records")
    features = [extract_features(trace) for trace in traces]

    # The current dataset does not preserve arguments, so exclude
    # has_arguments rather than treating its value as meaningful.
    structured_features = [
        {
            "has_error": item["has_error"],
            "has_missing_argument_error": item[
                "has_missing_argument_error"
            ],
            "has_unknown_tool_error": item["has_unknown_tool_error"],
            "has_invalid_input_error": item["has_invalid_input_error"],
        }
        for item in features
    ]

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    # Model 1: original text-only baseline.
    baseline_vectorizer = TfidfVectorizer()
    X_train_text = baseline_vectorizer.fit_transform(
        [texts[i] for i in train_idx]
    )
    X_test_text = baseline_vectorizer.transform(
        [texts[i] for i in test_idx]
    )

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_text, labels[train_idx])
    baseline_predictions = baseline_model.predict(X_test_text)

    # Model 2: text plus structured features.
    text_vectorizer = TfidfVectorizer()
    dict_vectorizer = DictVectorizer()

    X_train_text_enhanced = text_vectorizer.fit_transform(
        [texts[i] for i in train_idx]
    )
    X_test_text_enhanced = text_vectorizer.transform(
        [texts[i] for i in test_idx]
    )

    X_train_structured = dict_vectorizer.fit_transform(
        [structured_features[i] for i in train_idx]
    )
    X_test_structured = dict_vectorizer.transform(
        [structured_features[i] for i in test_idx]
    )

    X_train_combined = hstack(
        [X_train_text_enhanced, X_train_structured]
    )
    X_test_combined = hstack(
        [X_test_text_enhanced, X_test_structured]
    )

    enhanced_model = LogisticRegression(max_iter=1000)
    enhanced_model.fit(X_train_combined, labels[train_idx])
    enhanced_predictions = enhanced_model.predict(X_test_combined)

    baseline_accuracy = accuracy_score(
        labels[test_idx], baseline_predictions
    )
    enhanced_accuracy = accuracy_score(
        labels[test_idx], enhanced_predictions
    )

    print("\nMODEL COMPARISON")
    print("----------------")
    print(f"Test examples: {len(test_idx)}")
    print(f"Text-only accuracy: {baseline_accuracy:.3f}")
    print(f"Text + structured accuracy: {enhanced_accuracy:.3f}")

    print("\nEnhanced model classification report:")
    print(
        classification_report(
            labels[test_idx],
            enhanced_predictions,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "text_vectorizer": text_vectorizer,
            "dict_vectorizer": dict_vectorizer,
            "classifier": enhanced_model,
        },
        MODEL_DIR / "combined_failure_classifier.joblib",
    )
    print("Combined model saved to models/combined_failure_classifier.joblib")


if __name__ == "__main__":
    main()
