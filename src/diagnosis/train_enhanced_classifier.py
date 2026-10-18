
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer


DATA_FILE = Path("data/processed/ml_dataset.csv")
MODEL_FILE = Path("models/enhanced_failure_classifier.joblib")

TEXT_COLUMNS = [
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
]

def main():
    df = pd.read_csv(DATA_FILE).fillna("")

    X = df[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
    y = df["failure_label"].astype(str)

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

    print("Enhanced classifier baseline")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
