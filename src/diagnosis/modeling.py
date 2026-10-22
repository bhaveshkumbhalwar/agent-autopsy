"""Model construction shared by training, evaluation and inference.

Both models in this project are scikit-learn ``Pipeline`` objects built
here, and both take the *same* input: a sequence of execution traces
(mappings). That is what makes "training and prediction use exactly the
same feature definitions and order" true rather than aspirational:

* the text input is produced by :func:`src.diagnosis.features.build_text`,
  which joins :data:`src.diagnosis.features.TEXT_COLUMNS` in a fixed
  order;
* the structured input is produced by
  :func:`src.diagnosis.features.build_structured_features`, which returns
  :data:`src.diagnosis.features.STRUCTURED_FEATURE_NAMES` in a fixed
  order;
* the vectorizer and the scaler are fitted inside the pipeline, on the
  training rows only.

The text-only baseline and the combined model differ in exactly one
place: whether the structured block is appended to the text block. The
comparison between them is therefore controlled -- see
``src/diagnosis/compare_models.py``.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.diagnosis.features import (
    IDENTIFIER_COLUMNS,
    STRUCTURED_FEATURE_NAMES,
    TEXT_COLUMNS,
    build_structured_features,
    build_text,
)


# Hyperparameters of the classifier. These are the ones the project has
# always used; they were not chosen by looking at any test set.
TEXT_VECTORIZER_PARAMS: Dict[str, Any] = dict()
CLASSIFIER_PARAMS: Dict[str, Any] = dict(max_iter=1000)

RANDOM_STATE = 42

MODEL_DIR = Path("models")

# The model this project can serve, and the file each one is saved to.
# Training and inference both read this mapping, so they cannot disagree
# about where a model lives.
MODEL_FILES: Dict[str, Path] = {
    "text": MODEL_DIR / "failure_classifier.joblib",
    "combined": MODEL_DIR / "combined_failure_classifier.joblib",
}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "text": (
        "TF-IDF of the trace text only (the baseline the project started "
        "with)"
    ),
    "combined": (
        "TF-IDF of the trace text plus the standardised structured "
        "features"
    ),
}


class TraceFeatureExtractor(BaseEstimator, TransformerMixin):
    """Turn execution traces into the matrix the classifier consumes.

    The matrix has two blocks:

    * the TF-IDF vectorisation of the trace's text, and
    * the structured features of :mod:`src.diagnosis.features`.

    The structured block is standardised, because a TF-IDF row is
    L2-normalised while the structured block is a handful of indicators;
    without standardisation the regularisation would treat the two
    blocks very differently. The scaler is fitted here, on the training
    rows only.

    Only fields of the trace itself are read. Nothing in this class looks
    at a label, so a feature can never be derived from the answer.

    Args:
        use_structured_features: Whether to append the structured block.
            This is the only difference between the text-only baseline
            and the combined model.
    """

    def __init__(self, use_structured_features: bool = False):
        self.use_structured_features = use_structured_features

    def fit(self, X: Sequence[Mapping[str, Any]], y: Any = None):
        """Fit the vectorizer and the scaler on the training traces."""
        self.tfidf_ = TfidfVectorizer(**TEXT_VECTORIZER_PARAMS)
        self.tfidf_.fit([build_text(trace) for trace in X])

        if self.use_structured_features:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(self._structured_block(X))

        return self

    def transform(self, X: Sequence[Mapping[str, Any]]):
        """Turn traces into the feature matrix."""
        text = self.tfidf_.transform([build_text(trace) for trace in X])

        if not self.use_structured_features:
            return text

        structured = self.scaler_.transform(self._structured_block(X))

        return hstack([text, csr_matrix(structured)])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the name of every column of the feature matrix."""
        names = [
            str(name) for name in self.tfidf_.get_feature_names_out()
        ]

        if self.use_structured_features:
            names += list(STRUCTURED_FEATURE_NAMES)

        return np.array(names)

    @staticmethod
    def _structured_block(
        X: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        """Return the structured features of the traces, in a fixed order."""
        rows = []

        for trace in X:
            features = build_structured_features(trace)
            rows.append([
                float(features[name]) for name in STRUCTURED_FEATURE_NAMES
            ])

        return np.array(rows, dtype=float)


def build_text_pipeline() -> Pipeline:
    """Return the text-only baseline pipeline.

    The returned pipeline is unfitted: the vectorizer learns its
    vocabulary from whatever traces it is given.
    """
    return Pipeline([
        (
            "features",
            TraceFeatureExtractor(use_structured_features=False),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=CLASSIFIER_PARAMS["max_iter"],
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_combined_pipeline() -> Pipeline:
    """Return the combined text-plus-structured pipeline."""
    return Pipeline([
        (
            "features",
            TraceFeatureExtractor(use_structured_features=True),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=CLASSIFIER_PARAMS["max_iter"],
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_pipeline(name: str) -> Pipeline:
    """Return the pipeline called ``name``.

    Args:
        name: Either ``"text"`` or ``"combined"``.

    Raises:
        ValueError: If the name is not one of the two models.
    """
    if name == "text":
        return build_text_pipeline()
    if name == "combined":
        return build_combined_pipeline()

    raise ValueError(
        f"Unknown model {name!r}; expected 'text' or 'combined'."
    )


def feature_configuration() -> dict:
    """Return a description of the feature configuration for the reports."""
    return {
        "input": "a sequence of execution traces (mappings)",
        "text_columns": list(TEXT_COLUMNS),
        "identifier_columns_excluded": list(IDENTIFIER_COLUMNS),
        "structured_feature_names": list(STRUCTURED_FEATURE_NAMES),
        "text_block": {
            "type": "TfidfVectorizer",
            "params": dict(TEXT_VECTORIZER_PARAMS),
        },
        "structured_block": {
            "type": "StandardScaler",
            "applied": "to the structured features only",
        },
        "classifier": {
            "type": "LogisticRegression",
            "params": dict(CLASSIFIER_PARAMS),
        },
        "random_state": RANDOM_STATE,
    }
