"""Diagnose an execution trace with a trained classifier.

    python -m src.diagnosis.predict_failure
    python -m src.diagnosis.predict_failure --model combined

Two models are available:

``text``
    The text-only baseline.
``combined``
    Text plus the structured features.

Both are scikit-learn pipelines that take a sequence of traces, so the
caller passes the same mapping to either one. The model is loaded once and
cached, so repeated calls do not read the file from disk every time.
"""

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib

from src.diagnosis.modeling import MODEL_FILES, TraceFeatureExtractor


DEFAULT_MODEL = "text"


@lru_cache(maxsize=4)
def load_model(model: str = DEFAULT_MODEL, path: Optional[str] = None):
    """Load and cache a trained classifier.

    Args:
        model: Which model to load: ``"text"`` or ``"combined"``.
        path: An explicit model file, overriding ``model``.

    Returns:
        The fitted pipeline.

    Raises:
        ValueError: If the model name is unknown, or if the file does not
            contain a pipeline built by this project.
        FileNotFoundError: If the model has not been trained yet.
    """
    if path is None:
        if model not in MODEL_FILES:
            raise ValueError(
                f"Unknown model {model!r}; expected one of "
                f"{', '.join(sorted(MODEL_FILES))}."
            )
        model_file = MODEL_FILES[model]
    else:
        model_file = Path(path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}. "
            "Train it first:\n"
            "    python -m src.diagnosis.train_classifier\n"
            "    python -m src.diagnosis.train_combined_model"
        )

    pipeline = joblib.load(model_file)

    # A model saved before the pipelines took trace mappings would be
    # handed a dictionary and fail with a confusing error inside the
    # vectorizer. Fail with an instruction instead.
    steps = getattr(pipeline, "named_steps", {})
    if not isinstance(steps.get("features"), TraceFeatureExtractor):
        raise ValueError(
            f"{model_file} does not contain a pipeline built by this "
            "project. Retrain it with "
            "python -m src.diagnosis.train_classifier or "
            "python -m src.diagnosis.train_combined_model."
        )

    return pipeline


def predict_with_pipeline(
    pipeline: object,
    trace: Mapping[str, Any],
) -> dict:
    """Predict the failure category of one trace with a loaded pipeline.

    This is the shared inference path: the CLI loads a model and calls
    this, and the API loads a model once and calls this per request. It
    never reads from disk.

    Args:
        pipeline: A fitted pipeline built by :mod:`src.diagnosis.modeling`.
        trace: A trace mapping. Missing fields are treated as empty, so a
            trace without an error message can still be diagnosed.

    Returns:
        The task id, the predicted label and the classifier's class
        scores. The scores are the classifier's decision values mapped
        through a softmax; they are reported as ``class_scores`` rather
        than probabilities because the model is not calibrated.

    Raises:
        ValueError: If the trace is not a mapping.
    """
    if not isinstance(trace, Mapping):
        raise ValueError(
            f"A trace must be a mapping, got {type(trace).__name__}."
        )

    # The pipeline builds the features itself, from the same trace
    # mapping training used, so a trace cannot be featurised differently
    # at inference time.
    prediction = pipeline.predict([trace])[0]
    scores = pipeline.predict_proba([trace])[0]

    classifier = pipeline.named_steps["classifier"]

    return {
        "task_id": str(trace.get("task_id", "unknown")),
        "predicted_label": str(prediction),
        "class_scores": {
            str(label): round(float(score), 4)
            for label, score in zip(classifier.classes_, scores)
        },
    }


def diagnose_trace(trace: Mapping[str, Any], model: str = DEFAULT_MODEL) -> dict:
    """Predict the failure category of one execution trace.

    Args:
        trace: A trace mapping.
        model: Which model to use: ``"text"`` or ``"combined"``.

    Returns:
        The task id, the predicted label and the classifier's class
        scores.

    Raises:
        ValueError: If the trace is not a mapping.
        FileNotFoundError: If the model has not been trained yet.
    """
    return predict_with_pipeline(load_model(model), trace)


def diagnose_traces(
    traces: List[Mapping[str, Any]],
    model: str = DEFAULT_MODEL,
) -> List[dict]:
    """Predict the failure category of several traces."""
    return [diagnose_trace(trace, model=model) for trace in traces]


EXAMPLES = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FILES),
        default=DEFAULT_MODEL,
        help="Which trained model to use.",
    )
    arguments = parser.parse_args()

    for example in EXAMPLES:
        diagnosis = diagnose_trace(example, model=arguments.model)
        print("\nDiagnosis:")
        for key, value in diagnosis.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
