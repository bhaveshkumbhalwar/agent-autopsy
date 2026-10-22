"""Feature definitions shared by training, evaluation and inference.

The feature configuration lives in one place so that a model is always
trained and served with exactly the same inputs in exactly the same order
(see ``TEXT_COLUMNS`` below and ``src/diagnosis/predict_failure.py``).

Two groups of columns are deliberately kept apart:

``TEXT_COLUMNS``
    The fields turned into the model's text input, in the order the model
    expects them.

``IDENTIFIER_COLUMNS``
    Fields that identify a row or say *when* it happened -- task ids,
    durations and timestamps. They are never model inputs. In the generated
    dataset the task id encodes the label outright and the duration varies
    systematically with the label, so feeding either to the model would be a
    shortcut rather than a diagnosis. ``src/evaluation/leakage.py`` measures
    exactly how strong those shortcuts are.
"""

import json
from typing import Any, Dict, Mapping, Tuple

# The text fields the classifier is trained on, in a fixed order.
TEXT_COLUMNS: Tuple[str, ...] = (
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
)

# Fields that must never be used as model inputs.
IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "task_id",
    "duration_ms",
    "started_at",
    "finished_at",
)

# The structured (non-text) features used by the combined model.
STRUCTURED_FEATURE_NAMES: Tuple[str, ...] = (
    "has_error",
    "has_arguments",
    "has_missing_argument_error",
    "has_unknown_tool_error",
    "has_invalid_input_error",
)


def _as_text(value: Any) -> str:
    """Return a string for a cell, treating missing values as empty.

    ``None`` and ``NaN`` both become an empty string so that a missing
    field never leaks the literal text ``"nan"`` into the model input.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        # NaN is the only value that is not equal to itself.
        return ""
    return str(value)


def parse_arguments(value: Any) -> Dict[str, Any]:
    """Return the arguments of a trace as a dictionary.

    Accepts a mapping, or the JSON text used by the CSV datasets, and
    returns an empty dictionary when the arguments are missing or unusable.
    """
    if isinstance(value, Mapping):
        return dict(value)

    text = _as_text(value).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}

    if isinstance(parsed, Mapping):
        return dict(parsed)

    return {}


def build_text(record: Any) -> str:
    """Return the model's text input for one trace or dataset row.

    Args:
        record: A mapping (or a pandas row) with the fields listed in
            :data:`TEXT_COLUMNS`.

    Returns:
        The fields joined by a single space, in a fixed order.
    """
    return " ".join(
        _as_text(record.get(column)) for column in TEXT_COLUMNS
    )


def extract_features(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract structured diagnostic features from an execution trace.

    The boolean features are *heuristic signals*: they look for phrases that
    the calculator agent happens to emit. They are useful for the combined
    model, but they are not a general failure taxonomy -- a trace that
    reports a missing argument in different words will not set
    ``has_missing_argument_error``. ``src/diagnosis/compare_models.py``
    measures how strongly each one tracks the label and how much weight the
    combined model gives it.
    """
    tool_name = _as_text(trace.get("tool_name"))
    agent_type = _as_text(trace.get("agent_type"))
    status = _as_text(trace.get("status"))
    error_type = _as_text(trace.get("error_type"))
    error = _as_text(trace.get("error"))
    arguments = parse_arguments(trace.get("arguments"))

    error_lower = error.lower()

    return {
        "agent_type": agent_type,
        "tool_name": tool_name,
        "status": status,
        "error_type": error_type,
        "error": error,
        "has_error": bool(error),
        "has_arguments": bool(arguments),
        "has_missing_argument_error": (
            error_type == "KeyError"
        ),
        "has_unknown_tool_error": (
            "unknown tool" in error_lower
        ),
        "has_invalid_input_error": (
            "must be numbers" in error_lower
        ),
    }


def build_structured_features(
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return only the structured features, in a fixed order.

    The result is what the combined model consumes alongside the text
    input. Returning them in a fixed order keeps training and inference in
    step (see ``src/diagnosis/train_combined_model.py``).
    """
    features = extract_features(trace)

    return {
        name: features[name] for name in STRUCTURED_FEATURE_NAMES
    }


if __name__ == "__main__":
    sample_trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "arguments": {"a": 10},
    }

    print(extract_features(sample_trace))
