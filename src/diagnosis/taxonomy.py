"""Single source of truth for the failure labels used by Agent Autopsy.

The labels below are the ones produced by the synthetic trace generator in
``src/evaluation/build_larger_dataset.py``. Every failure that generator can
produce maps onto exactly one of them, so the taxonomy is closed over the
training data and the classifier is trained on exactly these classes.

Real agent traces also contain failure modes the generator never produces:
HTTP errors, runtime errors inside generated code, timeouts, and so on. Such
a trace is *outside* the trained taxonomy. There is no correct label to
predict for it, so it must not be scored as a model error. See
``src/evaluation/test_generalization.py`` for how these cases are reported.
"""

from typing import Dict, Tuple


SUPPORTED_LABELS: Tuple[str, ...] = (
    "none",
    "unknown_tool",
    "missing_argument",
    "invalid_input",
)


LABEL_DEFINITIONS: Dict[str, str] = {
    "none": "The tool call completed successfully.",
    "unknown_tool": "The requested tool is not present in the tool registry.",
    "missing_argument": "A required argument was absent from the tool call.",
    "invalid_input": "A supplied argument had an unusable type or value.",
}


def is_supported(label: str) -> bool:
    """Return ``True`` when ``label`` belongs to the trained taxonomy."""
    return label in SUPPORTED_LABELS


def describe(label: str) -> str:
    """Return a human readable definition of ``label``.

    Raises:
        ValueError: If ``label`` is not part of the trained taxonomy.
    """
    if label not in LABEL_DEFINITIONS:
        raise ValueError(
            f"Unknown label {label!r}. "
            f"Supported labels: {', '.join(SUPPORTED_LABELS)}"
        )
    return LABEL_DEFINITIONS[label]
