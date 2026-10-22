"""Shared vocabulary for execution statuses and failure labels.

Every module that produces or consumes traces, datasets and diagnoses agrees
on the terms defined here:

* an **execution status** says whether the tool call ran to completion;
* a **failure label** says *why* it did not.

The label set is closed. A value that is not part of it is never silently
rewritten into one that is: it is reported as invalid instead. That rule is
what keeps a dataset honest -- a classifier is only ever evaluated against
labels it was actually trained on.
"""

from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - Literal is available from Python 3.8 onwards
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore


ExecutionStatus = Literal["success", "failure"]

ALLOWED_STATUSES: Tuple[str, ...] = ("success", "failure")

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


def is_allowed_status(status: str) -> bool:
    """Return ``True`` when ``status`` is a known execution status."""
    return status in ALLOWED_STATUSES


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


def normalize_label(label: Optional[str]) -> Optional[str]:
    """Validate a label and return it in canonical form.

    Whitespace is trimmed and an empty value is reported as ``None``
    ("not determined"). A value outside the taxonomy raises instead of being
    replaced by a valid one, so an unusable label can never be mistaken for
    a real one.

    Raises:
        ValueError: If ``label`` is not part of the trained taxonomy.
    """
    if label is None:
        return None

    text = str(label).strip()
    if not text:
        return None

    if not is_supported(text):
        raise ValueError(
            f"Invalid failure label {label!r}. "
            f"Supported labels: {', '.join(SUPPORTED_LABELS)}"
        )
    return text


def validate_labels(
    labels: Iterable[object],
    ids: Optional[Iterable[object]] = None,
    source: str = "dataset",
) -> None:
    """Check that every label is present and inside the trained taxonomy.

    A missing or unknown label is an error, never something to guess at or
    drop on the floor.

    Args:
        labels: The label values to check.
        ids: Optional identifiers used to describe offending rows.
        source: Where the labels came from, used in the error message.

    Raises:
        ValueError: If any label is missing or outside the taxonomy.
    """
    problems: List[str] = []
    identifiers: List[object] = list(ids) if ids is not None else []

    for index, label in enumerate(labels):
        identifier = (
            identifiers[index] if index < len(identifiers) else None
        )

        label_text = "" if label is None else str(label).strip()
        where = (
            f"row {identifier}" if identifier is not None else f"row {index}"
        )

        if not label_text:
            problems.append(f"{where}: missing failure label")
        elif not is_supported(label_text):
            problems.append(
                f"{where}: invalid failure label {label_text!r}; supported: "
                f"{', '.join(SUPPORTED_LABELS)}"
            )

    if problems:
        detail = "\n  - ".join(problems[:10])
        remaining = len(problems) - 10
        suffix = f"\n  ... and {remaining} more" if remaining > 0 else ""
        raise ValueError(
            f"{source} contains {len(problems)} unusable failure label(s):\n"
            f"  - {detail}{suffix}\n"
            "Labels are never guessed or replaced automatically."
        )
