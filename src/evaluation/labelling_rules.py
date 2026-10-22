"""Labelling rules: a failure label describes the *immediate* cause.

The taxonomy in :mod:`src.tracing.labels` answers one narrow question:

    what was wrong with the tool call?

It does not answer "what was wrong with the world?". Those two questions
are easy to confuse, and confusing them is how a trace gets the wrong
label. Two examples from an earlier version of this project:

* ``gen-005`` -- generated Python raised ``NameError: name 'total' is not
  defined``. The call was well formed: the tool existed and every
  argument was present and of the right type. The *code* was wrong. The
  old exception-type mapping turned ``NameError`` into ``invalid_input``,
  which blames the caller's argument for a bug in the program.
* ``gen-006`` -- ``fetch_page`` returned HTTP 404. The tool exists and was
  called correctly; the remote resource was absent. The old mapping turned
  ``HTTPError`` into ``unknown_tool``, which says the tool is not
  registered when it plainly is.

Both would have been annotations invented to make every trace fit the
taxonomy. The rule below refuses to do that.

The rule
--------
A trace is labelled by **where it failed**, not by which exception class
carried the failure:

``success``
    The call completed. Label ``none``.

``call_validation``
    The call never reached the tool body. Exactly one of:

    * ``tool_not_registered`` -- the agent asked for a tool that is not in
      its registry -> ``unknown_tool``;
    * ``argument_absent`` -- a required argument was missing ->
      ``missing_argument``;
    * ``argument_unusable`` -- an argument was supplied but its type or
      value cannot be used -> ``invalid_input``.

``tool_execution``
    The call was valid and the tool (or the code it generated, or the
    service it contacted) failed while running. There is **no label**.
    The cause may be real -- a bug, a 404, a permission error, a timeout --
    but it is not a defect in the tool call, so the taxonomy cannot
    express it. These traces are reported as a coverage gap and are never
    scored as model errors.

Precedence inside ``call_validation`` is the order above, because that is
the order an agent discovers the problems: a call to a missing tool never
gets as far as validating its arguments.

Why exception type is not enough
--------------------------------
``KeyError`` can mean "you omitted the argument that this code reads" (in
taxonomy) or "the JSON response had no ``id`` field" (out of taxonomy).
``ValueError`` can mean "you passed a string where a number was required"
(in taxonomy) or "the service rejected the request" (out of taxonomy).
The exception type narrows the search; the stage decides the label.

How the dataset uses these rules
--------------------------------
Every case in :mod:`src.evaluation.build_cross_agent_dataset` declares a
``failure_stage`` and a ``detail_kind``, and its ``failure_label`` is
*derived* from this table rather than written by hand. A case cannot
disagree with the rules, and an illegal combination raises instead of
producing a label.
"""

from typing import Dict, List, Optional, Tuple


FAILURE_STAGES: Tuple[str, ...] = (
    "success",
    "call_validation",
    "tool_execution",
)

DETAIL_KINDS: Tuple[str, ...] = (
    "success",
    "tool_not_registered",
    "argument_absent",
    "argument_unusable",
    "execution_error",
)

#: ``(failure_stage, detail_kind) -> label``. ``None`` means "no label in
#: the trained taxonomy applies"; such a case is a coverage gap, not an
#: error.
RULE_TABLE: Dict[Tuple[str, str], Optional[str]] = {
    ("success", "success"): "none",
    ("call_validation", "tool_not_registered"): "unknown_tool",
    ("call_validation", "argument_absent"): "missing_argument",
    ("call_validation", "argument_unusable"): "invalid_input",
    ("tool_execution", "execution_error"): None,
}

RULE_STATEMENTS: Dict[Tuple[str, str], str] = {
    ("success", "success"): (
        "The call completed, so the trace records no failure."
    ),
    ("call_validation", "tool_not_registered"): (
        "The requested tool is not in the agent's registry, so the call "
        "never started. The label is unknown_tool whatever exception type "
        "or wording accompanies it."
    ),
    ("call_validation", "argument_absent"): (
        "A required argument was absent. The supplied arguments were "
        "usable, so this is missing_argument rather than invalid_input."
    ),
    ("call_validation", "argument_unusable"): (
        "An argument was present but its type or value cannot be used, so "
        "this is invalid_input rather than missing_argument."
    ),
    ("tool_execution", "execution_error"): (
        "The call was valid and the tool failed while running. The "
        "underlying cause may be real, but it is not a defect in the tool "
        "call, so no label in the taxonomy applies."
    ),
}

#: The exception-type mapping this project used before the rules above
#: were written down. It is kept here only so the audit can show what it
#: got wrong; nothing derives a label from it.
NAIVE_ERROR_TYPE_LABELS: Dict[str, Optional[str]] = {
    "ValueError": "invalid_input",
    "TypeError": "invalid_input",
    "NameError": "invalid_input",
    "KeyError": "missing_argument",
    "HTTPError": "unknown_tool",
    "FileNotFoundError": "missing_argument",
    "PermissionError": "invalid_input",
    "TimeoutError": "unknown_tool",
    "OverflowError": "invalid_input",
}

#: Why the naive mapping was wrong, for the cases the audit disputes.
AUDIT_NOTES: Dict[Tuple[str, str], str] = {
    ("NameError", "tool_execution"): (
        "A NameError is raised by the code the tool ran. The call's "
        "arguments were present and usable, so this is not invalid_input."
    ),
    ("HTTPError", "tool_execution"): (
        "An HTTP error is returned by a service the tool contacted. The "
        "tool exists and was called correctly, so this is not unknown_tool."
    ),
    ("FileNotFoundError", "tool_execution"): (
        "The path was supplied and well typed; the file simply does not "
        "exist. The call was valid, so this is not missing_argument."
    ),
    ("PermissionError", "tool_execution"): (
        "The arguments were usable; the process lacked permission. This is "
        "not invalid_input."
    ),
    ("TimeoutError", "tool_execution"): (
        "The tool exists and was called correctly; the service was slow. "
        "This is not unknown_tool."
    ),
    ("KeyError", "tool_execution"): (
        "The KeyError was raised inside the tool's own data, not by an "
        "absent call argument, so this is not missing_argument."
    ),
    ("ValueError", "tool_execution"): (
        "The ValueError came from the tool or service rather than from "
        "validating an argument, so this is not invalid_input."
    ),
    ("TypeError", "tool_execution"): (
        "The TypeError came from inside the executed code, not from the "
        "call's arguments, so this is not invalid_input."
    ),
    ("OverflowError", "tool_execution"): (
        "The values were accepted and the computation overflowed, so this "
        "is not invalid_input."
    ),
}

VALID_PAIRS: Tuple[Tuple[str, str], ...] = tuple(RULE_TABLE)


def describe_rule(failure_stage: str, detail_kind: str) -> str:
    """Return the statement behind a ``(stage, kind)`` pair.

    Raises:
        ValueError: If the pair is not a rule.
    """
    key = (failure_stage, detail_kind)

    if key not in RULE_STATEMENTS:
        raise ValueError(
            f"No labelling rule for failure_stage={failure_stage!r}, "
            f"detail_kind={detail_kind!r}. Valid pairs: "
            f"{sorted(VALID_PAIRS)}"
        )

    return RULE_STATEMENTS[key]


def decide_label(failure_stage: str, detail_kind: str) -> Optional[str]:
    """Derive the failure label for a ``(stage, kind)`` pair.

    This is the only place a label is derived from the rules. Passing a
    hand-written label around is what let the earlier mapping invent
    annotations for traces the taxonomy cannot describe.

    Args:
        failure_stage: One of :data:`FAILURE_STAGES`.
        detail_kind: One of :data:`DETAIL_KINDS`.

    Returns:
        The label, or ``None`` when the taxonomy cannot express the
        failure.

    Raises:
        ValueError: If the pair is not a rule.
    """
    key = (failure_stage, detail_kind)

    if key not in RULE_TABLE:
        raise ValueError(
            f"No labelling rule for failure_stage={failure_stage!r}, "
            f"detail_kind={detail_kind!r}. Valid pairs: "
            f"{sorted(VALID_PAIRS)}"
        )

    return RULE_TABLE[key]


def is_out_of_taxonomy(failure_stage: str, detail_kind: str) -> bool:
    """Return ``True`` when no label in the taxonomy applies."""
    return decide_label(failure_stage, detail_kind) is None


def naive_label_for(error_type: str, status: str) -> Optional[str]:
    """Return the label the old exception-type mapping would have chosen.

    Kept for the audit only. It is not used to label anything.

    Args:
        error_type: The exception class recorded on the trace.
        status: ``"success"`` or ``"failure"``.

    Returns:
        ``"none"`` for a successful call, otherwise the mapped label, or
        ``None`` when the exception type was not in the old mapping.
    """
    if status == "success":
        return "none"

    return NAIVE_ERROR_TYPE_LABELS.get(str(error_type).strip())


def audit_case(case: dict) -> dict:
    """Compare a case's declared label with the rules and the old mapping.

    Args:
        case: A case with ``status``, ``error_type``, ``failure_stage``,
            ``detail_kind`` and an optional ``expected_label``.

    Returns:
        One audit row.

    Raises:
        ValueError: If the case's stage/kind pair is not a rule.
    """
    stage = case["failure_stage"]
    kind = case["detail_kind"]

    rule_label = decide_label(stage, kind)
    declared = case.get("expected_label", case.get("failure_label"))

    if declared in ("", None):
        declared = None

    naive = naive_label_for(case.get("error_type", ""), case["status"])
    note = ""

    if naive != rule_label and (case.get("error_type", ""), stage) in (
        AUDIT_NOTES
    ):
        note = AUDIT_NOTES[(case["error_type"], stage)]

    return {
        "task_id": case.get("task_id", ""),
        "error_type": case.get("error_type", ""),
        "failure_stage": stage,
        "detail_kind": kind,
        "exception_type_would_say": "" if naive is None else naive,
        "rules_say": "" if rule_label is None else rule_label,
        "declared_label": "" if declared is None else declared,
        "rules_agree_with_declared": rule_label == declared,
        "exception_type_agrees_with_rules": naive == rule_label,
        "note": note,
    }


def audit_cases(cases: List[dict]) -> List[dict]:
    """Audit a list of cases. See :func:`audit_case`."""
    return [audit_case(case) for case in cases]
