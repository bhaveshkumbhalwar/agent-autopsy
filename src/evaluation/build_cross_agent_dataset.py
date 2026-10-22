"""Build the cross-agent evaluation dataset.

Why this exists
---------------
The training data and the in-domain test set both describe a *calculator*
tool agent. They cannot tell us anything about the model on a trace from a
different kind of agent, because the model has never seen one. This dataset
puts that question on record: the same four-label taxonomy, applied to
traces from four agent domains, of which three never appear in training.

What is in it
-------------
Twelve cases per domain:

``calculator``
    The domain the model was trained on. Expected to be the easiest.
``python_execution``
    An agent that writes and runs Python. Includes the ``NameError`` case
    an earlier version of this project mislabelled as ``invalid_input``.
``web``
    An agent that fetches and searches. Includes the HTTP-404 case an
    earlier version mislabelled as ``unknown_tool``.
``file_operations``
    An agent that reads and writes files. Includes filesystem failures
    that are not defects in the tool call.

Every case declares ``failure_stage`` and ``detail_kind``, and its
``failure_label`` is *derived* from
:mod:`src.evaluation.labelling_rules`. A case cannot carry a label that
contradicts the rules.

What it is not
--------------
This is **hand-authored simulation**, not captured agent behaviour. No
agent was run to produce these rows; the errors are realistic strings
written by hand. ``provenance`` says so on every row, and the report
repeats it. It is also small (48 cases), so the accuracy it measures has
a wide confidence interval. It is evidence about *these cases*, not
evidence of broad cross-agent generalisation.

Training isolation
------------------
This dataset is never read by a training script and never used to choose
a feature or a hyperparameter. ``src/evaluation/evaluate_cross_agent.py``
only loads an already-trained model and scores it.

How to regenerate
-----------------
    python -m src.evaluation.build_cross_agent_dataset
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.labelling_rules import (
    DETAIL_KINDS,
    FAILURE_STAGES,
    decide_label,
    describe_rule,
)
from src.tracing.labels import SUPPORTED_LABELS, is_supported


OUTPUT_FILE = Path("data/evaluation/cross_agent_eval.csv")

PROVENANCE = "hand_authored_simulation"

DOMAINS = (
    "calculator",
    "python_execution",
    "web",
    "file_operations",
)

CASE_KINDS = (
    "in_domain_transfer",
    "new_agent",
    "paraphrased_message",
    "out_of_taxonomy",
)

COLUMNS = (
    "task_id",
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
    "failure_stage",
    "detail_kind",
    "failure_label",
    "in_taxonomy",
    "out_of_taxonomy_reason",
    "domain",
    "case_kind",
    "label_rationale",
    "provenance",
)


def _case(
    task_id: str,
    domain: str,
    agent_type: str,
    tool_name: str,
    status: str,
    error_type: str,
    error: str,
    failure_stage: str,
    detail_kind: str,
    case_kind: str,
) -> Dict[str, Any]:
    """Build one case, deriving its label from the shared rules."""
    label = decide_label(failure_stage, detail_kind)

    out_of_taxonomy_reason = ""

    if label is None:
        out_of_taxonomy_reason = (
            "The call was valid and the tool failed while running, so no "
            "label in the trained taxonomy applies. "
            + describe_rule(failure_stage, detail_kind)
        )

    return {
        "task_id": task_id,
        "agent_type": agent_type,
        "tool_name": tool_name,
        "status": status,
        "error_type": error_type,
        "error": error,
        "failure_stage": failure_stage,
        "detail_kind": detail_kind,
        "failure_label": "" if label is None else label,
        "in_taxonomy": label is not None,
        "out_of_taxonomy_reason": out_of_taxonomy_reason,
        "domain": domain,
        "case_kind": case_kind,
        "label_rationale": describe_rule(failure_stage, detail_kind),
        "provenance": PROVENANCE,
    }


#: The evaluation cases: twelve per domain, nine of them labelled.
EVALUATION_CASES: List[Dict[str, Any]] = [
    # --- calculator: the domain the model was trained on --------------------
    _case("calc-001", "calculator", "calculator", "add", "success", "", "",
          "success", "success", "in_domain_transfer"),
    _case("calc-002", "calculator", "calculator", "square_root", "success",
          "", "", "success", "success", "in_domain_transfer"),
    _case("calc-003", "calculator", "calculator", "power", "failure",
          "ValueError", "Unknown tool: power", "call_validation",
          "tool_not_registered", "in_domain_transfer"),
    _case("calc-004", "calculator", "calculator", "modulo", "failure",
          "ValueError", "no such tool: modulo", "call_validation",
          "tool_not_registered", "paraphrased_message"),
    _case("calc-005", "calculator", "calculator", "add", "failure", "KeyError",
          "'b'", "call_validation", "argument_absent",
          "in_domain_transfer"),
    _case("calc-006", "calculator", "calculator", "multiply", "failure",
          "TypeError", "multiply() missing 1 required positional argument: 'b'",
          "call_validation", "argument_absent", "paraphrased_message"),
    _case("calc-007", "calculator", "calculator", "add", "failure",
          "ValueError", "Arguments a and b must be numbers", "call_validation",
          "argument_unusable", "in_domain_transfer"),
    _case("calc-008", "calculator", "calculator", "divide", "failure",
          "TypeError", "unsupported operand type(s) for /: 'str' and 'int'",
          "call_validation", "argument_unusable", "paraphrased_message"),
    _case("calc-009", "calculator", "calculator", "power", "failure",
          "OverflowError", "math range error", "tool_execution",
          "execution_error", "out_of_taxonomy"),
    _case("calc-010", "calculator", "calculator", "power", "failure",
          "RecursionError", "maximum recursion depth exceeded",
          "tool_execution", "execution_error", "out_of_taxonomy"),
    _case("calc-011", "calculator", "calculator", "add", "failure",
          "MemoryError", "Unable to allocate 8.00 GiB for an array",
          "tool_execution", "execution_error", "out_of_taxonomy"),
    _case("calc-012", "calculator", "calculator", "average", "success", "",
          "", "success", "success", "in_domain_transfer"),

    # --- python execution: a new agent ------------------------------------
    _case("py-001", "python_execution", "python", "execute_code", "success",
          "", "", "success", "success", "new_agent"),
    _case("py-002", "python_execution", "python", "execute_code", "success",
          "", "", "success", "success", "new_agent"),
    _case("py-003", "python_execution", "python", "run_python", "failure",
          "ValueError", "Unknown tool: run_python", "call_validation",
          "tool_not_registered", "new_agent"),
    _case("py-004", "python_execution", "python", "eval_expression", "failure",
          "ValueError", "Tool 'eval_expression' is not registered",
          "call_validation", "tool_not_registered", "paraphrased_message"),
    _case("py-005", "python_execution", "python", "execute_code", "failure",
          "TypeError",
          "execute_code() missing 1 required positional argument: 'code'",
          "call_validation", "argument_absent", "new_agent"),
    _case("py-006", "python_execution", "python", "execute_code", "failure",
          "KeyError", "'code'", "call_validation", "argument_absent",
          "paraphrased_message"),
    _case("py-007", "python_execution", "python", "execute_code", "failure",
          "TypeError", "code must be a string, got list", "call_validation",
          "argument_unusable", "new_agent"),
    _case("py-008", "python_execution", "python", "execute_code", "failure",
          "ValueError", "Unsupported language: 'ruby'. Expected 'python'.",
          "call_validation", "argument_unusable", "paraphrased_message"),
    _case("py-009", "python_execution", "python", "execute_code", "failure",
          "NameError", "name 'total' is not defined", "tool_execution",
          "execution_error", "out_of_taxonomy"),
    _case("py-010", "python_execution", "python", "execute_code", "failure",
          "TypeError", "unsupported operand type(s) for +: 'int' and 'str'",
          "tool_execution", "execution_error", "out_of_taxonomy"),
    _case("py-011", "python_execution", "python", "execute_code", "failure",
          "IndexError", "list index out of range", "tool_execution",
          "execution_error", "out_of_taxonomy"),
    _case("py-012", "python_execution", "python", "execute_code", "failure",
          "TimeoutError", "Execution timed out after 30s", "tool_execution",
          "execution_error", "out_of_taxonomy"),

    # --- web: a new agent -------------------------------------------------
    _case("web-001", "web", "web", "search", "success", "", "", "success",
          "success", "new_agent"),
    _case("web-002", "web", "web", "fetch_page", "success", "", "", "success",
          "success", "new_agent"),
    _case("web-003", "web", "web", "download_file", "failure", "ValueError",
          "Unknown tool: download_file", "call_validation",
          "tool_not_registered", "new_agent"),
    _case("web-004", "web", "web", "browse", "failure", "ValueError",
          "Unsupported tool 'browse'", "call_validation",
          "tool_not_registered", "paraphrased_message"),
    _case("web-005", "web", "web", "fetch_page", "failure", "KeyError",
          "'url'", "call_validation", "argument_absent", "new_agent"),
    _case("web-006", "web", "web", "fetch_page", "failure", "TypeError",
          "fetch_page() missing 1 required positional argument: 'url'",
          "call_validation", "argument_absent", "paraphrased_message"),
    _case("web-007", "web", "web", "fetch_page", "failure", "TypeError",
          "url must be a string, got int", "call_validation",
          "argument_unusable", "new_agent"),
    _case("web-008", "web", "web", "fetch_page", "failure", "ValueError",
          "Invalid URL scheme: 'ftp'", "call_validation",
          "argument_unusable", "paraphrased_message"),
    _case("web-009", "web", "web", "fetch_page", "failure", "HTTPError",
          "404 Client Error: Not Found for url: https://example.com/missing",
          "tool_execution", "execution_error", "out_of_taxonomy"),
    _case("web-010", "web", "web", "fetch_page", "failure", "HTTPError",
          "500 Server Error: Internal Server Error", "tool_execution",
          "execution_error", "out_of_taxonomy"),
    _case("web-011", "web", "web", "search", "failure", "ConnectionError",
          "connection reset by peer", "tool_execution", "execution_error",
          "out_of_taxonomy"),
    _case("web-012", "web", "web", "fetch_page", "failure", "SSLError",
          "certificate verify failed", "tool_execution", "execution_error",
          "out_of_taxonomy"),

    # --- file operations: a new agent -------------------------------------
    _case("file-001", "file_operations", "file", "read_file", "success", "",
          "", "success", "success", "new_agent"),
    _case("file-002", "file_operations", "file", "write_file", "success", "",
          "", "success", "success", "new_agent"),
    _case("file-003", "file_operations", "file", "move_file", "failure",
          "ValueError", "Unknown tool: move_file", "call_validation",
          "tool_not_registered", "new_agent"),
    _case("file-004", "file_operations", "file", "copy_file", "failure",
          "ValueError", "Tool 'copy_file' is not registered", "call_validation",
          "tool_not_registered", "paraphrased_message"),
    _case("file-005", "file_operations", "file", "read_file", "failure",
          "KeyError", "'path'", "call_validation", "argument_absent",
          "new_agent"),
    _case("file-006", "file_operations", "file", "write_file", "failure",
          "TypeError", "write_file() missing 1 required positional argument: "
          "'path'", "call_validation", "argument_absent",
          "paraphrased_message"),
    _case("file-007", "file_operations", "file", "read_file", "failure",
          "TypeError", "path must be a string, got int", "call_validation",
          "argument_unusable", "new_agent"),
    _case("file-008", "file_operations", "file", "write_file", "failure",
          "ValueError", "Unsupported mode: 'banana'", "call_validation",
          "argument_unusable", "paraphrased_message"),
    _case("file-009", "file_operations", "file", "read_file", "failure",
          "FileNotFoundError", "No such file or directory: 'notes.txt'",
          "tool_execution", "execution_error", "out_of_taxonomy"),
    _case("file-010", "file_operations", "file", "write_file", "failure",
          "PermissionError", "Permission denied: '/etc/hosts'",
          "tool_execution", "execution_error", "out_of_taxonomy"),
    _case("file-011", "file_operations", "file", "write_file", "failure",
          "OSError", "No space left on device", "tool_execution",
          "execution_error", "out_of_taxonomy"),
    _case("file-012", "file_operations", "file", "list_directory", "failure",
          "NotADirectoryError", "Not a directory: 'notes.txt'",
          "tool_execution", "execution_error", "out_of_taxonomy"),
]


def validate_cases(cases: List[Dict[str, Any]]) -> None:
    """Check the dataset before it is written.

    Raises:
        ValueError: If a task id repeats, a stage/kind pair is not a rule,
            the derived label disagrees with the rule, the status
            contradicts the label, or a case is missing its provenance.
    """
    seen_ids = set()

    for case in cases:
        task_id = case["task_id"]

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen_ids.add(task_id)

        if case["domain"] not in DOMAINS:
            raise ValueError(
                f"{task_id}: unknown domain {case['domain']!r}"
            )

        if case["case_kind"] not in CASE_KINDS:
            raise ValueError(
                f"{task_id}: unknown case kind {case['case_kind']!r}"
            )

        if case["failure_stage"] not in FAILURE_STAGES:
            raise ValueError(
                f"{task_id}: unknown failure stage "
                f"{case['failure_stage']!r}"
            )

        if case["detail_kind"] not in DETAIL_KINDS:
            raise ValueError(
                f"{task_id}: unknown detail kind {case['detail_kind']!r}"
            )

        derived = decide_label(case["failure_stage"], case["detail_kind"])
        declared = case["failure_label"] or None

        if derived != declared:
            raise ValueError(
                f"{task_id}: label {declared!r} contradicts the labelling "
                f"rule for ({case['failure_stage']!r}, "
                f"{case['detail_kind']!r}), which gives {derived!r}."
            )

        if case["status"] not in ("success", "failure"):
            raise ValueError(
                f"{task_id}: invalid status {case['status']!r}"
            )

        if (declared == "none") != (case["status"] == "success"):
            raise ValueError(
                f"{task_id}: label {declared!r} contradicts status "
                f"{case['status']!r}"
            )

        if declared is not None:
            if not is_supported(declared):
                raise ValueError(
                    f"{task_id}: label {declared!r} is outside the taxonomy"
                )
            if case["out_of_taxonomy_reason"]:
                raise ValueError(
                    f"{task_id}: a labelled case must not carry an "
                    "out_of_taxonomy_reason"
                )
        elif not case["out_of_taxonomy_reason"]:
            raise ValueError(
                f"{task_id}: an unlabelled case must explain why no label "
                "applies"
            )

        if case["provenance"] != PROVENANCE:
            raise ValueError(
                f"{task_id}: provenance must be {PROVENANCE!r}"
            )

    for label in SUPPORTED_LABELS:
        if not any(case["failure_label"] == label for case in cases):
            raise ValueError(f"The dataset contains no {label!r} case")


def summarize_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return counts by domain, label and case kind."""
    by_domain: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    by_domain_scored: Dict[str, int] = {}

    for case in cases:
        by_domain[case["domain"]] = by_domain.get(case["domain"], 0) + 1
        by_kind[case["case_kind"]] = by_kind.get(case["case_kind"], 0) + 1

        label = case["failure_label"] or "out_of_taxonomy"
        by_label[label] = by_label.get(label, 0) + 1

        if case["failure_label"]:
            by_domain_scored[case["domain"]] = (
                by_domain_scored.get(case["domain"], 0) + 1
            )

    return {
        "n_cases": len(cases),
        "scored": sum(1 for case in cases if case["failure_label"]),
        "unscored": sum(1 for case in cases if not case["failure_label"]),
        "by_domain": dict(sorted(by_domain.items())),
        "by_domain_scored": dict(sorted(by_domain_scored.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_case_kind": dict(sorted(by_kind.items())),
    }


def build_dataset() -> Path:
    """Write the cross-agent evaluation dataset.

    Returns:
        The path the dataset was written to.
    """
    validate_cases(EVALUATION_CASES)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()

        for case in EVALUATION_CASES:
            writer.writerow(case)

    summary = summarize_cases(EVALUATION_CASES)

    print(f"Cross-agent evaluation dataset written to: {OUTPUT_FILE}")
    print(f"Provenance: {PROVENANCE} (hand-authored, not captured)")
    print(
        f"Cases: {summary['n_cases']} "
        f"({summary['scored']} labelled, "
        f"{summary['unscored']} outside the taxonomy)"
    )

    print("\nBy domain (labelled / total):")
    for domain in DOMAINS:
        print(
            f"  {domain:20} {summary['by_domain_scored'].get(domain, 0):3d} /"
            f" {summary['by_domain'][domain]:3d}"
        )

    print("\nBy label:")
    for label, count in summary["by_label"].items():
        print(f"  {label:20} {count}")

    print("\nBy case kind:")
    for kind, count in summary["by_case_kind"].items():
        print(f"  {kind:20} {count}")

    return OUTPUT_FILE


if __name__ == "__main__":
    build_dataset()
