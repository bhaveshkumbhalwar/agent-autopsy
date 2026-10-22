"""Build the independent, in-domain evaluation dataset.

Why this exists
---------------
The training dataset is synthetic and its rows collapse into a handful of
templates (see ``src/evaluation/leakage.py``). A held-out slice of it
therefore contains rows the model has already seen in every meaningful
respect, and a score measured on it says nothing about how the model
behaves on new input.

This dataset is the final test set of the project. It is

* **independent** -- it is never used to fit a model, choose a
  hyperparameter or select a split;
* **in-domain** -- it covers the same agent domain as training
  (a calculator tool agent), so it isolates the question "does the model
  generalise to new situations in the domain it was trained on?";
* **stressed** -- it contains tool names the training set never used and
  error messages phrased differently from the training templates.

Provenance
----------
Every row here is **hand-authored and simulated**. It was not produced by
running an agent, and no real agent execution is represented. The rows
describe situations a calculator-style tool agent can plausibly hit; the
values (arguments, timestamps) are invented. ``provenance`` records this
on every row so that a reader of the report can never mistake it for
captured agent behaviour.

How to regenerate
-----------------
    python -m src.evaluation.build_eval_dataset
"""

import csv
from pathlib import Path
from typing import Dict, List

from src.tracing.labels import SUPPORTED_LABELS, validate_labels


OUTPUT_FILE = Path("data/evaluation/in_domain_eval.csv")

PROVENANCE = "hand_authored_simulation"

# The stress a case puts on the model. Used to group the results.
CASE_KINDS = (
    "seen_template",
    "unseen_tool",
    "paraphrased_message",
    "unseen_error_type",
)


RATIONALE = {
    "none": (
        "The tool call completed, so the trace records no failure."
    ),
    "unknown_tool": (
        "The requested tool is not in the registry, so the call could not "
        "start. This is unknown_tool regardless of how the message is "
        "worded or which exception type carries it."
    ),
    "missing_argument": (
        "A required argument was absent from the call. The arguments "
        "themselves were valid, so this is not invalid_input."
    ),
    "invalid_input": (
        "An argument was supplied but had an unusable type or value, so "
        "this is invalid_input and not missing_argument."
    ),
}


def _case(
    task_id: str,
    tool_name: str,
    status: str,
    error_type: str,
    error: str,
    failure_label: str,
    case_kind: str,
    rationale: str = "",
) -> Dict[str, str]:
    return {
        "task_id": task_id,
        "agent_type": "calculator",
        "tool_name": tool_name,
        "status": status,
        "error_type": error_type,
        "error": error,
        "failure_label": failure_label,
        "case_kind": case_kind,
        "label_rationale": rationale or RATIONALE[failure_label],
        "provenance": PROVENANCE,
    }


#: The evaluation cases. Twelve per label, each stressing a different
#: aspect of the model's generalisation.
EVALUATION_CASES: List[Dict[str, str]] = [
    # --- successful executions ------------------------------------------------
    _case("eval-001", "add", "success", "", "", "none", "seen_template"),
    _case("eval-002", "multiply", "success", "", "", "none", "seen_template"),
    _case("eval-003", "modulo", "success", "", "", "none", "unseen_tool"),
    _case("eval-004", "exponentiate", "success", "", "", "none", "unseen_tool"),
    _case("eval-005", "round", "success", "", "", "none", "unseen_tool"),
    _case("eval-006", "floor", "success", "", "", "none", "unseen_tool"),
    _case("eval-007", "ceil", "success", "", "", "none", "unseen_tool"),
    _case("eval-008", "negate", "success", "", "", "none", "unseen_tool"),
    _case("eval-009", "absolute", "success", "", "", "none", "unseen_tool"),
    _case("eval-010", "logarithm", "success", "", "", "none", "unseen_tool"),
    _case("eval-011", "square_root", "success", "", "", "none", "unseen_tool"),
    _case("eval-012", "average", "success", "", "", "none", "unseen_tool"),

    # --- unknown tool ---------------------------------------------------------
    _case("eval-013", "convert", "failure", "ValueError",
          "Unknown tool: convert", "unknown_tool", "unseen_tool"),
    _case("eval-014", "transpose", "failure", "ValueError",
          "Unknown tool: transpose", "unknown_tool", "unseen_tool"),
    _case("eval-015", "plot", "failure", "ValueError",
          "Unknown tool: plot", "unknown_tool", "unseen_tool"),
    _case("eval-016", "translate", "failure", "ValueError",
          "Tool 'translate' is not registered", "unknown_tool",
          "paraphrased_message"),
    _case("eval-017", "compress", "failure", "ValueError",
          "Tool 'compress' is not registered", "unknown_tool",
          "paraphrased_message"),
    _case("eval-018", "encrypt", "failure", "ValueError",
          "unknown tool: encrypt", "unknown_tool", "paraphrased_message"),
    _case("eval-019", "schedule", "failure", "KeyError",
          "no such tool: schedule", "unknown_tool", "paraphrased_message"),
    _case("eval-020", "benchmark", "failure", "KeyError",
          "no such tool: benchmark", "unknown_tool", "paraphrased_message"),
    _case("eval-021", "annotate", "failure", "ValueError",
          "Unsupported tool 'annotate'", "unknown_tool",
          "paraphrased_message"),
    _case("eval-022", "cluster", "failure", "ValueError",
          "Unsupported tool 'cluster'", "unknown_tool",
          "paraphrased_message"),
    _case("eval-023", "render", "failure", "LookupError",
          "'render' is not a registered tool", "unknown_tool",
          "unseen_error_type"),
    _case("eval-024", "compile", "failure", "LookupError",
          "'compile' is not a registered tool", "unknown_tool",
          "unseen_error_type"),

    # --- missing argument -----------------------------------------------------
    _case("eval-025", "add", "failure", "KeyError", "'b'",
          "missing_argument", "seen_template"),
    _case("eval-026", "multiply", "failure", "KeyError", "'a'",
          "missing_argument", "seen_template"),
    _case("eval-027", "add", "failure", "KeyError", "KeyError: 'b'",
          "missing_argument", "paraphrased_message"),
    _case("eval-028", "multiply", "failure", "KeyError", "KeyError: 'a'",
          "missing_argument", "paraphrased_message"),
    _case("eval-029", "add", "failure", "KeyError",
          "required argument 'b' not provided",
          "missing_argument", "paraphrased_message"),
    _case("eval-030", "multiply", "failure", "KeyError",
          "required argument 'a' not provided",
          "missing_argument", "paraphrased_message"),
    _case("eval-031", "add", "failure", "ValueError",
          "Missing required argument: b",
          "missing_argument", "paraphrased_message"),
    _case("eval-032", "multiply", "failure", "ValueError",
          "Missing required argument: a",
          "missing_argument", "paraphrased_message"),
    _case("eval-033", "add", "failure", "TypeError",
          "add() missing 1 required positional argument: 'b'",
          "missing_argument", "unseen_error_type"),
    _case("eval-034", "multiply", "failure", "TypeError",
          "multiply() missing 1 required positional argument: 'a'",
          "missing_argument", "unseen_error_type"),
    _case("eval-035", "exponentiate", "failure", "TypeError",
          "exponentiate() missing 1 required positional argument: 'b'",
          "missing_argument", "unseen_error_type"),
    _case("eval-036", "round", "failure", "KeyError",
          "required argument 'b' not provided",
          "missing_argument", "unseen_tool"),

    # --- invalid input --------------------------------------------------------
    _case("eval-037", "add", "failure", "ValueError",
          "Arguments a and b must be numbers", "invalid_input",
          "seen_template"),
    _case("eval-038", "multiply", "failure", "ValueError",
          "Arguments a and b must be numbers", "invalid_input",
          "seen_template"),
    _case("eval-039", "add", "failure", "ValueError",
          "invalid literal for int() with base 10: 'hello'",
          "invalid_input", "paraphrased_message"),
    _case("eval-040", "multiply", "failure", "ValueError",
          "invalid literal for int() with base 10: 'world'",
          "invalid_input", "paraphrased_message"),
    _case("eval-041", "add", "failure", "TypeError",
          "unsupported operand type(s) for +: 'str' and 'int'",
          "invalid_input", "unseen_error_type"),
    _case("eval-042", "multiply", "failure", "TypeError",
          "unsupported operand type(s) for *: 'NoneType' and 'int'",
          "invalid_input", "unseen_error_type"),
    _case("eval-043", "add", "failure", "TypeError",
          "argument 'a' must be an int or float, not str",
          "invalid_input", "unseen_error_type"),
    _case("eval-044", "multiply", "failure", "TypeError",
          "argument 'b' must be an int or float, not NoneType",
          "invalid_input", "unseen_error_type"),
    _case("eval-045", "add", "failure", "ValueError",
          "argument 'a' is not a number: 'hello'",
          "invalid_input", "paraphrased_message"),
    _case("eval-046", "multiply", "failure", "ValueError",
          "argument 'b' is not a number: 'abc'",
          "invalid_input", "paraphrased_message"),
    _case("eval-047", "modulo", "failure", "TypeError",
          "unsupported operand type(s) for %: 'str' and 'int'",
          "invalid_input", "unseen_error_type"),
    _case("eval-048", "round", "failure", "ValueError",
          "argument 'a' is not a number: 'ten'",
          "invalid_input", "unseen_tool"),
]


COLUMNS = [
    "task_id",
    "agent_type",
    "tool_name",
    "status",
    "error_type",
    "error",
    "failure_label",
    "case_kind",
    "label_rationale",
    "provenance",
]


def validate_cases(cases: List[Dict[str, str]]) -> None:
    """Check the evaluation cases before they are written.

    Raises:
        ValueError: If a label is outside the taxonomy, if two cases share
            a task id, or if a case states a status the label contradicts.
    """
    seen_ids = set()

    for case in cases:
        task_id = case["task_id"]
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id in the evaluation set: {task_id}")
        seen_ids.add(task_id)

        label = case["failure_label"]
        if label not in SUPPORTED_LABELS:
            raise ValueError(
                f"{task_id}: label {label!r} is outside the trained taxonomy."
            )

        status = case["status"]
        if status not in ("success", "failure"):
            raise ValueError(f"{task_id}: invalid status {status!r}")

        if (label == "none") != (status == "success"):
            raise ValueError(
                f"{task_id}: label {label!r} contradicts status {status!r}."
            )

        if case["case_kind"] not in CASE_KINDS:
            raise ValueError(
                f"{task_id}: unknown case kind {case['case_kind']!r}."
            )

    validate_labels(
        [case["failure_label"] for case in cases],
        [case["task_id"] for case in cases],
        source="the independent evaluation dataset",
    )


def build_dataset() -> Path:
    """Write the independent evaluation dataset.

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

    print(f"Independent evaluation dataset written to: {OUTPUT_FILE}")
    print(f"Cases: {len(EVALUATION_CASES)}")

    kinds: Dict[str, int] = {}
    labels: Dict[str, int] = {}
    for case in EVALUATION_CASES:
        kinds[case["case_kind"]] = kinds.get(case["case_kind"], 0) + 1
        labels[case["failure_label"]] = labels.get(case["failure_label"], 0) + 1

    print("\nBy label:")
    for label, count in sorted(labels.items()):
        print(f"  {label:20} {count}")

    print("\nBy stress kind:")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:20} {count}")

    return OUTPUT_FILE


if __name__ == "__main__":
    build_dataset()
