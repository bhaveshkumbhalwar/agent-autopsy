"""Evaluate the trained classifier on held-out traces and report coverage.

The suite deliberately contains two kinds of case:

* **Scored** cases -- the failure mode has a class in the trained taxonomy
  (see ``src/tracing/labels.py``), so the classifier's prediction is
  compared against a known ground truth.
* **Out-of-taxonomy** cases -- the trace failed in a way the trained taxonomy
  cannot express, for example a ``NameError`` raised inside generated code or
  an HTTP 404 returned by a tool. There is no correct label for these, so
  they are reported (with the label the model forced onto them) but excluded
  from the accuracy figure. Counting them as model errors would report a
  false failure rate.

An earlier version of this file assigned every case an expected label by
analogy to the nearest training class. That mislabelled the out-of-taxonomy
cases and made the suite report model errors that were actually annotation
errors. The rules that replaced it are documented in
:mod:`src.evaluation.labelling_rules` and applied by
:mod:`src.evaluation.audit_generalization_labels`:

* the label describes the **immediate** cause, not the underlying one;
* a call that was never validated (missing tool, absent argument, unusable
  argument) has a label;
* a call that was valid and then failed while running (a ``NameError`` in
  generated code, an HTTP 404 from a real tool) has **no** label, even
  though the underlying cause is real.

Each case therefore records ``failure_stage`` and ``detail_kind``, and the
audit derives the label from them. A case cannot quietly carry a label
that contradicts the rules.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.diagnosis.predict_failure import diagnose_trace
from src.tracing.labels import is_supported


OUTPUT_FILE = Path("reports/generalization_results.csv")


# Held-out cases. These are not copied from the training CSV.
#
# ``expected_label`` is None when the trace's failure mode has no class in
# the trained taxonomy. Those cases must carry an
# ``out_of_taxonomy_reason`` explaining why no label applies.
#
# ``failure_stage`` and ``detail_kind`` describe where the call failed and
# are the only inputs the label is derived from. See
# ``src/evaluation/labelling_rules.py``.
TEST_CASES: List[Dict[str, Any]] = [
    {
        "task_id": "gen-001",
        "agent_type": "calculator",
        "tool_name": "power",
        "status": "failure",
        "error_type": "ValueError",
        "error": "Unknown tool: power",
        "failure_stage": "call_validation",
        "detail_kind": "tool_not_registered",
        "expected_label": "unknown_tool",
    },
    {
        "task_id": "gen-002",
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "failure_stage": "call_validation",
        "detail_kind": "argument_absent",
        "expected_label": "missing_argument",
    },
    {
        "task_id": "gen-003",
        "agent_type": "calculator",
        "tool_name": "multiply",
        "status": "failure",
        "error_type": "ValueError",
        "error": "Arguments a and b must be numbers",
        "failure_stage": "call_validation",
        "detail_kind": "argument_unusable",
        "expected_label": "invalid_input",
    },
    {
        "task_id": "gen-004",
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "success",
        "error_type": "",
        "error": "",
        "failure_stage": "success",
        "detail_kind": "success",
        "expected_label": "none",
    },
    {
        "task_id": "gen-005",
        "agent_type": "python",
        "tool_name": "execute_code",
        "status": "failure",
        "error_type": "NameError",
        "error": "name 'total' is not defined",
        "failure_stage": "tool_execution",
        "detail_kind": "execution_error",
        "expected_label": None,
        "out_of_taxonomy_reason": (
            "NameError raised inside generated code is a runtime error. The "
            "trained taxonomy covers tool selection, missing arguments and "
            "argument types only, so no label applies."
        ),
    },
    {
        "task_id": "gen-006",
        "agent_type": "web",
        "tool_name": "fetch_page",
        "status": "failure",
        "error_type": "HTTPError",
        "error": "The requested page could not be found",
        "failure_stage": "tool_execution",
        "detail_kind": "execution_error",
        "expected_label": None,
        "out_of_taxonomy_reason": (
            "A 404 from fetch_page is a tool execution error, not an unknown "
            "tool: fetch_page is a registered tool that returned an error. "
            "The trained taxonomy has no class for upstream failures."
        ),
    },
]


def build_result_row(
    case: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    """Turn one evaluated case into a result row.

    Args:
        case: An entry of ``TEST_CASES``.
        prediction: The output of ``diagnose_trace`` for that case.

    Returns:
        A row with ``scored`` set to ``False`` and ``correct`` left blank
        when the case falls outside the trained taxonomy.

    Raises:
        ValueError: If a case declares an expected label that is not part of
            the trained taxonomy.
    """
    expected: Optional[str] = case.get("expected_label")
    scored = expected is not None

    if scored and not is_supported(expected):
        raise ValueError(
            f"Case {case.get('task_id')!r} expects label {expected!r}, which "
            "is not in the trained taxonomy. Use expected_label=None with an "
            "out_of_taxonomy_reason for cases the taxonomy cannot express."
        )

    predicted = prediction["predicted_label"]

    return {
        "task_id": case.get("task_id", "unknown"),
        "expected_label": expected or "",
        "predicted_label": predicted,
        "correct": (expected == predicted) if scored else "",
        "scored": scored,
        "out_of_taxonomy_reason": case.get("out_of_taxonomy_reason", ""),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize evaluation rows without hiding out-of-taxonomy cases.

    The accuracy is computed over scored cases only. ``coverage`` reports how
    much of the suite the trained taxonomy was able to express at all.
    """
    scored = [row for row in rows if row["scored"]]
    unscored = [row for row in rows if not row["scored"]]
    correct = sum(1 for row in scored if bool(row["correct"]))

    return {
        "total": len(rows),
        "scored": len(scored),
        "unscored": len(unscored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "coverage": (len(scored) / len(rows)) if rows else None,
    }


def main() -> Dict[str, Any]:
    """Run the generalization suite and write the results CSV."""
    rows = [
        build_result_row(case, diagnose_trace(case))
        for case in TEST_CASES
    ]

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    summary = summarize(rows)

    scored_df = df[df["scored"]]
    unscored_df = df[~df["scored"]]

    print("\nGeneralization results (scored cases):")
    print(scored_df[
        ["task_id", "expected_label", "predicted_label", "correct"]
    ].to_string(index=False))

    if not unscored_df.empty:
        print("\nOutside the trained taxonomy (reported, not scored):")
        print(unscored_df[
            ["task_id", "predicted_label", "out_of_taxonomy_reason"]
        ].to_string(index=False))

    accuracy = summary["accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.1%}"
    coverage = summary["coverage"]
    coverage_text = "n/a" if coverage is None else f"{coverage:.1%}"

    print(
        f"\nAccuracy on scored cases: "
        f"{summary['correct']}/{summary['scored']} = {accuracy_text}"
    )
    print(
        f"Taxonomy coverage: {summary['scored']}/{summary['total']} "
        f"cases have a label in the trained taxonomy ({coverage_text})"
    )
    print(f"Results saved to: {OUTPUT_FILE}")

    return summary


if __name__ == "__main__":
    main()
