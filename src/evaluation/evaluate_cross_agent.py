"""Score the trained model on the cross-agent evaluation dataset.

    python -m src.evaluation.evaluate_cross_agent

The model is **not retrained here** and the dataset is never used to
choose a feature, a hyperparameter or a split. This script loads the
already-trained text-only baseline and asks one question:

    when the same four-label taxonomy is applied to traces from four
    agent domains, how many of the labelled ones does the model get right?

Three things are reported separately, because conflating them is how a
small experiment turns into a false claim:

``metrics``
    Accuracy, per-class scores and a confusion matrix over the labelled
    cases only. Unlabelled cases have no correct answer, so counting them
    as errors would report a failure rate that is really an annotation
    gap.

``coverage``
    How much of this dataset the training data covers at all. If the
    agent type, the tool or the error text never appeared in training,
    the model is extrapolating, and the score should be read that way.

``out_of_taxonomy``
    The cases the taxonomy cannot express, with the label the model
    forced onto them. These are evidence about the taxonomy, not about
    the model.

The dataset is a hand-authored simulation of 48 cases, not captured
agent behaviour, and the report says so on every page.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.diagnosis.features import build_text
from src.diagnosis.modeling import MODEL_FILES
from src.diagnosis.predict_failure import diagnose_trace, load_model
from src.evaluation.build_cross_agent_dataset import (
    DOMAINS,
    PROVENANCE,
    summarize_cases,
)
from src.evaluation.labelling_rules import audit_cases
from src.evaluation.metrics import (
    bootstrap_accuracy_interval,
    compute_metrics,
    write_confusion_matrix,
    write_predictions,
)
from src.tracing.labels import SUPPORTED_LABELS


EVAL_FILE = Path("data/evaluation/cross_agent_eval.csv")
TRAIN_FILE = Path("data/processed/ml_dataset.csv")
REPORTS_DIR = Path("reports")

REPORT_FILE = REPORTS_DIR / "cross_agent_eval.json"
PREDICTIONS_FILE = REPORTS_DIR / "cross_agent_predictions.csv"
ERRORS_FILE = REPORTS_DIR / "cross_agent_error_analysis.csv"
COVERAGE_FILE = REPORTS_DIR / "cross_agent_coverage.csv"
CONFUSION_FILE = REPORTS_DIR / "cross_agent_confusion_matrix.csv"
AUDIT_FILE = REPORTS_DIR / "cross_agent_label_audit.csv"

MODEL_NAME = "text"

LIMITATIONS = [
    (
        "The dataset is hand-authored simulation, not captured agent "
        "behaviour. No agent was run to produce it."
    ),
    (
        "It has 48 cases, of which 33 are labelled. Accuracy on 33 cases "
        "has a wide confidence interval; the interval is reported."
    ),
    (
        "Three of the four agent types (python, web, file) never appear in "
        "the training data. Anything the model gets right there is "
        "extrapolation, not learned cross-agent behaviour."
    ),
    (
        "Fifteen cases are outside the trained taxonomy by design. They "
        "are coverage gaps, not model errors, and are excluded from "
        "accuracy."
    ),
    (
        "The taxonomy has four classes and no class for tool execution "
        "failures, so a third of this dataset cannot be scored at all."
    ),
    (
        "The dataset is never used for tuning. Improving these numbers by "
        "editing features would turn the measurement into a training set."
    ),
]


def _load_eval_frame() -> pd.DataFrame:
    """Load the cross-agent dataset, keeping empty strings empty."""
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Cross-agent dataset not found: {EVAL_FILE}. "
            "Run python -m src.evaluation.build_cross_agent_dataset first."
        )

    frame = pd.read_csv(EVAL_FILE, keep_default_na=False)

    required = {
        "task_id",
        "agent_type",
        "tool_name",
        "status",
        "error_type",
        "error",
        "failure_label",
        "in_taxonomy",
        "domain",
        "case_kind",
        "provenance",
    }

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{EVAL_FILE} is missing columns: {sorted(missing)}. "
            "Regenerate it with python -m "
            "src.evaluation.build_cross_agent_dataset."
        )

    return frame


def _is_scored(row: pd.Series) -> bool:
    """Return ``True`` when the row has a label in the trained taxonomy."""
    return str(row["failure_label"]).strip() in SUPPORTED_LABELS


def _as_trace(row: pd.Series) -> Dict[str, Any]:
    """Return the model input for one row.

    Only the five text fields are passed: everything else is metadata
    that the model must not see.
    """
    return {
        "task_id": row["task_id"],
        "agent_type": row["agent_type"],
        "tool_name": row["tool_name"],
        "status": row["status"],
        "error_type": row["error_type"],
        "error": row["error"],
    }


def coverage(train_frame: pd.DataFrame, eval_frame: pd.DataFrame) -> Dict[str, Any]:
    """Measure how much of the evaluation set the training data covers.

    A row is "covered" for a field when that exact value appears somewhere
    in the training set. The agent type and the tool name are the two that
    matter: if neither has been seen, the model has no learned behaviour
    to fall back on.

    Returns:
        Counts overall and per domain.
    """
    train_rows = train_frame.to_dict("records")

    seen_text = {build_text(row) for row in train_rows}
    seen_agents = {str(row.get("agent_type", "")) for row in train_rows}
    seen_tools = {str(row.get("tool_name", "")) for row in train_rows}
    seen_error_types = {str(row.get("error_type", "")) for row in train_rows}
    seen_errors = {str(row.get("error", "")) for row in train_rows}

    rows = []

    for _, row in eval_frame.iterrows():
        trace = _as_trace(row)

        rows.append({
            "task_id": row["task_id"],
            "domain": row["domain"],
            "agent_type_seen": trace["agent_type"] in seen_agents,
            "tool_name_seen": trace["tool_name"] in seen_tools,
            "error_type_seen": trace["error_type"] in seen_error_types,
            "error_message_seen": trace["error"] in seen_errors,
            "exact_input_seen": build_text(trace) in seen_text,
        })

    coverage_frame = pd.DataFrame(rows)

    totals = {
        "n_cases": int(len(coverage_frame)),
        "exact_input_seen_in_training": int(
            coverage_frame["exact_input_seen"].sum()
        ),
        "agent_type_seen_in_training": int(
            coverage_frame["agent_type_seen"].sum()
        ),
        "tool_name_seen_in_training": int(
            coverage_frame["tool_name_seen"].sum()
        ),
        "error_type_seen_in_training": int(
            coverage_frame["error_type_seen"].sum()
        ),
        "error_message_seen_in_training": int(
            coverage_frame["error_message_seen"].sum()
        ),
    }

    by_domain = {}

    for domain in DOMAINS:
        subset = coverage_frame[coverage_frame["domain"] == domain]

        if subset.empty:
            continue

        by_domain[domain] = {
            "n_cases": int(len(subset)),
            "agent_type_seen_in_training": int(
                subset["agent_type_seen"].sum()
            ),
            "tool_name_seen_in_training": int(
                subset["tool_name_seen"].sum()
            ),
            "error_type_seen_in_training": int(
                subset["error_type_seen"].sum()
            ),
            "error_message_seen_in_training": int(
                subset["error_message_seen"].sum()
            ),
            "exact_input_seen_in_training": int(
                subset["exact_input_seen"].sum()
            ),
        }

    return {
        "totals": totals,
        "by_domain": by_domain,
        "rows": rows,
    }


def _accuracy_rows(
    frame: pd.DataFrame,
    truth: List[str],
    predictions: List[str],
    column: str,
) -> List[dict]:
    """Accuracy per value of ``column`` over the labelled rows."""
    rows = []

    for value in sorted(frame[column].unique()):
        indices = [
            index
            for index, (_, row) in enumerate(frame.iterrows())
            if row[column] == value
        ]

        expected = [truth[index] for index in indices]
        predicted = [predictions[index] for index in indices]

        if not expected:
            continue

        correct = sum(
            1 for true, guess in zip(expected, predicted) if true == guess
        )

        rows.append({
            column: value,
            "n_cases": len(expected),
            "correct": correct,
            "accuracy": round(correct / len(expected), 4),
        })

    return rows


def _error_rows(
    frame: pd.DataFrame,
    truth: List[str],
    predictions: List[str],
    scores: List[Dict[str, float]],
    coverage_rows: Dict[str, dict],
) -> List[dict]:
    """Describe each misclassification and the novelty around it."""
    rows = []

    for index, (_, row) in enumerate(frame.iterrows()):
        if not _is_scored(row):
            continue

        expected = truth[index]
        predicted = predictions[index]

        if expected == predicted:
            continue

        covered = coverage_rows[str(row["task_id"])]

        signs = [
            field
            for field in (
                "exact_input_seen",
                "agent_type_seen",
                "tool_name_seen",
                "error_type_seen",
                "error_message_seen",
            )
            if not covered[field]
        ]

        novelty = (
            "Nothing about this trace was seen in training: "
            + ", ".join(signs)
            if signs
            else "Every field of this trace appeared in training."
        )

        rows.append({
            "task_id": row["task_id"],
            "domain": row["domain"],
            "error_type": row["error_type"],
            "error": row["error"],
            "expected_label": expected,
            "predicted_label": predicted,
            "score_of_predicted_label": round(
                float(scores[index].get(predicted, 0.0)), 4
            ),
            "analysis": (
                f"Expected {expected}, predicted {predicted}. {novelty} "
                "The label is decided by where the call failed, so this "
                "is a model error, not an annotation dispute."
            ),
        })

    return rows


def evaluate() -> Dict[str, Any]:
    """Run the evaluation and write the reports.

    Returns:
        The report dictionary, also written to :data:`REPORT_FILE`.
    """
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}. "
            "Run python -m src.evaluation.build_larger_dataset and "
            "python -m src.evaluation.prepare_dataset first."
        )

    eval_frame = _load_eval_frame()
    train_frame = pd.read_csv(TRAIN_FILE, keep_default_na=False)

    # Fail clearly instead of silently scoring an empty set.
    if not eval_frame["failure_label"].apply(
        lambda value: str(value).strip() in SUPPORTED_LABELS
    ).any():
        raise ValueError(
            f"{EVAL_FILE} contains no labelled cases; there is nothing to "
            "score."
        )

    # The model is loaded from disk, exactly as a caller would load it.
    # Nothing here fits it.
    load_model(MODEL_NAME)

    predictions: List[str] = []
    scores: List[Dict[str, float]] = []

    for _, row in eval_frame.iterrows():
        diagnosis = diagnose_trace(_as_trace(row), model=MODEL_NAME)
        predictions.append(diagnosis["predicted_label"])
        scores.append(diagnosis["class_scores"])

    truth = [
        "" if not _is_scored(row) else str(row["failure_label"]).strip()
        for _, row in eval_frame.iterrows()
    ]

    scored_indices = [
        index for index, value in enumerate(truth) if value
    ]
    scored_truth = [truth[index] for index in scored_indices]
    scored_predictions = [predictions[index] for index in scored_indices]

    metrics = compute_metrics(
        scored_truth, scored_predictions, labels=list(SUPPORTED_LABELS)
    )
    uncertainty = bootstrap_accuracy_interval(
        scored_truth, scored_predictions
    )

    coverage_result = coverage(train_frame, eval_frame)
    coverage_rows = {
        row["task_id"]: row for row in coverage_result["rows"]
    }

    scored_frame = eval_frame.loc[
        [eval_frame.index[index] for index in scored_indices]
    ]

    by_domain = _accuracy_rows(
        scored_frame, scored_truth, scored_predictions, "domain"
    )
    by_case_kind = _accuracy_rows(
        scored_frame, scored_truth, scored_predictions, "case_kind"
    )

    unscored = [
        {
            "task_id": row["task_id"],
            "domain": row["domain"],
            "error_type": row["error_type"],
            "error": row["error"],
            "predicted_label": predictions[index],
            "out_of_taxonomy_reason": row["out_of_taxonomy_reason"],
        }
        for index, (_, row) in enumerate(eval_frame.iterrows())
        if not _is_scored(row)
    ]

    predicted_unscored: Dict[str, int] = {}

    for row in unscored:
        label = row["predicted_label"]
        predicted_unscored[label] = predicted_unscored.get(label, 0) + 1

    errors = _error_rows(
        eval_frame, truth, predictions, scores, coverage_rows
    )

    prediction_rows = []

    for index, (_, row) in enumerate(eval_frame.iterrows()):
        covered = coverage_rows[str(row["task_id"])]

        prediction_rows.append({
            "task_id": row["task_id"],
            "domain": row["domain"],
            "case_kind": row["case_kind"],
            "agent_type": row["agent_type"],
            "tool_name": row["tool_name"],
            "error_type": row["error_type"],
            "expected_label": truth[index],
            "predicted_label": predictions[index],
            "correct": (
                "" if not truth[index] else truth[index] == predictions[index]
            ),
            "scored": bool(truth[index]),
            "exact_input_seen_in_training": covered["exact_input_seen"],
            "agent_type_seen_in_training": covered["agent_type_seen"],
            "tool_name_seen_in_training": covered["tool_name_seen"],
            "error_message_seen_in_training": covered[
                "error_message_seen"
            ],
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(EVAL_FILE),
            "provenance": PROVENANCE,
            "is_captured_agent_behaviour": False,
            **summarize_cases(
                eval_frame.to_dict("records")
            ),
        },
        "model": {
            "name": MODEL_NAME,
            "file": str(MODEL_FILES[MODEL_NAME]),
            "retrained_for_this_evaluation": False,
        },
        "training_dataset": {
            "path": str(TRAIN_FILE),
            "n_rows": int(len(train_frame)),
        },
        "metrics": metrics,
        "uncertainty": uncertainty,
        "by_domain": by_domain,
        "by_case_kind": by_case_kind,
        "coverage": {
            "totals": coverage_result["totals"],
            "by_domain": coverage_result["by_domain"],
        },
        "out_of_taxonomy": {
            "n_cases": len(unscored),
            "predicted_label_distribution": dict(
                sorted(predicted_unscored.items())
            ),
            "cases": unscored,
            "note": (
                "These traces have no correct label in the taxonomy. They "
                "are a coverage gap, not model errors, and are excluded "
                "from accuracy."
            ),
        },
        "error_analysis": errors,
        "limitations": LIMITATIONS,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_FILE.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    write_predictions(prediction_rows, PREDICTIONS_FILE)
    write_confusion_matrix(metrics, CONFUSION_FILE)

    pd.DataFrame(errors).to_csv(ERRORS_FILE, index=False)

    coverage_frame = pd.DataFrame(coverage_result["rows"])
    coverage_frame.to_csv(COVERAGE_FILE, index=False)

    pd.DataFrame(audit_cases(eval_frame.to_dict("records"))).to_csv(
        AUDIT_FILE, index=False
    )

    return report


def _print_report(report: Dict[str, Any]) -> None:
    """Print a readable summary, including what the numbers do not show."""
    dataset = report["dataset"]
    metrics = report["metrics"]
    uncertainty = report["uncertainty"]

    print("=" * 72)
    print("CROSS-AGENT EVALUATION")
    print("=" * 72)

    print(f"\nDataset: {dataset['path']}")
    print(f"Provenance: {dataset['provenance']}")
    print(
        "  This is hand-authored simulation, not captured agent "
        "behaviour."
    )
    print(
        f"Cases: {dataset['n_cases']} "
        f"({dataset['scored']} labelled, "
        f"{dataset['unscored']} outside the taxonomy)"
    )

    print("\nBy domain (labelled / total):")
    for domain in DOMAINS:
        total = dataset["by_domain"].get(domain, 0)
        scored = dataset["by_domain_scored"].get(domain, 0)
        print(f"  {domain:20} {scored:3d} / {total:3d}")

    print(f"\nModel: {report['model']['name']} "
          f"({report['model']['file']}), loaded, not retrained")

    print("\n--- Train/evaluation coverage ---")
    totals = report["coverage"]["totals"]

    for field, count in totals.items():
        if field == "n_cases":
            continue

        share = count / totals["n_cases"] if totals["n_cases"] else 0.0
        print(
            f"  {field:34} {count:3d}/{totals['n_cases']:3d} "
            f"({share:.1%})"
        )

    print("\n--- Accuracy on the labelled cases ---")
    print(
        f"  {metrics['accuracy']:.4f} "
        f"({sum(1 for row in report['error_analysis'])} errors), "
        f"macro F1 {metrics['macro_avg']['f1_score']:.4f}"
    )
    print(
        f"  95% bootstrap interval: [{uncertainty['ci_low']:.3f}, "
        f"{uncertainty['ci_high']:.3f}] over n={uncertainty['n_cases']}"
    )

    print("\n  Per class:")
    print(f"  {'class':20} {'prec':>7} {'rec':>7} {'f1':>7} {'supp':>6}")

    for label, scores in metrics["per_class"].items():
        print(
            f"  {label:20} {scores['precision']:7.3f} "
            f"{scores['recall']:7.3f} {scores['f1_score']:7.3f} "
            f"{scores['support']:6d}"
        )

    print("\n  By domain:")
    for row in report["by_domain"]:
        print(
            f"    {row['domain']:20} {row['correct']:3d}/{row['n_cases']:3d}"
            f"  accuracy {row['accuracy']:.3f}"
        )

    print("\n  By case kind:")
    for row in report["by_case_kind"]:
        print(
            f"    {row['case_kind']:22} {row['correct']:3d}/"
            f"{row['n_cases']:3d}  accuracy {row['accuracy']:.3f}"
        )

    out = report["out_of_taxonomy"]
    print(
        f"\n--- Outside the taxonomy: {out['n_cases']} cases "
        "(not scored) ---"
    )
    for label, count in out["predicted_label_distribution"].items():
        print(f"  forced onto {label}: {count}")

    if report["error_analysis"]:
        print("\n--- Misclassified labelled cases ---")
        for row in report["error_analysis"]:
            print(
                f"  {row['task_id']:10} {row['domain']:18} "
                f"expected {row['expected_label']:17} "
                f"predicted {row['predicted_label']}"
            )

    print("\n--- What this does not show ---")
    for limitation in report["limitations"]:
        print(f"  - {limitation}")

    print("\nReports written:")
    for path in (
        REPORT_FILE,
        PREDICTIONS_FILE,
        ERRORS_FILE,
        COVERAGE_FILE,
        CONFUSION_FILE,
        AUDIT_FILE,
    ):
        print(f"  {path}")


def main() -> None:
    _print_report(evaluate())


if __name__ == "__main__":
    main()
