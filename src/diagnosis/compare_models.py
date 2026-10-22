"""Compare the text-only baseline with the combined model.

    python -m src.diagnosis.compare_models

Both models are scored on the same three splits, so the only thing that
differs between them is the five structured columns:

``group_cross_validation``
    Grouped folds with pooled out-of-fold predictions. No row is scored
    by a model that saw its template. This is the result to quote.

``stratified_random_split``
    The split the project used before the leakage audit. Kept only so the
    size of the illusion is on record.

``independent_evaluation_set``
    The final test set: separately authored cases with unseen tools and
    rephrased messages. It is scored once, for both models, and no
    decision in this project is based on it.

The report also measures what the structured features actually are,
because that decides whether the combined model *can* help: if a
structured feature is a near-copy of the label in the training data, it
adds no information there, and if it only matches one exact phrase it
will miss every paraphrase at inference time.

If the combined model is not better, this script reports that. It does
not tune anything to make it better.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.diagnosis.features import (
    STRUCTURED_FEATURE_NAMES,
    build_structured_features,
)
from src.diagnosis.modeling import (
    MODEL_DESCRIPTIONS,
    build_pipeline,
    feature_configuration,
)
from src.evaluation.metrics import (
    bootstrap_accuracy_interval,
    mcnemar_exact_test,
    write_confusion_matrix,
    write_predictions,
)
from src.evaluation.runner import (
    by_case_kind,
    cross_validate,
    evaluate_dataset,
    evaluate_holdout,
)
from src.evaluation.splitting import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    TARGET_COLUMN,
    split_frame,
    stratified_split,
)
from src.diagnosis.training import DATA_FILE


INDEPENDENT_FILE = Path("data/evaluation/in_domain_eval.csv")

REPORTS_DIR = Path("reports")

COMPARISON_FILE = REPORTS_DIR / "model_comparison.json"
STRUCTURED_FILE = REPORTS_DIR / "model_comparison_structured_features.json"
BY_CASE_KIND_FILE = REPORTS_DIR / "model_comparison_by_case_kind.csv"


def confusion_file(model: str, split: str) -> Path:
    """Return the confusion-matrix path for one model and one split."""
    return REPORTS_DIR / f"model_comparison_confusion_matrix_{model}_{split}.csv"


def predictions_file(model: str, split: str) -> Path:
    """Return the predictions path for one model and one split."""
    return REPORTS_DIR / f"model_comparison_predictions_{model}_{split}.csv"


def structured_feature_diagnostics(df: pd.DataFrame) -> List[dict]:
    """Measure how informative each structured feature is.

    For every feature this reports how often it fires, and what the labels
    look like when it does and when it does not. A feature that fires
    almost only for one label is a copy of the label rather than a
    general signal; a feature that never fires is dead weight.
    """
    traces = df.to_dict("records")
    labels = df[TARGET_COLUMN].to_numpy()

    values = {
        name: np.array([
            bool(build_structured_features(trace)[name]) for trace in traces
        ])
        for name in STRUCTURED_FEATURE_NAMES
    }

    rows = []

    for name in STRUCTURED_FEATURE_NAMES:
        column = values[name]
        n_true = int(column.sum())

        when_true: Dict[str, int] = {}
        when_false: Dict[str, int] = {}

        for label, fired in zip(labels, column):
            bucket = when_true if fired else when_false
            bucket[str(label)] = bucket.get(str(label), 0) + 1

        rows.append({
            "feature": name,
            "n_true": n_true,
            "n_false": int(len(column) - n_true),
            "always_fires": n_true == len(column),
            "never_fires": n_true == 0,
            "labels_when_true": dict(sorted(when_true.items())),
            "labels_when_false": dict(sorted(when_false.items())),
            "distinct_labels_when_true": len(when_true),
        })

    return rows


def structured_coefficients() -> List[dict]:
    """Return the learned weight of each structured feature per class.

    The pipeline is fitted on the whole training set here, exactly as the
    shipped model is, and the structured columns are the last ones in the
    feature matrix (they are appended after the TF-IDF columns).
    """
    df = pd.read_csv(DATA_FILE)
    traces = df.to_dict("records")

    model = build_pipeline("combined")
    model.fit(traces, df[TARGET_COLUMN].to_numpy())

    classifier = model.named_steps["classifier"]
    names = list(model.named_steps["features"].get_feature_names_out())

    start = len(names) - len(STRUCTURED_FEATURE_NAMES)

    rows = []

    for label, weights in zip(classifier.classes_, classifier.coef_):
        for offset, feature in enumerate(STRUCTURED_FEATURE_NAMES):
            rows.append({
                "class": str(label),
                "feature": feature,
                "coefficient": round(float(weights[start + offset]), 6),
            })

    return rows


def _delta(
    baseline: dict,
    combined: dict,
    changed_ids: List[str],
    truth: List[str],
) -> dict:
    """Summarise how the combined model differs from the baseline."""
    baseline_metrics = baseline["metrics"]
    combined_metrics = combined["metrics"]

    per_class = {}

    for label, scores in baseline_metrics["per_class"].items():
        other = combined_metrics["per_class"][label]
        per_class[label] = {
            "baseline_f1": scores["f1_score"],
            "combined_f1": other["f1_score"],
            "delta_f1": round(
                other["f1_score"] - scores["f1_score"], 6
            ),
        }

    baseline_predictions = [
        row["predicted_label"] for row in baseline["predictions"]
    ]
    combined_predictions = [
        row["predicted_label"] for row in combined["predictions"]
    ]

    return {
        "baseline_accuracy": baseline_metrics["accuracy"],
        "combined_accuracy": combined_metrics["accuracy"],
        "delta_accuracy": round(
            combined_metrics["accuracy"] - baseline_metrics["accuracy"], 6
        ),
        "baseline_macro_f1": baseline_metrics["macro_avg"]["f1_score"],
        "combined_macro_f1": combined_metrics["macro_avg"]["f1_score"],
        "delta_macro_f1": round(
            combined_metrics["macro_avg"]["f1_score"]
            - baseline_metrics["macro_avg"]["f1_score"],
            6,
        ),
        "per_class": per_class,
        "n_rows_where_prediction_changed": len(changed_ids),
        "task_ids_where_prediction_changed": changed_ids,
        "uncertainty": {
            "text": bootstrap_accuracy_interval(
                truth, baseline_predictions
            ),
            "combined": bootstrap_accuracy_interval(
                truth, combined_predictions
            ),
            "mcnemar": mcnemar_exact_test(
                truth, baseline_predictions, combined_predictions
            ),
        },
    }


def _changed_task_ids(baseline: dict, combined: dict) -> List[str]:
    """Return the rows whose prediction differs between the two models."""
    changed = []

    for first, second in zip(
        baseline["predictions"], combined["predictions"]
    ):
        if first["task_id"] != second["task_id"]:
            raise ValueError(
                "The two models were scored on different rows: "
                f"{first['task_id']} and {second['task_id']}."
            )

        if first["predicted_label"] != second["predicted_label"]:
            changed.append(str(first["task_id"]))

    return changed


def _write_split_reports(
    model: str,
    split_name: str,
    result: dict,
) -> None:
    """Write the confusion matrix and predictions of one result."""
    write_confusion_matrix(
        result["metrics"], confusion_file(model, split_name)
    )
    write_predictions(
        result["predictions"], predictions_file(model, split_name)
    )


def compare() -> dict:
    """Score both models on the same splits and write the reports.

    Returns:
        The comparison dictionary, which is also written to
        :data:`COMPARISON_FILE`.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run python -m src.evaluation.build_larger_dataset and "
            "python -m src.evaluation.prepare_dataset first."
        )

    if not INDEPENDENT_FILE.exists():
        raise FileNotFoundError(
            f"Independent evaluation set not found: {INDEPENDENT_FILE}. "
            "Run python -m src.evaluation.build_eval_dataset first."
        )

    df = pd.read_csv(DATA_FILE)
    independent = pd.read_csv(INDEPENDENT_FILE)

    naive = stratified_split(
        df, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE
    )

    _, naive_test_frame = split_frame(df, naive)

    truths = {
        "group_cross_validation": df[TARGET_COLUMN].tolist(),
        "stratified_random_split": naive_test_frame[TARGET_COLUMN].tolist(),
        "independent_evaluation_set": independent[TARGET_COLUMN].tolist(),
    }

    results: Dict[str, dict] = {}

    for name in ("text", "combined"):
        factory = lambda name=name: build_pipeline(name)

        grouped = cross_validate(factory, df, seed=DEFAULT_SEED)
        heldout = evaluate_holdout(factory, df, naive)
        final = evaluate_dataset(
            factory, df, independent, extra_columns=["case_kind"]
        )

        predictions = [row["predicted_label"] for row in final["predictions"]]

        results[name] = {
            "description": MODEL_DESCRIPTIONS[name],
            "group_cross_validation": grouped,
            "stratified_random_split": heldout,
            "independent_evaluation_set": {
                "dataset": str(INDEPENDENT_FILE),
                "n_cases": int(len(independent)),
                "provenance": (
                    "hand-authored simulation; not captured agent "
                    "behaviour"
                ),
                "metrics": final["metrics"],
                "by_case_kind": by_case_kind(independent, predictions),
                "predictions": final["predictions"],
            },
        }

    comparison = {}

    for split_name in (
        "group_cross_validation",
        "stratified_random_split",
        "independent_evaluation_set",
    ):
        baseline = results["text"][split_name]
        combined = results["combined"][split_name]

        comparison[split_name] = _delta(
            baseline,
            combined,
            _changed_task_ids(baseline, combined),
            truths[split_name],
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": DEFAULT_SEED,
        "dataset": {
            "path": str(DATA_FILE),
            "n_rows": int(len(df)),
            "label_distribution": {
                str(key): int(value)
                for key, value in
                df[TARGET_COLUMN].value_counts().items()
            },
        },
        "independent_dataset": {
            "path": str(INDEPENDENT_FILE),
            "n_cases": int(len(independent)),
        },
        "feature_configuration": feature_configuration(),
        "structured_feature_diagnostics": structured_feature_diagnostics(df),
        "structured_feature_coefficients": structured_coefficients(),
        "models": results,
        "comparison": comparison,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    COMPARISON_FILE.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    for name in ("text", "combined"):
        _write_split_reports(name, "group_cv", results[name][
            "group_cross_validation"
        ])
        _write_split_reports(name, "independent", results[name][
            "independent_evaluation_set"
        ])

    rows = []

    for name in ("text", "combined"):
        for row in results[name]["independent_evaluation_set"][
            "by_case_kind"
        ]:
            rows.append({"model": name, **row})

    pd.DataFrame(rows).to_csv(BY_CASE_KIND_FILE, index=False)

    pd.DataFrame(report["structured_feature_diagnostics"]).to_json(
        STRUCTURED_FILE, orient="records", indent=2
    )

    return report


def _print_report(report: dict) -> None:
    """Print a readable summary of the comparison."""
    print("=" * 70)
    print("MODEL COMPARISON: text baseline vs text + structured")
    print("=" * 70)

    print(f"\nSeed: {report['seed']}")
    print(f"Dataset: {report['dataset']['path']} "
          f"({report['dataset']['n_rows']} rows)")
    for name in ("text", "combined"):
        print(f"  {name:9} {report['models'][name]['description']}")

    print("\n--- Structured feature diagnostics (training data) ---")
    for row in report["structured_feature_diagnostics"]:
        print(
            f"  {row['feature']:28} fires {row['n_true']:3d}/{row['n_true'] + row['n_false']:3d} "
            f"| labels when true: {row['labels_when_true']}"
        )

    for split_name, title in (
        ("group_cross_validation", "Grouped cross-validation (reported)"),
        ("stratified_random_split", "Stratified random split (leaky)"),
        ("independent_evaluation_set", "Independent evaluation set"),
    ):
        print(f"\n--- {title} ---")
        print(f"{'model':10} {'accuracy':>9} {'macro F1':>9}")

        for name in ("text", "combined"):
            metrics = report["models"][name][split_name]["metrics"]
            print(
                f"{name:10} {metrics['accuracy']:9.4f} "
                f"{metrics['macro_avg']['f1_score']:9.4f}"
            )

        summary = report["comparison"][split_name]
        print(
            f"  delta accuracy {summary['delta_accuracy']:+.4f} | "
            f"delta macro F1 {summary['delta_macro_f1']:+.4f} | "
            f"predictions changed on "
            f"{summary['n_rows_where_prediction_changed']} rows"
        )

        uncertainty = summary["uncertainty"]
        for name in ("text", "combined"):
            interval = uncertainty[name]
            print(
                f"  {name:9} accuracy {interval['accuracy']:.4f} "
                f"95% CI [{interval['ci_low']:.3f}, "
                f"{interval['ci_high']:.3f}] (n={interval['n_cases']})"
            )

        mcnemar = uncertainty["mcnemar"]
        print(
            f"  McNemar exact: text-only correct {mcnemar['only_first_correct']}"
            f", combined correct {mcnemar['only_second_correct']}"
            f" -> p = {mcnemar['exact_p_value']:.3f}"
        )

        for name in ("text", "combined"):
            metrics = report["models"][name][split_name]["metrics"]
            print(f"\n  {name} per-class:")
            print(f"  {'class':20} {'prec':>7} {'rec':>7} {'f1':>7} {'supp':>6}")
            for label, scores in metrics["per_class"].items():
                print(
                    f"  {label:20} {scores['precision']:7.3f} "
                    f"{scores['recall']:7.3f} {scores['f1_score']:7.3f} "
                    f"{scores['support']:6d}"
                )

    print("\n--- Independent set by stress kind ---")
    for name in ("text", "combined"):
        rows = report["models"][name]["independent_evaluation_set"][
            "by_case_kind"
        ]
        print(f"  {name}:")
        for row in rows:
            print(
                f"    {row['case_kind']:22} {row['correct']:3d}/"
                f"{row['n_cases']:3d}  accuracy {row['accuracy']:.3f}"
            )

    print("\n--- Learned structured coefficients (combined model) ---")
    coefficients: Dict[str, Dict[str, float]] = {}

    for row in report["structured_feature_coefficients"]:
        coefficients.setdefault(row["class"], {})[row["feature"]] = row[
            "coefficient"
        ]

    header = "  " + " " * 28 + "".join(
        f"{label[:10]:>21}" for label in coefficients
    )
    print(header)

    for feature in STRUCTURED_FEATURE_NAMES:
        cells = "".join(
            f"{coefficients[label][feature]:>21.4f}"
            for label in coefficients
        )
        print(f"  {feature:28}{cells}")

    print("\nReports written:")
    for path in (
        COMPARISON_FILE,
        STRUCTURED_FILE,
        BY_CASE_KIND_FILE,
        *(confusion_file(name, split) for name in ("text", "combined")
          for split in ("group_cv", "independent")),
        *(predictions_file(name, split) for name in ("text", "combined")
          for split in ("group_cv", "independent")),
    ):
        if path.exists():
            print(f"  {path}")


def main() -> None:
    _print_report(compare())


if __name__ == "__main__":
    main()
