"""Leakage-aware evaluation of the diagnosis classifier.

Run it with::

    python -m src.evaluation.evaluate_model

It produces three numbers, and the difference between them is the point of
this script:

``stratified_random_split``
    The split the project used before the leakage audit. Reported so the
    size of the illusion is on record.

``group_cross_validation``
    The reported result. The data is split into grouped folds, so a row is
    never tested by a model that saw its template, and every row is tested
    exactly once. The scores are computed over the pooled out-of-fold
    predictions, which keeps the per-class support balanced.

``independent_evaluation_set``
    The honest number. The model is evaluated on a separately authored
    dataset it has never seen, with unseen tool names and error messages
    phrased differently from the training templates.

Nothing in this script tunes anything. The independent evaluation set is
never used to fit a model, choose a hyperparameter or select a split; it
is a final test set and is treated as one.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from src.diagnosis.features import build_text
from src.diagnosis.modeling import build_text_pipeline, feature_configuration
from src.evaluation.leakage import audit_dataset
from src.evaluation.metrics import (
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
    stratified_split,
)


DATA_FILE = Path("data/processed/ml_dataset.csv")
INDEPENDENT_FILE = Path("data/evaluation/in_domain_eval.csv")

REPORTS_DIR = Path("reports")

METRICS_FILE = REPORTS_DIR / "evaluation_metrics.json"
AUDIT_FILE = REPORTS_DIR / "leakage_audit.json"
CONFUSION_FILE = REPORTS_DIR / "evaluation_confusion_matrix.csv"
CONFUSION_INDEPENDENT_FILE = (
    REPORTS_DIR / "evaluation_confusion_matrix_independent.csv"
)
PREDICTIONS_FILE = REPORTS_DIR / "evaluation_predictions.csv"
PREDICTIONS_INDEPENDENT_FILE = (
    REPORTS_DIR / "evaluation_predictions_independent.csv"
)
CASE_KIND_FILE = REPORTS_DIR / "evaluation_by_case_kind.csv"


def _independent_result(df: pd.DataFrame, independent: pd.DataFrame) -> dict:
    """Score the independent evaluation set.

    The model is fitted on the whole training dataset. The independent set
    is disjoint from it -- checked here -- so fitting on everything is
    legitimate and never involves the final test set.
    """
    overlap = set(independent["task_id"]) & set(df["task_id"])
    if overlap:
        raise ValueError(
            "The independent evaluation set overlaps the training data: "
            f"{sorted(overlap)}"
        )

    training_inputs = set(build_text(row) for _, row in df.iterrows())
    independent_inputs = [build_text(row) for _, row in independent.iterrows()]
    shared = sum(
        1 for text in independent_inputs if text in training_inputs
    )

    result = evaluate_dataset(
        build_text_pipeline,
        df,
        independent,
        extra_columns=["case_kind"],
    )

    predictions = [row["predicted_label"] for row in result["predictions"]]

    return {
        "model": (
            "text-only pipeline fitted on all "
            f"{len(df)} rows of {DATA_FILE}; the independent set is "
            "disjoint from it and was never used for fitting"
        ),
        "dataset": str(INDEPENDENT_FILE),
        "n_cases": int(len(independent)),
        "provenance": (
            "hand-authored simulation; not captured agent behaviour"
        ),
        "model_inputs_shared_with_training": int(shared),
        "label_distribution": {
            str(key): int(value)
            for key, value in
            independent[TARGET_COLUMN].value_counts().items()
        },
        "case_kind_distribution": {
            str(key): int(value)
            for key, value in
            independent["case_kind"].value_counts().items()
        },
        "metrics": result["metrics"],
        "by_case_kind": by_case_kind(independent, predictions),
        "predictions": result["predictions"],
    }


def evaluate() -> dict:
    """Run the whole evaluation and write the reports.

    Returns:
        The results dictionary, which is also written to
        :data:`METRICS_FILE`.
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

    # --- the leakage audit --------------------------------------------------
    audit = audit_dataset(df, seed=DEFAULT_SEED)

    # --- the two in-dataset evaluations ------------------------------------
    grouped = cross_validate(build_text_pipeline, df, seed=DEFAULT_SEED)
    grouped["model"] = (
        "one text-only pipeline per fold, fitted on that fold's training "
        "rows only"
    )

    naive_split = stratified_split(
        df, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE
    )
    naive = evaluate_holdout(build_text_pipeline, df, naive_split)
    naive["model"] = (
        "text-only pipeline fitted on the split's training rows only"
    )

    # --- the independent evaluation set ------------------------------------
    independent_result = _independent_result(df, independent)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": DEFAULT_SEED,
        "dataset": {
            "path": str(DATA_FILE),
            "n_rows": int(len(df)),
            "label_distribution": {
                str(key): int(value)
                for key, value in df[TARGET_COLUMN].value_counts().items()
            },
        },
        "feature_configuration": feature_configuration(),
        "leakage_audit": audit,
        "results": {
            "stratified_random_split": naive,
            "group_cross_validation": grouped,
            "independent_evaluation_set": independent_result,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    METRICS_FILE.write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    AUDIT_FILE.write_text(
        json.dumps(audit, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    write_confusion_matrix(grouped["metrics"], CONFUSION_FILE)
    write_predictions(grouped["predictions"], PREDICTIONS_FILE)

    write_confusion_matrix(
        independent_result["metrics"], CONFUSION_INDEPENDENT_FILE
    )
    write_predictions(
        independent_result["predictions"], PREDICTIONS_INDEPENDENT_FILE
    )
    pd.DataFrame(independent_result["by_case_kind"]).to_csv(
        CASE_KIND_FILE, index=False
    )

    return results


def _print_summary(results: dict) -> None:
    """Print a readable summary of the evaluation."""
    audit = results["leakage_audit"]

    print("=" * 70)
    print("LEAKAGE-AWARE EVALUATION")
    print("=" * 70)

    print(f"\nSeed: {results['seed']}")
    print(f"Dataset: {results['dataset']['path']}")
    print(f"Rows: {results['dataset']['n_rows']}")
    print(f"Label distribution: {results['dataset']['label_distribution']}")

    print("\n--- Leakage audit ---")
    print(
        f"Distinct model inputs: {audit['distinct_model_inputs']} "
        f"(rows: {audit['n_rows']}, duplicate rows: "
        f"{audit['duplicate_rows']})"
    )
    print(f"Distinct template families: {audit['distinct_template_families']}")
    print(f"Ambiguous model inputs: {audit['ambiguous_model_inputs']}")

    for name, overlap in audit["split_overlap"].items():
        print(
            f"  {name:24} test rows copied from train: "
            f"{overlap['test_rows_copied_from_train']}/"
            f"{overlap['test_rows']} "
            f"({overlap['test_rows_copied_fraction']:.1%})"
        )

    print("\n--- Identifier shortcut probes ---")
    for column, probe in audit["identifier_shortcuts"].items():
        neighbour = probe.get("ordinal_neighbour_probe", {})
        print(
            f"  {column:14} model-alone={probe['held_out_accuracy']:.3f} "
            f"neighbour={neighbour.get('neighbour_accuracy', 0.0):.3f}"
        )

    for key, title in (
        ("stratified_random_split", "Stratified random split (old method)"),
        ("group_cross_validation", "Grouped cross-validation (reported)"),
        ("independent_evaluation_set", "Independent evaluation set"),
    ):
        result = results["results"][key]
        if result is None:
            continue

        print_result(title, result)

    independent = results["results"]["independent_evaluation_set"]
    if independent and independent["by_case_kind"]:
        print("\n--- Independent set, by stress kind ---")
        for row in independent["by_case_kind"]:
            print(
                f"  {row['case_kind']:22} {row['correct']:3d}/{row['n_cases']:3d}"
                f"  accuracy {row['accuracy']:.3f}"
            )

    print("\nReports written:")
    for path in (
        METRICS_FILE,
        AUDIT_FILE,
        CONFUSION_FILE,
        PREDICTIONS_FILE,
        CONFUSION_INDEPENDENT_FILE,
        PREDICTIONS_INDEPENDENT_FILE,
        CASE_KIND_FILE,
    ):
        if path.exists():
            print(f"  {path}")


def print_result(title: str, result: dict) -> None:
    """Print one result block (shared with the model comparison)."""
    metrics = result["metrics"]

    print(f"\n--- {title} ---")
    if "n_train" in result:
        print(
            f"Method: {result['split_method']} | "
            f"train {result['n_train']} / test {result['n_test']}"
        )
    elif "n_splits" in result:
        print(
            f"Method: {result['split_method']} | "
            f"{result['n_splits']} folds | scored {result['n_scored']} rows"
        )
        for fold in result["folds"]:
            print(
                f"    fold {fold['fold']}: train {fold['n_train']:3d} "
                f"test {fold['n_test']:3d} "
                f"accuracy {fold['test_accuracy']:.3f} "
                f"{fold['test_label_distribution']}"
            )
    else:
        print(f"Dataset: {result['dataset']} ({result['n_cases']} cases)")

    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"macro F1: {metrics['macro_avg']['f1_score']:.4f} | "
        f"weighted F1: {metrics['weighted_avg']['f1_score']:.4f}"
    )
    print(f"{'class':20} {'prec':>7} {'rec':>7} {'f1':>7} {'supp':>6}")

    for label, scores in metrics["per_class"].items():
        print(
            f"{label:20} {scores['precision']:7.3f} "
            f"{scores['recall']:7.3f} {scores['f1_score']:7.3f} "
            f"{scores['support']:6d}"
        )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    labels = metrics["confusion_matrix"]["labels"]
    header = " " * 20 + "".join(f"{label[:9]:>11}" for label in labels)
    print(header)
    for label, row in zip(labels, metrics["confusion_matrix"]["matrix"]):
        cells = "".join(f"{value:>11d}" for value in row)
        print(f"{label:20}{cells}")


def main() -> None:
    results = evaluate()
    _print_summary(results)


if __name__ == "__main__":
    main()
