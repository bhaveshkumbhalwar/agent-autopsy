"""Deprecated alias for :mod:`src.diagnosis.train_classifier`.

This script used to fit a second copy of the text-only classifier on a
plain random split and print the accuracy of that split. The leakage audit
(``src/evaluation/leakage.py``) showed that such a split puts a literal
copy of every test row in the training set, so the printed number measured
memorisation rather than diagnosis.

It is kept as an alias so that an old command still works, but it no
longer prints a score:

* the text-only baseline is trained by ``train_classifier``;
* the combined text-plus-structured model is trained by
  ``train_combined_model``;
* the leakage-free scores are produced by
  ``src.evaluation.evaluate_model`` and ``src.diagnosis.compare_models``.

    python -m src.diagnosis.train_enhanced_classifier
"""

from src.diagnosis.train_classifier import main as train_text_baseline


def main() -> None:
    print(
        "train_enhanced_classifier is deprecated and now trains the "
        "text-only baseline.\n"
        "  combined model:  python -m src.diagnosis.train_combined_model\n"
        "  evaluation:      python -m src.evaluation.evaluate_model\n"
        "  comparison:      python -m src.diagnosis.compare_models\n"
    )
    train_text_baseline()


if __name__ == "__main__":
    main()
