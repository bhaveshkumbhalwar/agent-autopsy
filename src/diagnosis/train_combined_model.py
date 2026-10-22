"""Train the combined text-plus-structured classifier.

    python -m src.diagnosis.train_combined_model

Writes ``models/combined_failure_classifier.joblib``.

The model is the *same* pipeline as the text-only baseline with one flag
changed: :class:`src.diagnosis.modeling.TraceFeatureExtractor` appends the
structured feature block. Both models read the same input, use the same
text construction and are compared on the same splits by
:mod:`src.diagnosis.compare_models`.

The structured features are heuristic. The five of them are described in
:data:`src.diagnosis.features.STRUCTURED_FEATURE_NAMES`, and their limits
are measured and written to ``reports/model_comparison.json`` by the
comparison script.
"""

from src.diagnosis.modeling import build_combined_pipeline
from src.diagnosis.training import train_and_save


def main() -> None:
    train_and_save("combined", build_combined_pipeline)


if __name__ == "__main__":
    main()
