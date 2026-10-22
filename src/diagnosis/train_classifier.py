"""Train the text-only baseline classifier.

    python -m src.diagnosis.train_classifier

Writes ``models/failure_classifier.joblib``. The combined model is a
separate script (:mod:`src.diagnosis.train_combined_model`), and the
comparison between the two is
:mod:`src.diagnosis.compare_models`.
"""

from src.diagnosis.modeling import build_text_pipeline
from src.diagnosis.training import train_and_save


def main() -> None:
    train_and_save("text", build_text_pipeline)


if __name__ == "__main__":
    main()
