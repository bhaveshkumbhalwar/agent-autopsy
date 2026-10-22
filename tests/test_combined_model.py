"""Tests for the combined model and for training/serving consistency.

The combined model differs from the text-only baseline in one flag, and
that is the point: everything else -- the input, the text construction,
the vectorizer, the classifier -- is shared. These tests pin that down,
and cover the trace shapes inference actually meets: missing fields,
unknown tools, missing arguments, invalid inputs, successful executions,
and error messages that appear nowhere in the training data.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.diagnosis.features import (
    STRUCTURED_FEATURE_NAMES,
    build_structured_features,
    build_text,
)
from src.diagnosis.modeling import (
    TraceFeatureExtractor,
    build_combined_pipeline,
    build_pipeline,
    build_text_pipeline,
)
from src.diagnosis.predict_failure import diagnose_trace, load_model
from src.tracing.labels import SUPPORTED_LABELS


def _training_frame() -> pd.DataFrame:
    """A small frame with the shape of the generated dataset.

    Eight template groups, two per label, four rows each. The rows of a
    group differ only in values the model never sees.
    """
    templates = [
        ("add", "success", "", "", "none", {"a": 1, "b": 2}),
        ("multiply", "success", "", "", "none", {"a": 1, "b": 2}),
        ("divide", "failure", "ValueError", "Unknown tool: divide",
         "unknown_tool", {"a": 1, "b": 2}),
        ("subtract", "failure", "ValueError", "Unknown tool: subtract",
         "unknown_tool", {"a": 1, "b": 2}),
        ("add", "failure", "KeyError", "'b'", "missing_argument",
         {"a": 1}),
        ("multiply", "failure", "KeyError", "'a'", "missing_argument",
         {"b": 2}),
        ("add", "failure", "ValueError",
         "Arguments a and b must be numbers", "invalid_input",
         {"a": "x", "b": 2}),
        ("multiply", "failure", "ValueError",
         "Arguments a and b must be numbers", "invalid_input",
         {"a": 1, "b": "y"}),
    ]

    rows = []
    number = 0

    for tool, status, error_type, error, label, arguments in templates:
        for _ in range(4):
            number += 1
            rows.append({
                "task_id": f"t-{number:03d}",
                "agent_type": "calculator",
                "tool_name": tool,
                "arguments": json.dumps(arguments, sort_keys=True),
                "status": status,
                "error_type": error_type,
                "error": error,
                "failure_label": label,
            })

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_models():
    """Fit both pipelines once on the small frame."""
    df = _training_frame()
    traces = df.to_dict("records")
    labels = df["failure_label"].to_numpy()

    models = {}

    for name, factory in (
        ("text", build_text_pipeline),
        ("combined", build_combined_pipeline),
    ):
        model = factory()
        model.fit(traces, labels)
        models[name] = model

    return models


def _scenarios():
    """One trace per kind of input inference is expected to meet."""
    return {
        "success": {
            "task_id": "s-1",
            "agent_type": "calculator",
            "tool_name": "add",
            "arguments": {"a": 1, "b": 2},
            "status": "success",
            "error_type": None,
            "error": None,
        },
        "unknown_tool": {
            "task_id": "s-2",
            "agent_type": "calculator",
            "tool_name": "teleport",
            "arguments": {"a": 1},
            "status": "failure",
            "error_type": "ValueError",
            "error": "Unknown tool: teleport",
        },
        "missing_argument": {
            "task_id": "s-3",
            "agent_type": "calculator",
            "tool_name": "add",
            "arguments": {"a": 1},
            "status": "failure",
            "error_type": "KeyError",
            "error": "'b'",
        },
        "invalid_input": {
            "task_id": "s-4",
            "agent_type": "calculator",
            "tool_name": "multiply",
            "arguments": {"a": "x", "b": 2},
            "status": "failure",
            "error_type": "ValueError",
            "error": "Arguments a and b must be numbers",
        },
        "unseen_error_message": {
            "task_id": "s-5",
            "agent_type": "calculator",
            "tool_name": "add",
            "arguments": {"a": 1},
            "status": "failure",
            "error_type": "ValueError",
            "error": "the operand was rejected by the parser at column 3",
        },
        "missing_fields": {
            "task_id": "s-6",
            "agent_type": "calculator",
        },
        "empty_trace": {},
    }


def test_the_two_pipelines_share_the_text_block(fitted_models):
    """The comparison is controlled: only the structured block differs."""
    text_model = fitted_models["text"]
    combined_model = fitted_models["combined"]

    text_features = text_model.named_steps["features"]
    combined_features = combined_model.named_steps["features"]

    traces = [scenario for scenario in _scenarios().values()]

    text_matrix = text_features.transform(traces)
    combined_matrix = combined_features.transform(traces)

    text_width = text_matrix.shape[1]

    assert text_matrix.shape[1] == combined_matrix.shape[1] - len(
        STRUCTURED_FEATURE_NAMES
    )

    # The leading columns are the same TF-IDF block, value for value.
    assert np.allclose(
        text_matrix.toarray(),
        combined_matrix.toarray()[:, :text_width],
    )


def test_only_the_structured_flag_differs():
    text_model = build_text_pipeline()
    combined_model = build_combined_pipeline()

    text_extractor = text_model.named_steps["features"]
    combined_extractor = combined_model.named_steps["features"]

    assert text_extractor.use_structured_features is False
    assert combined_extractor.use_structured_features is True

    assert (
        text_model.named_steps["classifier"].get_params()
        == combined_model.named_steps["classifier"].get_params()
    )


def test_feature_names_end_with_the_structured_features(fitted_models):
    extractor = fitted_models["combined"].named_steps["features"]

    names = list(extractor.get_feature_names_out())

    assert names[-len(STRUCTURED_FEATURE_NAMES):] == list(
        STRUCTURED_FEATURE_NAMES
    )


def test_the_structured_block_follows_the_documented_order(fitted_models):
    """Column i of the block must be STRUCTURED_FEATURE_NAMES[i]."""
    extractor = fitted_models["combined"].named_steps["features"]

    trace = _scenarios()["invalid_input"]
    matrix = extractor.transform([trace]).toarray()

    structured = matrix[0, -len(STRUCTURED_FEATURE_NAMES):]

    raw = np.array([
        float(build_structured_features(trace)[name])
        for name in STRUCTURED_FEATURE_NAMES
    ])

    scaled = extractor.scaler_.transform([raw])[0]

    assert np.allclose(structured, scaled)


def test_the_feature_matrix_does_not_depend_on_row_order(fitted_models):
    """Featurising traces one at a time matches featurising them in bulk."""
    extractor = fitted_models["combined"].named_steps["features"]

    traces = list(_scenarios().values())

    bulk = extractor.transform(traces).toarray()

    for index, trace in enumerate(traces):
        single = extractor.transform([trace]).toarray()[0]
        assert np.allclose(single, bulk[index])


@pytest.mark.parametrize("model_name", ["text", "combined"])
@pytest.mark.parametrize("scenario", list(_scenarios()))
def test_every_trace_shape_is_scored_within_the_taxonomy(
    fitted_models, model_name, scenario
):
    """Missing fields, unknown tools and unseen messages must not crash."""
    model = fitted_models[model_name]

    trace = _scenarios()[scenario]

    prediction = model.predict([trace])[0]
    scores = model.predict_proba([trace])[0]

    assert str(prediction) in SUPPORTED_LABELS
    assert scores.shape == (len(model.named_steps["classifier"].classes_),)
    assert float(scores.sum()) == pytest.approx(1.0)


def test_missing_fields_are_treated_as_empty():
    trace = {"task_id": "s-6", "agent_type": "calculator"}

    assert build_text(trace) == "calculator    "
    assert build_structured_features(trace)["has_error"] is False
    assert build_structured_features(trace)["has_arguments"] is False


def test_an_unseen_error_message_produces_no_phrase_match():
    """The phrase detectors are exact: a paraphrase fires nothing."""
    trace = _scenarios()["unseen_error_message"]

    features = build_structured_features(trace)

    assert features["has_error"] is True
    assert features["has_missing_argument_error"] is False
    assert features["has_unknown_tool_error"] is False
    assert features["has_invalid_input_error"] is False


def test_the_features_never_read_the_label(fitted_models):
    """A trace's label must not reach the feature matrix.

    This is the guard against a target-derived feature: a trace that is
    identical except for its label has to produce an identical row.
    """
    extractor = fitted_models["combined"].named_steps["features"]

    clean = _scenarios()["missing_argument"]
    labelled = dict(clean)
    labelled["failure_label"] = "LEAK_MARKER_XYZ"

    assert "LEAK_MARKER_XYZ" not in build_text(labelled)

    for name in STRUCTURED_FEATURE_NAMES:
        assert (
            build_structured_features(labelled)[name]
            == build_structured_features(clean)[name]
        )

    assert np.allclose(
        extractor.transform([clean]).toarray(),
        extractor.transform([labelled]).toarray(),
    )


def test_a_saved_model_can_score_a_new_trace(tmp_path):
    """Everything inference needs must survive the round trip to disk."""
    df = _training_frame()

    model = build_combined_pipeline()
    model.fit(df.to_dict("records"), df["failure_label"].to_numpy())

    model_file = tmp_path / "combined.joblib"
    joblib.dump(model, model_file)

    scores = model.predict_proba([_scenarios()["unknown_tool"]])[0]

    loaded = load_model(path=str(model_file))
    loaded_scores = loaded.predict_proba(
        [_scenarios()["unknown_tool"]]
    )[0]

    assert np.allclose(scores, loaded_scores)

    # The preprocessing objects travelled with the classifier.
    extractor = loaded.named_steps["features"]
    assert isinstance(extractor, TraceFeatureExtractor)
    assert len(extractor.tfidf_.vocabulary_) > 0
    assert hasattr(extractor, "scaler_")


def test_diagnose_trace_scores_a_new_trace(tmp_path, monkeypatch):
    """The public inference helper works on a trace it has never seen."""
    import src.diagnosis.predict_failure as predict_failure

    df = _training_frame()

    model = build_combined_pipeline()
    model.fit(df.to_dict("records"), df["failure_label"].to_numpy())

    model_file = tmp_path / "combined.joblib"
    joblib.dump(model, model_file)

    predict_failure.load_model.cache_clear()
    monkeypatch.setattr(
        predict_failure,
        "MODEL_FILES",
        {"text": Path("does-not-exist.joblib"), "combined": model_file},
    )

    diagnosis = predict_failure.diagnose_trace(
        _scenarios()["unknown_tool"], model="combined"
    )

    assert diagnosis["task_id"] == "s-2"
    assert diagnosis["predicted_label"] in SUPPORTED_LABELS
    assert set(diagnosis["class_scores"]) == set(SUPPORTED_LABELS)
    assert sum(diagnosis["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert max(diagnosis["class_scores"], key=diagnosis["class_scores"].get) == (
        diagnosis["predicted_label"]
    )

    predict_failure.load_model.cache_clear()


def test_a_model_without_the_project_extractor_is_rejected(tmp_path):
    """An old model file must produce an instruction, not a traceback."""
    df = _training_frame()

    legacy = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    legacy.fit(
        [build_text(row) for _, row in df.iterrows()],
        df["failure_label"].to_numpy(),
    )

    model_file = tmp_path / "legacy.joblib"
    joblib.dump(legacy, model_file)

    with pytest.raises(ValueError, match="Retrain"):
        load_model(path=str(model_file))


def test_a_missing_model_file_says_how_to_train_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_classifier"):
        load_model(path=str(tmp_path / "absent.joblib"))


def test_an_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model(model="telepathic")


def test_build_pipeline_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("telepathic")


def test_a_trace_that_is_not_a_mapping_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        diagnose_trace("calculator add success")


def test_the_combined_matrix_is_sparse_not_dense():
    """The text block stays sparse, so the model scales with the vocabulary."""
    model = build_combined_pipeline()

    df = _training_frame()
    model.fit(df.to_dict("records"), df["failure_label"].to_numpy())

    matrix = model.named_steps["features"].transform(
        [_scenarios()["success"]]
    )

    assert hasattr(matrix, "toarray")
