
from src.diagnosis.features import (
    IDENTIFIER_COLUMNS,
    STRUCTURED_FEATURE_NAMES,
    TEXT_COLUMNS,
    build_structured_features,
    build_text,
    extract_features,
    parse_arguments,
)


def test_missing_argument_features():
    trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "arguments": {"a": 10},
    }

    features = extract_features(trace)

    assert features["has_error"] is True
    assert features["has_arguments"] is True
    assert features["has_missing_argument_error"] is True


def test_success_features():
    trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "success",
        "error_type": "",
        "error": "",
        "arguments": {"a": 10, "b": 5},
    }

    features = extract_features(trace)

    assert features["has_error"] is False
    assert features["has_missing_argument_error"] is False


def test_identifiers_are_never_model_inputs():
    """Task ids and durations must stay out of the feature set.

    In the generated dataset the task id encodes the label outright, so
    using it as a feature would be a shortcut rather than a diagnosis.
    """
    assert "task_id" not in TEXT_COLUMNS
    assert "duration_ms" not in TEXT_COLUMNS
    assert "task_id" in IDENTIFIER_COLUMNS
    assert "duration_ms" in IDENTIFIER_COLUMNS


def test_build_text_joins_the_columns_in_a_fixed_order():
    trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
    }

    assert build_text(trace) == "calculator add failure KeyError 'b'"


def test_build_text_ignores_identifiers_and_missing_values():
    trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "task_id": "d7-101",
        "duration_ms": 3,
    }

    text = build_text(trace)

    assert "d7-101" not in text
    assert "3" not in text


def test_build_text_turns_a_missing_field_into_empty_text():
    trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": None,
        "error": float("nan"),
    }

    text = build_text(trace)

    assert "nan" not in text
    assert text == "calculator add failure  "


def test_parse_arguments_reads_the_json_stored_in_the_datasets():
    assert parse_arguments('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments("") == {}
    assert parse_arguments(float("nan")) == {}
    assert parse_arguments("not json") == {}
    assert parse_arguments("[1, 2]") == {}


def test_has_arguments_ignores_an_empty_argument_mapping():
    """An empty JSON object must not count as "the call had arguments"."""
    assert extract_features({"arguments": "{}"})["has_arguments"] is False
    assert extract_features({"arguments": '{"a": 1}'})["has_arguments"] is True
    assert extract_features({})["has_arguments"] is False


def test_build_structured_features_returns_the_named_features_in_order():
    trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "arguments": {"a": 10},
    }

    structured = build_structured_features(trace)

    assert list(structured.keys()) == list(STRUCTURED_FEATURE_NAMES)
    assert structured["has_missing_argument_error"] is True
    assert structured["has_unknown_tool_error"] is False
