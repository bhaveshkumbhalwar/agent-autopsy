
from src.diagnosis.features import extract_features


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
