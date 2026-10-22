"""Tests for the diagnosis API.

The model is stubbed here on purpose: these tests are about HTTP
behaviour -- validation, status codes, caching and error handling -- not
about prediction quality, which the evaluation reports cover.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import create_app
from src.tracing.labels import SUPPORTED_LABELS


class StubClassifier:
    """The bit of a classifier the inference helper reads."""

    classes_ = np.array(sorted(SUPPORTED_LABELS))


class StubModel:
    """A pipeline stand-in with a predictable answer."""

    def __init__(self, label: str = "unknown_tool", fail: bool = False):
        self.label = label
        self.fail = fail
        self.named_steps = {"classifier": StubClassifier()}

    def predict(self, traces):
        if self.fail:
            raise RuntimeError(
                "secret-token leaked from the trace body: Unknown tool: x"
            )
        return [self.label for _ in traces]

    def predict_proba(self, traces):
        if self.fail:
            raise RuntimeError("secret-token")

        classes = list(self.named_steps["classifier"].classes_)
        scores = [0.1] * len(classes)
        scores[classes.index(self.label)] = 0.7

        return [scores for _ in traces]


def _client(monkeypatch, **kwargs):
    """Build a test client whose model is stubbed."""
    model = kwargs.pop("model", StubModel())
    calls = []

    def loader(*args, **inner):
        calls.append(args)
        if isinstance(model, Exception):
            raise model
        return model

    monkeypatch.setattr(app_module, "load_model", loader)
    app = create_app(**kwargs)

    return TestClient(app), calls


VALID_TRACE = {
    "task_id": "demo-1",
    "agent_type": "calculator",
    "tool_name": "divide",
    "status": "failure",
    "error_type": "ValueError",
    "error": "Unknown tool: divide",
}


def test_health_reports_the_loaded_model(monkeypatch):
    client, _ = _client(monkeypatch)

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "text",
        "model_loaded": True,
        "detail": None,
    }


def test_health_still_answers_when_the_model_is_missing(monkeypatch):
    client, _ = _client(
        monkeypatch,
        model=FileNotFoundError(
            "Model not found. Train it with train_classifier."
        ),
    )

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert "train_classifier" in body["detail"]


def test_diagnose_returns_the_documented_fields(monkeypatch):
    client, _ = _client(monkeypatch)

    with client:
        response = client.post("/diagnose", json=VALID_TRACE)

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "task_id",
        "predicted_label",
        "class_scores",
        "model",
    }
    assert body["task_id"] == "demo-1"
    assert body["predicted_label"] in SUPPORTED_LABELS
    assert body["model"] == "text"
    assert set(body["class_scores"]) == set(SUPPORTED_LABELS)
    assert sum(body["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert max(body["class_scores"], key=body["class_scores"].get) == (
        body["predicted_label"]
    )


def test_diagnose_accepts_a_full_agent_trace(monkeypatch):
    """A trace straight from the tracing layer must not be rejected."""
    client, _ = _client(monkeypatch)

    trace = dict(VALID_TRACE)
    trace.update({
        "arguments": {"a": 1},
        "failure_label": "unknown_tool",
        "started_at": "2026-10-22T10:00:00Z",
        "finished_at": "2026-10-22T10:00:01Z",
        "duration_ms": 1000.0,
    })

    with client:
        response = client.post("/diagnose", json=trace)

    assert response.status_code == 200
    assert response.json()["task_id"] == "demo-1"


def test_a_minimal_trace_is_accepted(monkeypatch):
    client, _ = _client(monkeypatch)

    with client:
        response = client.post("/diagnose", json={"task_id": "only-id"})

    assert response.status_code == 200
    assert response.json()["task_id"] == "only-id"


def test_a_missing_task_id_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch)

    trace = dict(VALID_TRACE)
    del trace["task_id"]

    with client:
        response = client.post("/diagnose", json=trace)

    assert response.status_code == 422


def test_an_empty_task_id_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch)

    trace = dict(VALID_TRACE)
    trace["task_id"] = ""

    with client:
        response = client.post("/diagnose", json=trace)

    assert response.status_code == 422


def test_an_invalid_status_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch)

    trace = dict(VALID_TRACE)
    trace["status"] = "crashed"

    with client:
        response = client.post("/diagnose", json=trace)

    assert response.status_code == 422


def test_a_non_object_body_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch)

    with client:
        response = client.post("/diagnose", json=["not", "a", "trace"])

    assert response.status_code == 422


def test_diagnose_returns_503_when_no_model_is_loaded(monkeypatch):
    client, _ = _client(
        monkeypatch,
        model=FileNotFoundError("Train it with train_classifier."),
    )

    with client:
        response = client.post("/diagnose", json=VALID_TRACE)

    assert response.status_code == 503
    assert "train_classifier" in response.json()["detail"]


def test_inference_errors_are_500_and_do_not_leak_the_trace(monkeypatch):
    client, _ = _client(monkeypatch, model=StubModel(fail=True))

    with client:
        response = client.post("/diagnose", json=VALID_TRACE)

    assert response.status_code == 500

    body = response.text
    assert "secret-token" not in body
    assert "Unknown tool" not in body
    assert response.json()["detail"] == "The model failed to score this trace."


def test_the_model_is_loaded_once_not_per_request(monkeypatch):
    client, calls = _client(monkeypatch)

    with client:
        client.post("/diagnose", json=VALID_TRACE)
        client.post("/diagnose", json=VALID_TRACE)
        client.post("/diagnose", json=VALID_TRACE)

    assert len(calls) == 1


def test_class_scores_are_not_described_as_probabilities(monkeypatch):
    client, _ = _client(monkeypatch)

    with client:
        schema = client.get("/openapi.json").json()

    properties = schema["components"]["schemas"]["DiagnosisResponse"][
        "properties"
    ]
    description = properties["class_scores"]["description"].lower()

    assert "not calibrated probabilities" in description
    assert "confidence" in description


def test_an_unknown_model_name_is_rejected_at_construction(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", lambda *a, **k: StubModel())

    with pytest.raises(ValueError, match="Unknown model"):
        create_app(model_name="telepathic")


def test_the_combined_model_can_be_served(monkeypatch):
    client, _ = _client(monkeypatch, model_name="combined")

    with client:
        health = client.get("/health").json()
        response = client.post("/diagnose", json=VALID_TRACE).json()

    assert health["model"] == "combined"
    assert response["model"] == "combined"
