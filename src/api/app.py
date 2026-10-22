"""HTTP API for the failure diagnosis model.

Run it:

    uvicorn src.api.app:app --reload
    uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Call it:

    curl -s http://127.0.0.1:8000/health

    curl -s -X POST http://127.0.0.1:8000/diagnose \
      -H 'content-type: application/json' \
      -d '{
        "task_id": "demo-1",
        "agent_type": "calculator",
        "tool_name": "divide",
        "status": "failure",
        "error_type": "ValueError",
        "error": "Unknown tool: divide"
      }'

    {"task_id":"demo-1","predicted_label":"unknown_tool",
     "class_scores":{...},"model":"text"}

Design notes
------------
* The model is loaded **once**, during application startup, and cached on
  ``app.state``. A request never touches the disk.
* If the model file is missing, the service still starts and ``/health``
  answers, but ``/diagnose`` returns ``503`` with the command needed to
  train it. A missing artefact is an operational problem, not a crash.
* ``class_scores`` are the classifier's scores, **not calibrated
  probabilities**. The model was never calibrated, so the API does not
  call them probabilities and neither should a client.
* There is no authentication, rate limiting or TLS here. This module is a
  local service, not a deployment.

The request body accepts the fields the classifier reads. A full
``AgentTrace`` (with timestamps, arguments and a label) is also accepted:
fields the model does not use are ignored rather than rejected, so the
same payload can be posted from the tracing layer.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.diagnosis.modeling import MODEL_FILES
from src.diagnosis.predict_failure import (
    DEFAULT_MODEL,
    load_model,
    predict_with_pipeline,
)


LOGGER = logging.getLogger(__name__)

API_VERSION = "1.0.0"


class DiagnosisRequest(BaseModel):
    """The part of an execution trace the classifier needs.

    Extra fields are ignored, so a complete trace can be posted as-is.
    """

    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(
        min_length=1,
        max_length=200,
        description="Identifier of the execution being diagnosed.",
        examples=["demo-1"],
    )
    agent_type: str = Field(
        default="",
        max_length=100,
        description="Kind of agent that produced the trace.",
        examples=["calculator"],
    )
    tool_name: str = Field(
        default="",
        max_length=200,
        description="Tool the agent tried to call.",
        examples=["divide"],
    )
    status: Literal["success", "failure"] = Field(
        default="success",
        description="Whether the tool call ran to completion.",
    )
    error_type: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Exception class recorded on the trace, if any.",
        examples=["ValueError"],
    )
    error: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="Error message recorded on the trace, if any.",
        examples=["Unknown tool: divide"],
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments the agent passed. Recorded, not shown to "
        "the text-only model.",
    )

    def as_trace(self) -> Dict[str, Any]:
        """Return the mapping the classifier consumes."""
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "tool_name": self.tool_name,
            "status": self.status,
            "error_type": self.error_type or "",
            "error": self.error or "",
            "arguments": self.arguments,
        }


class DiagnosisResponse(BaseModel):
    """The diagnosis for one trace."""

    task_id: str = Field(description="Echo of the request's task id.")
    predicted_label: str = Field(
        description="The failure category the model chose.",
        examples=["unknown_tool"],
    )
    class_scores: Dict[str, float] = Field(
        description=(
            "Score for every category, summing to roughly 1. These are "
            "the classifier's scores, not calibrated probabilities: the "
            "model was never calibrated, so they must not be read as "
            "confidence that the label is correct."
        )
    )
    model: str = Field(
        description="Which trained model answered the request.",
        examples=["text"],
    )


class HealthResponse(BaseModel):
    """Liveness and readiness in one response."""

    status: str = Field(description="'ok' when the process is serving.")
    model: str = Field(description="The model this process is configured for.")
    model_loaded: bool = Field(
        description="Whether a model is available for diagnosis."
    )
    detail: Optional[str] = Field(
        default=None,
        description="Why the model is unavailable, when it is.",
    )


def create_app(*, model_name: str = DEFAULT_MODEL) -> FastAPI:
    """Build the app, loading the model once during startup.

    Args:
        model_name: Which trained model to serve: ``"text"`` (the default)
            or ``"combined"``.

    Returns:
        The configured :class:`fastapi.FastAPI` application.

    Raises:
        ValueError: If ``model_name`` is not a known model. This is a
            configuration error and is raised at construction time, not
            at request time.
    """
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of "
            f"{', '.join(sorted(MODEL_FILES))}."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once, not per request. A failure is recorded rather than
        # raised so that /health can explain what is wrong.
        app.state.model = None
        app.state.model_error = None

        try:
            app.state.model = load_model(model_name)
        except (FileNotFoundError, ValueError) as error:
            LOGGER.error("Model %r is unavailable: %s", model_name, error)
            app.state.model_error = str(error)
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error loading model %r", model_name)
            app.state.model_error = "The model could not be loaded."

        yield

        app.state.model = None

    app = FastAPI(
        title="Agent Autopsy diagnosis API",
        version=API_VERSION,
        description=(
            "Diagnose an agent execution trace with the trained failure "
            "classifier. Class scores are not calibrated probabilities."
        ),
        lifespan=lifespan,
    )

    app.state.model_name = model_name

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report whether the process can serve diagnoses."""
        model = request.app.state.model

        return HealthResponse(
            status="ok",
            model=model_name,
            model_loaded=model is not None,
            detail=request.app.state.model_error,
        )

    @app.post(
        "/diagnose",
        response_model=DiagnosisResponse,
        responses={
            500: {"description": "The model failed while scoring the trace."},
            503: {"description": "No model is loaded."},
        },
    )
    def diagnose(
        request: Request,
        payload: DiagnosisRequest,
    ) -> DiagnosisResponse:
        """Diagnose one trace.

        Returns 422 for a request that fails validation, 503 when no
        model is available, and 500 when inference itself fails.
        """
        model = request.app.state.model

        if model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    request.app.state.model_error
                    or "No model is loaded."
                ),
            )

        try:
            result = predict_with_pipeline(model, payload.as_trace())
        except Exception:
            # Log the details, return none of them: an inference error can
            # carry trace content.
            LOGGER.exception(
                "Diagnosis failed for task %r", payload.task_id
            )
            raise HTTPException(
                status_code=500,
                detail="The model failed to score this trace.",
            )

        return DiagnosisResponse(
            task_id=result["task_id"],
            predicted_label=result["predicted_label"],
            class_scores=result["class_scores"],
            model=model_name,
        )

    return app


app = create_app()
