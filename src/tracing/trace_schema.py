"""Schema for a single recorded agent tool execution."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.tracing.labels import ExecutionStatus, normalize_label


class AgentTrace(BaseModel):
    """One recorded tool execution.

    The model encodes the data-quality rules that every trace must satisfy:

    * ``status`` is one of :data:`src.tracing.labels.ALLOWED_STATUSES`.
    * A successful trace carries no ``error`` and no ``error_type``.
    * A failed trace carries a non-empty ``error`` and ``error_type`` and no
      ``result``.
    * ``failure_label``, when present, is one of
      :data:`src.tracing.labels.SUPPORTED_LABELS` and agrees with ``status``.
    * Unknown fields are rejected, so a typo in a trace file is reported
      instead of silently dropping data.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_type: str
    tool_name: str

    arguments: Dict[str, Any] = Field(default_factory=dict)

    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # The taxonomy label for this execution. ``None`` means "not determined",
    # which is different from "none" (the label for a successful execution).
    failure_label: Optional[str] = None

    started_at: datetime
    finished_at: datetime
    duration_ms: float

    @classmethod
    def create(
        cls,
        task_id: str,
        agent_type: str,
        tool_name: str,
        arguments: dict,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        result=None,
        error=None,
        error_type=None,
        failure_label=None,
    ) -> "AgentTrace":
        """Build a trace and derive ``duration_ms`` from the timestamps."""
        duration_ms = (
            finished_at - started_at
        ).total_seconds() * 1000

        return cls(
            task_id=task_id,
            agent_type=agent_type,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            result=result,
            error=error,
            error_type=error_type,
            failure_label=failure_label,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    @field_validator("failure_label", mode="before")
    @classmethod
    def _validate_failure_label(cls, value):
        """Reject a label outside the taxonomy instead of accepting it."""
        return normalize_label(value)

    @model_validator(mode="after")
    def _check_trace_contract(self) -> "AgentTrace":
        """Check that status, result and error information agree."""
        if self.status == "success":
            if self.error is not None or self.error_type is not None:
                raise ValueError(
                    "A successful trace must not carry error information "
                    f"(error={self.error!r}, error_type={self.error_type!r})."
                )
            if self.failure_label not in (None, "none"):
                raise ValueError(
                    "A successful trace cannot carry the failure label "
                    f"{self.failure_label!r}; expected 'none'."
                )
        else:
            if not self.error:
                raise ValueError(
                    "A failed trace must carry a non-empty error message."
                )
            if not self.error_type:
                raise ValueError(
                    "A failed trace must carry a non-empty error_type."
                )
            if self.result is not None:
                raise ValueError(
                    "A failed trace must not carry a result, got "
                    f"{self.result!r}."
                )
            if self.failure_label == "none":
                raise ValueError(
                    "A failed trace cannot carry the failure label 'none'."
                )
        return self
