from typing import Any, Optional
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class AgentTrace(BaseModel):
    task_id: str
    agent_type: str
    tool_name: str

    arguments: dict[str, Any] = Field(default_factory=dict)

    status: str
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

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
    ):
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
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )
