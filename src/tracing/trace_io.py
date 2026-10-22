"""Reading and writing the JSONL trace format.

The on-disk format is one JSON object per line, each object matching
:mod:`src.tracing.trace_schema`.

Legacy note
-----------
Traces written before the trace contract existed have no ``duration_ms``
field and may omit ``error_type`` on successful executions.
:func:`load_traces` accepts those files: ``duration_ms`` is derived from
``started_at``/``finished_at``, and ``error_type``/``failure_label`` stay
``None`` because they cannot be recovered. The number of migrated records is
reported so the migration is never silent.

To regenerate the trace files with the current format, delete the existing
files and re-run the agent::

    rm data/traces/*.jsonl
    python -m src.evaluation.build_larger_dataset
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

from pydantic import TypeAdapter, ValidationError

from src.tracing.trace_schema import AgentTrace


class TraceFormatError(ValueError):
    """Raised when a trace file cannot be read against the trace contract."""


class TraceLoadResult(NamedTuple):
    """The outcome of reading a trace file."""

    traces: List[AgentTrace]
    migrated: int
    source: Path


_DATETIME_ADAPTER = TypeAdapter(datetime)


def _migrate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Derive ``duration_ms`` for records written before it was stored."""
    if record.get("duration_ms") is not None:
        return record

    started = record.get("started_at")
    finished = record.get("finished_at")
    if started is None or finished is None:
        return record

    started_at = _DATETIME_ADAPTER.validate_python(started)
    finished_at = _DATETIME_ADAPTER.validate_python(finished)

    migrated = dict(record)
    migrated["duration_ms"] = (
        finished_at - started_at
    ).total_seconds() * 1000
    return migrated


def load_traces(path: Path) -> TraceLoadResult:
    """Load and validate every trace in a JSONL file.

    Args:
        path: The JSONL file to read.

    Returns:
        A :class:`TraceLoadResult` with the validated traces and the number
        of records that needed the legacy ``duration_ms`` migration.

    Raises:
        TraceFormatError: If a line is not valid JSON or does not satisfy the
            trace contract.
    """
    path = Path(path)
    traces: List[AgentTrace] = []
    migrated = 0

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceFormatError(
                    f"{path}:{number} is not valid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TraceFormatError(
                    f"{path}:{number} is not a JSON object, got "
                    f"{type(record).__name__}."
                )

            updated = _migrate_record(record)
            if updated is not record:
                migrated += 1

            try:
                traces.append(AgentTrace.model_validate(updated))
            except ValidationError as exc:
                raise TraceFormatError(
                    f"{path}:{number} does not match the trace contract:\n"
                    f"{exc}"
                ) from exc

    return TraceLoadResult(traces, migrated, path)


def append_trace(trace: AgentTrace, path: Path) -> None:
    """Append one trace to a JSONL file, creating the file if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(trace.model_dump_json() + "\n")


def write_traces(traces: List[AgentTrace], path: Path) -> None:
    """Write traces to a JSONL file, replacing any previous content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(trace.model_dump_json() + "\n")
