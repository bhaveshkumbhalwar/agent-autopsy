
def extract_features(trace: dict) -> dict:
    """Extract structured diagnostic features from an execution trace."""

    tool_name = str(trace.get("tool_name") or "")
    agent_type = str(trace.get("agent_type") or "")
    status = str(trace.get("status") or "")
    error_type = str(trace.get("error_type") or "")
    error = str(trace.get("error") or "")
    arguments = trace.get("arguments") or {}

    error_lower = error.lower()

    return {
        "agent_type": agent_type,
        "tool_name": tool_name,
        "status": status,
        "error_type": error_type,
        "error": error,
        "has_error": bool(error),
        "has_arguments": bool(arguments),
        "has_missing_argument_error": (
            error_type == "KeyError"
        ),
        "has_unknown_tool_error": (
            "unknown tool" in error_lower
        ),
        "has_invalid_input_error": (
            "must be numbers" in error_lower
        ),
    }


if __name__ == "__main__":
    sample_trace = {
        "agent_type": "calculator",
        "tool_name": "add",
        "status": "failure",
        "error_type": "KeyError",
        "error": "'b'",
        "arguments": {"a": 10},
    }

    print(extract_features(sample_trace))
