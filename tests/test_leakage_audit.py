"""Tests for the leakage audit.

The audit is the evidence behind the split method, so its measurements
are checked against frames where the answer is known by construction.
"""

import pandas as pd
import pytest

from src.evaluation.leakage import (
    audit_dataset,
    probe_ordinal_neighbours,
    probe_single_feature,
    signature_counts,
    split_overlap,
    template_family,
)


def _make_frame() -> pd.DataFrame:
    """Two labels, two templates each, four rows each.

    The rows are grouped by template, so the labels appear in blocks --
    the layout the ordinal probe has to detect.
    """
    rows = []
    number = 0

    templates = [
        ("add", "success", "", "", "none"),
        ("multiply", "success", "", "", "none"),
        ("divide", "failure", "ValueError", "Unknown tool: divide",
         "unknown_tool"),
        ("subtract", "failure", "ValueError", "Unknown tool: subtract",
         "unknown_tool"),
    ]

    for tool, status, error_type, error, label in templates:
        for _ in range(4):
            number += 1
            rows.append({
                "task_id": f"t-{number:03d}",
                "agent_type": "calculator",
                "tool_name": tool,
                "status": status,
                "error_type": error_type,
                "error": error,
                "duration_ms": 1.5,
                "failure_label": label,
            })

    return pd.DataFrame(rows)


def _make_interleaved_frame() -> pd.DataFrame:
    """The same templates, but interleaved so any slice shares templates.

    The rows are reordered round-robin over the four templates, so the
    first half and the second half of the frame contain every template.
    """
    df = _make_frame().reset_index(drop=True)

    templates = list(dict.fromkeys(df["tool_name"]))
    blocks = {
        template: df.index[df["tool_name"] == template].tolist()
        for template in templates
    }

    order = []
    for position in range(4):
        for template in templates:
            order.append(blocks[template][position])

    interleaved = df.loc[order].reset_index(drop=True)
    interleaved["task_id"] = [
        f"i-{number:03d}" for number in range(1, len(interleaved) + 1)
    ]

    return interleaved


def test_signature_counts_finds_the_duplicate_rows():
    df = _make_frame()

    counts = signature_counts(df)

    assert len(counts) == 4
    assert sorted(counts.values()) == [4, 4, 4, 4]
    assert sum(counts.values()) == len(df)


def test_template_family_collapses_the_tool_names():
    df = _make_frame()

    divide = df[df["tool_name"] == "divide"].iloc[0]
    subtract = df[df["tool_name"] == "subtract"].iloc[0]

    assert template_family(divide) == template_family(subtract)
    assert template_family(divide) == (
        "calculator|failure|ValueError|Unknown tool: <tool>"
    )


def test_template_family_separates_the_failure_modes():
    df = _make_frame()

    success = df[df["status"] == "success"].iloc[0]
    unknown = df[df["tool_name"] == "divide"].iloc[0]

    assert template_family(success) != template_family(unknown)


def test_split_overlap_reports_the_copied_test_rows():
    df = _make_interleaved_frame()

    # Every template appears in both halves, so every test row is a copy
    # of a training row.
    overlap = split_overlap(df.iloc[:8], df.iloc[8:])

    assert overlap["test_rows"] == 8
    assert overlap["test_rows_copied_from_train"] == 8
    assert overlap["test_rows_copied_fraction"] == 1.0


def test_split_overlap_reports_no_copies_for_disjoint_templates():
    df = _make_frame()

    train = df[df["failure_label"] == "none"]
    test = df[df["failure_label"] == "unknown_tool"]

    overlap = split_overlap(train, test)

    assert overlap["test_rows_copied_from_train"] == 0
    assert overlap["test_rows_copied_fraction"] == 0.0


def test_probe_single_feature_rejects_an_unknown_column():
    df = _make_frame()

    with pytest.raises(ValueError):
        probe_single_feature(df, "not_a_column")


def test_probe_ordinal_neighbours_detects_block_structure():
    """Labels issued in blocks are recoverable from a neighbour's label."""
    df = _make_frame()

    probe = probe_ordinal_neighbours(df, "task_id")

    # The labels change exactly once, where the block changes, so every
    # neighbouring pair but the one across the boundary agrees.
    assert probe["label_transitions"] == 1
    assert probe["neighbour_accuracy"] == pytest.approx(14 / 15)


def test_probe_ordinal_neighbours_rejects_an_unknown_column():
    df = _make_frame()

    with pytest.raises(ValueError):
        probe_ordinal_neighbours(df, "not_a_column")


def test_the_audit_summarises_the_dataset():
    df = _make_frame()

    audit = audit_dataset(df, seed=42)

    assert audit["n_rows"] == 16
    assert audit["distinct_model_inputs"] == 4
    assert audit["duplicate_rows"] == 12
    assert audit["largest_group_size"] == 4
    assert audit["distinct_template_families"] == 2
    assert audit["ambiguous_model_inputs"] == 0
    assert audit["label_distribution"] == {"none": 8, "unknown_tool": 8}


def test_the_audit_reports_how_much_each_split_leaks():
    df = _make_frame()

    audit = audit_dataset(df, seed=42)

    naive = audit["split_overlap"]["stratified_random"]
    grouped = audit["split_overlap"]["stratified_group_kfold"]

    # The random split copies templates; the grouped one does not.
    assert naive["test_rows_copied_from_train"] > 0
    assert grouped["test_rows_copied_from_train"] == 0


def test_the_audit_names_the_columns_that_are_not_features():
    df = _make_frame()

    audit = audit_dataset(df, seed=42)

    assert "task_id" in audit["identifier_columns"]
    assert "duration_ms" in audit["identifier_columns"]
    assert "task_id" not in audit["text_columns"]
    assert "duration_ms" not in audit["text_columns"]


def test_the_audit_probes_the_identifier_columns():
    df = _make_frame()

    audit = audit_dataset(df, seed=42)

    assert "task_id" in audit["identifier_shortcuts"]
    assert "duration_ms" in audit["identifier_shortcuts"]

    neighbour = audit["identifier_shortcuts"]["task_id"][
        "ordinal_neighbour_probe"
    ]
    assert neighbour["neighbour_accuracy"] == pytest.approx(14 / 15)
