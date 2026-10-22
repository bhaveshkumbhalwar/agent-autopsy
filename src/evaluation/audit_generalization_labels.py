"""Audit the labels of the six generalisation cases.

    python -m src.evaluation.audit_generalization_labels

An earlier version of this project assigned a label to every case by
mapping its exception class onto the taxonomy: ``NameError`` became
``invalid_input`` and ``HTTPError`` became ``unknown_tool``. Both were
annotation errors. The cases are:

* ``gen-005`` -- a ``NameError`` raised by generated code. The call was
  valid; blaming the caller's argument is wrong.
* ``gen-006`` -- an HTTP 404 from ``fetch_page``. The tool exists; saying
  it is unknown is wrong.

This script applies the rules in :mod:`src.evaluation.labelling_rules` to
every case and compares the result with the declared label and with what
the old exception-type mapping would have said. It fails if any declared
label contradicts the rules, so the audit cannot silently rot.

The output is written to ``reports/generalization_label_audit.csv``.
"""

from pathlib import Path

import pandas as pd

from src.evaluation.labelling_rules import audit_cases
from src.evaluation.test_generalization import TEST_CASES


OUTPUT_FILE = Path("reports/generalization_label_audit.csv")


def main() -> None:
    rows = audit_cases(TEST_CASES)
    frame = pd.DataFrame(rows)

    disagreements = frame[~frame["rules_agree_with_declared"]]

    print("Generalisation label audit")
    print("=" * 60)
    print(frame[[
        "task_id",
        "error_type",
        "exception_type_would_say",
        "rules_say",
        "declared_label",
    ]].to_string(index=False))

    if not disagreements.empty:
        raise ValueError(
            "Declared labels contradict the labelling rules:\n"
            + disagreements.to_string(index=False)
        )

    corrected = frame[~frame["exception_type_agrees_with_rules"]]

    print(
        f"\nEvery declared label agrees with the rules "
        f"({len(frame)} cases)."
    )
    print(
        f"The old exception-type mapping would have mislabelled "
        f"{len(corrected)} case(s):"
    )

    for _, row in corrected.iterrows():
        print(
            f"  {row['task_id']}: {row['error_type']} -> "
            f"{row['exception_type_would_say']!r}, but the rules say "
            f"{row['rules_say'] or 'out of taxonomy'!r}. {row['note']}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_FILE, index=False)

    print(f"\nAudit saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
