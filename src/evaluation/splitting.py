"""Reproducible, leakage-aware dataset splitting.

The generated diagnosis dataset is synthetic: many rows are produced by the
same generation template and differ only in values the model never sees
(the argument numbers). A plain random split therefore puts *exact copies*
of a template on both sides, and the resulting score measures memorisation
rather than diagnosis.

Two split methods are provided:

``stratified_split``
    The classic random split. It is kept only so the evaluation report can
    show how flattering it is.

``group_split`` and ``group_folds``
    Splits in which no *group* appears on both sides of the same fold. A
    group is the set of rows that share the same model input (see
    :func:`case_group`), so a row in the test set is never a literal copy
    of a training row.

Why two folds and not four
--------------------------
The dataset holds two template groups for three of its four labels and ten
for the fourth. A label can only appear in as many folds as it has
groups, so a scheme with more folds than that would leave some folds
unable to measure some labels at all. The default of two folds gives every
fold every label, and folds of almost equal size.

Both are reproducible: the same seed always produces the same split.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.diagnosis.features import build_text


DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.25
DEFAULT_N_SPLITS = 2

TARGET_COLUMN = "failure_label"


@dataclass(frozen=True)
class SplitResult:
    """The outcome of a split, with everything needed to reproduce it."""

    train_index: np.ndarray
    test_index: np.ndarray
    method: str
    seed: int
    groups: Optional[np.ndarray] = None

    @property
    def n_train(self) -> int:
        return int(len(self.train_index))

    @property
    def n_test(self) -> int:
        return int(len(self.test_index))


def case_group(row: object) -> str:
    """Return the template group a row belongs to.

    Two rows with the same group produce the same model input, so they are
    copies of one another as far as the classifier is concerned. Grouping
    on that keeps a template's copies on one side of the split only.
    """
    return build_text(row)


def group_labels(df: pd.DataFrame) -> np.ndarray:
    """Return the template group of every row in ``df``."""
    return np.array([case_group(row) for _, row in df.iterrows()])


def stratified_group_assignment(
    groups: Sequence[str],
    labels: Sequence[str],
    n_splits: int,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Assign every group to a fold.

    Each label's groups are spread over the folds as evenly as possible: a
    group goes to the fold that currently holds the fewest groups of that
    label, and the fold index breaks ties. A label with at least
    ``n_splits`` groups therefore ends up in every fold.

    Within a label the larger groups are placed first, so the folds stay
    as balanced in size as the grouping allows. Groups of the same size
    are shuffled with ``seed``, which is what makes the assignment depend
    on the seed while staying reproducible.

    Args:
        groups: The template group of every row.
        labels: The label of every row.
        n_splits: The number of folds.
        seed: Random seed used to order equal-sized groups.

    Returns:
        The fold index of every row.

    Raises:
        ValueError: If a group carries more than one label, which would
            make it impossible to keep the group on one side of a split.
    """
    group_list = [str(group) for group in groups]
    label_list = [str(label) for label in labels]

    if len(group_list) != len(label_list):
        raise ValueError(
            f"Got {len(group_list)} groups and {len(label_list)} labels."
        )

    if n_splits < 2:
        raise ValueError(f"Need at least two folds, got {n_splits}.")

    group_label: Dict[str, str] = {}
    group_size: Dict[str, int] = {}

    for group, label in zip(group_list, label_list):
        if group in group_label and group_label[group] != label:
            raise ValueError(
                f"Group {group!r} carries more than one label "
                f"({group_label[group]!r} and {label!r}); it cannot be "
                "assigned to a single fold."
            )
        group_label[group] = label
        group_size[group] = group_size.get(group, 0) + 1

    rng = np.random.default_rng(seed)

    counts: Dict[Tuple[str, int], int] = {}
    assignment: Dict[str, int] = {}

    for label in sorted(set(group_label.values())):
        label_groups = [
            group for group in group_label if group_label[group] == label
        ]

        # Largest groups first; equal-sized groups are shuffled so the
        # seed changes the assignment without breaking the balance.
        by_size: Dict[int, List[str]] = {}
        for group in label_groups:
            by_size.setdefault(group_size[group], []).append(group)

        ordered: List[str] = []
        for size in sorted(by_size, reverse=True):
            same_size = sorted(by_size[size])
            rng.shuffle(same_size)
            ordered.extend(same_size)

        for group in ordered:
            fold = min(
                range(n_splits),
                key=lambda fold: (counts.get((label, fold), 0), fold),
            )
            counts[(label, fold)] = counts.get((label, fold), 0) + 1
            assignment[group] = fold

    return np.array([assignment[group] for group in group_list])


def stratified_split(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    target_column: str = TARGET_COLUMN,
) -> SplitResult:
    """Split randomly, stratifying on the label.

    This is the method the project used before the leakage audit. It is
    kept so the report can quantify how much it overstates performance.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    labels = df[target_column].to_numpy()

    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    return SplitResult(
        train_index=np.sort(train_index),
        test_index=np.sort(test_index),
        method="stratified_random",
        seed=seed,
    )


def group_folds(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
    target_column: str = TARGET_COLUMN,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return reproducible grouped folds.

    Every row is used for testing exactly once across the folds, and no
    template group ever appears in both the training and the testing part
    of the same fold. This is the scheme the reported numbers use.

    Args:
        df: The dataset to split.
        seed: Random seed; the same seed always gives the same folds.
        n_splits: Number of folds.
        target_column: The label column.

    Returns:
        A list of ``(train_index, test_index)`` pairs.
    """
    if len(df) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from {len(df)} rows."
        )

    groups = group_labels(df)
    labels = df[target_column].to_numpy()

    assignment = stratified_group_assignment(
        groups, labels, n_splits, seed=seed
    )

    folds = []
    for fold in range(n_splits):
        test_index = np.where(assignment == fold)[0]
        train_index = np.where(assignment != fold)[0]

        folds.append((np.sort(train_index), np.sort(test_index)))

    covered = np.concatenate([test_index for _, test_index in folds])
    if len(set(covered.tolist())) != len(df):
        raise ValueError("The folds do not cover every row exactly once.")

    return folds


def group_split(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
    test_fold: int = 0,
    target_column: str = TARGET_COLUMN,
) -> SplitResult:
    """Split so that no template group appears on both sides.

    The split takes one of the grouped folds as the test set. The label
    distribution is kept as even as the grouping allows.

    Args:
        df: The dataset to split.
        seed: Random seed; the same seed always gives the same split.
        n_splits: Number of folds to create.
        test_fold: Which fold becomes the test set.
        target_column: The label column.

    Returns:
        The split, with the group of every row.

    Raises:
        ValueError: If a label ends up on only one side of the split, which
            would make the test set unable to measure that label at all.
    """
    folds = group_folds(
        df, seed=seed, n_splits=n_splits, target_column=target_column
    )

    train_index, test_index = folds[test_fold]

    result = SplitResult(
        train_index=train_index,
        test_index=test_index,
        method=(
            f"stratified_group_kfold(n_splits={n_splits}, fold={test_fold})"
        ),
        seed=seed,
        groups=group_labels(df),
    )

    _assert_labels_on_both_sides(df, result, target_column)

    return result


def _assert_labels_on_both_sides(
    df: pd.DataFrame,
    split: SplitResult,
    target_column: str,
) -> None:
    """Fail loudly when the split cannot measure one of the labels."""
    labels = df[target_column].to_numpy()

    train_labels = set(labels[split.train_index])
    test_labels = set(labels[split.test_index])

    only_train = sorted(train_labels - test_labels)
    only_test = sorted(test_labels - train_labels)

    if only_train or only_test:
        raise ValueError(
            "The split cannot measure every label. "
            f"Only in train: {only_train}. Only in test: {only_test}. "
            "Use more template groups or a different fold."
        )


def label_distribution(
    df: pd.DataFrame,
    indices: Sequence[int],
    target_column: str = TARGET_COLUMN,
) -> dict:
    """Return the label counts of the given rows."""
    values = df[target_column].to_numpy()[np.asarray(indices)]

    counts: dict = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1

    return dict(sorted(counts.items()))


def split_frame(
    df: pd.DataFrame,
    split: SplitResult,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``(train, test)`` frames described by ``split``."""
    return (
        df.iloc[split.train_index].reset_index(drop=True),
        df.iloc[split.test_index].reset_index(drop=True),
    )
