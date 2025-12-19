"""
Splitting utilities for in-sample (IS) and out-of-sample (OOS) evaluation.

This module provides:

- unified entry point `splitter` for IS/OOS splitting
- K-fold and stratified K-fold logic for IS evaluation
- custom quartile-based OOS splitting for regression
- custom class-imbalance–controlled OOS splitting for classification
- optional normalization utilities
- reproducible sampling based on global SEED

No experiment orchestration is implemented here; the module focuses solely
on dataset partitioning strategies as required by the experimental design.
"""

from collections import Counter
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, KFold

from ..config.config import *


# ============================================================================
#                             PUBLIC INTERFACE
# ============================================================================

def splitter(df, problem, evaluation):
    """
    Wrapper for splitting strategy based on problem type and evaluation setting.

    Parameters
    ----------
    df : DataFrame
        Dataset including a 'target' column.
    problem : {'clf', 'regr'}
        Specifies task type.
    evaluation : {'IS', 'OOS'}
        In-sample or out-of-sample strategy.

    Returns
    -------
    list[dict]
        A list of fold dictionaries, each containing:
        { X_train, X_test, y_train, y_test, ... }
    """
    assert evaluation in ("IS", "OOS")
    assert problem in ("clf", "regr")

    if evaluation == "IS":
        return _is_splitter(df, problem)
    else:
        return _oos_splitter(df, problem)


# ============================================================================
#                           IN-SAMPLE SPLITTING
# ============================================================================

def _is_splitter(df, problem):
    """
    Perform standard K-fold (regression) or Stratified K-fold (classification)
    in-sample splitting.

    Parameters
    ----------
    df : DataFrame
    problem : {'clf', 'regr'}

    Returns
    -------
    list[dict]
        Splits with 'X_train', 'X_test', 'y_train', 'y_test'
    """
    splitted_data = []

    X = df.drop(columns=["target"])
    y = df["target"].values

    if problem == "clf":
        class_counts = Counter(y)
        if min(class_counts.values()) < K_SPLITS:
            raise Exception(
                f"Dataset contains at least one class with fewer than {K_SPLITS} samples."
            )

        skf = StratifiedKFold(
            n_splits=K_SPLITS, shuffle=True, random_state=SEED
        )
        splits_r = skf.split(X, y)

    else:  # regression
        kf = KFold(
            n_splits=K_SPLITS, shuffle=True, random_state=SEED
        )
        splits_r = kf.split(X)

    for train_idx, test_idx in splits_r:
        splitted_data.append({
            "X_train": X.iloc[train_idx],
            "X_test": X.iloc[test_idx],
            "y_train": y[train_idx],
            "y_test": y[test_idx],
        })

    return splitted_data


# ============================================================================
#                         OUT-OF-SAMPLE SPLITTING
# ============================================================================

def _oos_splitter(df, problem):
    """
    Generate K out-of-sample splits using:
    - quartile-based splitting for regression,
    - altered class-proportion sampling for classification.

    Returns
    -------
    list[dict]
        List of split dictionaries with indices included.
    """
    splitted_data = []

    for k in range(K_SPLITS):
        current_seed = SEED + k

        if problem == "regr":
            train_idx, test_idx = __oos_regression(
                df, random_state=current_seed
            )
        else:
            train_idx, test_idx = __oos_classification(
                df, random_state=current_seed
            )

        # Consistency checks
        assert set(train_idx).isdisjoint(test_idx), f"Overlap at fold {k}"
        assert set(train_idx).union(test_idx) == set(df.index), f"Missing observations at fold {k}"

        # Build split
        df_train = df.loc[train_idx].reset_index(drop=True)
        df_test = df.loc[test_idx].reset_index(drop=True)

        splitted_data.append({
            "X_train": df_train.drop(columns=["target"]),
            "X_test": df_test.drop(columns=["target"]),
            "y_train": df_train["target"].values,
            "y_test": df_test["target"].values
        })

    return splitted_data


# ============================================================================
#                       OOS REGRESSION SPLITTING
# ============================================================================

def __oos_regression(df, random_state):
    """
    Quartile-based out-of-sample splitting for regression.

    Steps:
    - choose a quartile at random
    - assign all samples in that quartile as test set
    - remaining samples become training set

    Returns
    -------
    (ndarray, ndarray)
        train_indexes, test_indexes
    """
    y = df["target"].values
    rng = np.random.default_rng(seed=random_state)

    # Choose quartile
    n_quartile = rng.choice([1, 2, 3, 4])
    quartiles = np.quantile(y, [0, 0.25, 0.50, 0.75, 1])

    q_low = quartiles[n_quartile - 1]
    q_high = quartiles[n_quartile]

    all_idx = df.index.to_numpy()

    # Select test indices
    test_idx = df[
        (df["target"] > q_low) & (df["target"] <= q_high)
    ].index.to_numpy()

    if len(test_idx) == 0:
        raise ValueError("Regression OOS split produced an empty test set.")

    train_idx = np.setdiff1d(all_idx, test_idx)

    logger.info(
        f"Splitting on target between {q_low:.2f} and {q_high:.2f} (quartile {n_quartile})"
    )

    return train_idx, test_idx


# ============================================================================
#                     OOS CLASSIFICATION SPLITTING
# ============================================================================

def __oos_classification(
    df, random_state=SEED
):
    """
    Perform OOS splitting for binary classification with altered class proportions.

    Returns
    -------
    (ndarray, ndarray)
        train_indexes, test_indexes

    Raises
    ------
    ValueError
        If constraints on class counts cannot be satisfied.
    """
    assert 0 < F_OOS_CLF < 1
    assert L_OOS_CLF <= U_OOS_CLF
    assert L_OOS_CLF >= 0 and U_OOS_CLF <= 1

    rng = np.random.default_rng(seed=random_state)

    y = df["target"]
    D = len(y)
    T = int(F_OOS_CLF * D)

    # Identify positive/negative labels
    unique_classes = np.sort(y.unique())
    if not np.array_equal(unique_classes, [0, 1]):
        pos_val = unique_classes[-1]
        neg_val = unique_classes[0]
        alpha_D = (y == pos_val).mean()
    else:
        pos_val, neg_val = 1, 0
        alpha_D = y.mean()

    # Bounds on fraction of positives in test
    if alpha_D > 0.5:
        lower = L_OOS_CLF * alpha_D / F_OOS_CLF
        upper = alpha_D
    else:
        lower = alpha_D
        upper = U_OOS_CLF * alpha_D / F_OOS_CLF

    lower = max(0, lower)
    upper = min(1, upper)

    # Sample alpha_test
    alpha_test = np.random.uniform(lower, upper)
    n_pos_test = int(alpha_test * T)
    n_neg_test = T - n_pos_test

    # Indices by class
    pos_idx = y[y == pos_val].index.to_numpy()
    neg_idx = y[y == neg_val].index.to_numpy()

    # Check feasibility
    if n_pos_test > len(pos_idx) or n_neg_test > len(neg_idx):
        raise ValueError(
            "Not enough positive/negative examples to satisfy constraints."
        )

    # Sampling
    pos_test_idx = rng.choice(pos_idx, size=n_pos_test, replace=False)
    neg_test_idx = rng.choice(neg_idx, size=n_neg_test, replace=False)

    test_idx = np.concatenate([pos_test_idx, neg_test_idx])
    test_idx = shuffle(test_idx, random_state=random_state)

    # Remaining to training
    all_idx = df.index.to_numpy()
    train_idx = np.setdiff1d(all_idx, test_idx)

    return train_idx, test_idx