"""
Preprocessing utilities for PMLB datasets used in classification and regression tasks.

This module provides:

- NaN filtering and consistency checks
- detection and encoding of categorical variables
- dataset-level cleaning routines
- error classes for preprocessing failures
- a unified single-dataset preprocessing pipeline
- a utility function for scale data

No experiment orchestration occurs here; the module focuses exclusively on
dataset loading, cleaning and feature-type handling.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from pmlb import fetch_data
from pmlb.support_funcs import get_feature_type

from ..config.config import *
from ..preprocessing.splitting import splitter


# ============================================================================
#                                 EXCEPTIONS
# ============================================================================

class NonBinaryDs(Exception):
    """Raised when a classification dataset is non-binary under CLF_ONLY_BINARY=True."""
    pass


class TooFewRowsBeforePrep(Exception):
    """Raised when a dataset has fewer than MIN_OBSERVATIONS rows."""
    pass


class TooFewCols(Exception):
    """Raised when remaining columns during NaN filtering are less than REMAINING_COL_AFTER_NAN_COL_ELIMINATION."""
    pass


class TooFewRowsAfterPrep(Exception):
    """
    Raised when too many rows are discarded during NaN removal,
    violating REMAINING_OBSERVATIONS_AFTER_NAN_DROP.
    """
    pass


# ============================================================================
#                              UTILITY FUNCTIONS
# ============================================================================

def remove_categorical(ds):
    """
    Perform categorical variable encoding using one-hot encoding.

    Categorical features are identified using PMLB's feature type utility.
    Binary features are treated as categorical and also encoded.

    Parameters
    ----------
    ds : DataFrame
        Original dataset including the target column.

    Returns
    -------
    (DataFrame, list)
        Dataset with categorical variables one-hot encoded,
        list of newly created binary columns.
    """
    feat_names = [col for col in ds.columns if col != TARGET_NAME]

    # Identify categorical or binary columns
    types = [get_feature_type(ds[col], include_binary=True) for col in feat_names]
    categorical_columns = [
        name for name, t in zip(feat_names, types)
        if t in ("categorical", "binary")
    ]

    before_cols = set(ds.columns)

    # Apply one-hot encoding
    ds = pd.get_dummies(ds, columns=categorical_columns, drop_first=True)

    after_cols = set(ds.columns)
    new_bin_cols = list(after_cols - before_cols)

    return ds, new_bin_cols


def remove_nan(ds):
    """
    Remove NaN values using this strategy:

    1. Drop columns where the proportion of NaN exceeds MAX_NAN_SINGLE_COLUMN.
    2. Drop rows still containing NaNs.
    3. Validate that at least REMAINING_COL_AFTER_NAN_COL_ELIMINATION
       and REMAINING_OBSERVATIONS_AFTER_NAN_DROP fractions remain.

    Parameters
    ----------
    ds : DataFrame
        Dataset with potential NaN values.

    Returns
    -------
    DataFrame
        Cleaned dataset after column/row filtering.

    Raises
    ------
    TooFewCols
        If too many columns are removed during step (1).
    TooFewRowsAfterPrep
        If too many rows are removed during step (2).
    """
    initial_n_features = ds.shape[1]

    # Drop columns with excessive NaN
    ds = ds.drop(columns=ds.columns[ds.isna().mean() > MAX_NAN_SINGLE_COLUMN])
    n_features = ds.shape[1]

    if n_features / initial_n_features < REMAINING_COL_AFTER_NAN_COL_ELIMINATION:
        print("Too few cols")
        raise TooFewCols(
            f"Remaining columns < {REMAINING_COL_AFTER_NAN_COL_ELIMINATION * 100}%"
        )

    # Drop rows with remaining NaN
    n_obs = len(ds)
    ds = ds.dropna()
    final_n_obs = len(ds)

    if final_n_obs / n_obs < REMAINING_OBSERVATIONS_AFTER_NAN_DROP:
        print("Too few rows after prep")
        raise TooFewRowsAfterPrep(
            f"Remaining observations < {REMAINING_OBSERVATIONS_AFTER_NAN_DROP * 100}%"
        )

    return ds


# ============================================================================
#                                SCALING UTILITY
# ============================================================================

def scaler(X_train, X_test, y_train, y_test, problem, bin_cols):
    """
    Apply robust scaling to numerical features and passthrough for binary features.
    Optionally scale the target in regression tasks.

    Parameters
    ----------
    X_train, X_test : DataFrame
        Train and test feature matrices.
    y_train, y_test : array-like
        Train and test labels/targets.
    problem : {'clf', 'regr'}
        Specifies task type.
    bin_cols : list
        Columns treated as binary features.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler or None)
        Scaled X_train, X_test, y_train, y_test, and optional y_transformer.
    """
    assert problem in ("regr", "clf")

    num_features = [c for c in X_train.columns if c not in bin_cols]
    print(f"ARRIVED TO SCALER: {len(bin_cols)} BIN COLS")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_features),
            ("bin", "passthrough", bin_cols),
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    y_transformer = None
    if problem == "regr":
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

        y_transformer = RobustScaler().fit(y_train)
        y_train = y_transformer.transform(y_train).flatten()
        y_test = y_transformer.transform(y_test).flatten()

    return X_train, X_test, y_train, y_test, y_transformer


# ============================================================================
#                         MAIN PREPROCESSING PIPELINE
# ============================================================================


def single_dataset_preprocessing(dataset_name, problem, evaluation=None, split=True):
    """
    Preprocess a single PMLB dataset.

    The preprocessing pipeline applies:

    - dataset loading
    - minimum row check
    - NaN filtering
    - categorical encoding
    - binary-class constraint for classification (if enabled)
    - train/test splitting through `splitter`

    Parameters
    ----------
    dataset_name : str or DataFrame
        Name of a PMLB dataset or a DataFrame already loaded.
    problem : {'clf', 'regr'}
        Task type (classification or regression).
    evaluation : {'OOS', 'IS'}, optional
        Determines splitting strategy (passed to splitter).
    split : bool
        If True, split into train/test; otherwise return full dataset.

    Returns
    -------
    (object, tuple, list)
        - processed dataset or split dictionary,
        - dataset shape after preprocessing,
        - list of generated binary columns.

    Raises
    ------
    TooFewRowsBeforePrep
        If dataset has fewer than MIN_OBSERVATIONS rows.
    NonBinaryDs
        If classification dataset is non-binary under CLF_ONLY_BINARY=True.
    TooFewCols / TooFewRowsAfterPrep
        For excessive column/row loss during preprocessing.
    """
    if evaluation:
        assert evaluation in ("OOS", "IS")

    # Load dataset
    if isinstance(dataset_name, str):
        ds = fetch_data(dataset_name, dropna=False)
    else:
        ds = dataset_name

    # Enforce binary classification constraint if required
    if problem == "clf" and CLF_ONLY_BINARY:
        if len(ds["target"].unique()) > 2:
            print("Non binary ds")
            raise NonBinaryDs(
                "Classification dataset has non-binary target (CLF_ONLY_BINARY=True)."
            )

    logger.info("=" * 30)
    logger.info(dataset_name)
    logger.info(f"Start: {ds.shape}")
    logger.info(problem)

    # Minimum observation check
    if len(ds) < MIN_OBSERVATIONS:
        print("Too few rows before prep")
        raise TooFewRowsBeforePrep(
            "Dataset contains fewer than MIN_OBSERVATIONS rows."
        )

    # Remove NaN values
    ds = remove_nan(ds)
    logger.info(f"After NaN removal: {ds.shape}")

    # Handle categorical variables
    ds, bin_cols = remove_categorical(ds)
    logger.info(f"After categorical handling: {ds.shape}")

    # Apply splitting strategy
    if split:
        data = splitter(ds, problem, evaluation)
    else:
        data = ds

    return data, ds.shape, bin_cols
