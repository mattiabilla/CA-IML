"""
Dataset complexity metrics used across classification and regression tasks.

This module provides:

- general-purpose metrics (T2, T3)
- classification-specific metrics (F1, N3, C1_clf)
- regression-specific metrics (C1_regr, L1, S3)

and

- a dataset scaling utility ensuring consistent preprocessing
- a batch-processing routine to compute metrics for multiple datasets

Experiment orchestration is not performed here; this module focuses solely
on dataset-level complexity metrics computation.
"""

import os
import numpy as np
import pandas as pd
from math import log
from collections import Counter
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

# ============================================================================
#                            GENERAL METRICS
# ============================================================================

def calculate_t2(ds_shape):
    """Compute T2 = (#features excluding target) / (#instances)"""
    return (ds_shape[1] - 1) / ds_shape[0]


def calculate_t3(ds, var_threshold=0.95):
    """
    Compute T3 = (#PCA components required for var_threshold) / (#samples).

    PCA is applied to feature columns only.
    """
    pca = PCA(n_components=var_threshold)
    reduced = pca.fit_transform(ds.drop("target", axis=1))
    return reduced.shape[1] / reduced.shape[0]


# ============================================================================
#                            CLASSIFICATION METRICS
# ============================================================================

def calculate_f1(ds):
    """Compute the transformed Fisher discriminant ratio (F1)"""
    X = ds.drop("target", axis=1).values
    y = ds["target"].values
    classes = np.unique(y)

    max_rfi = 0
    for i in range(X.shape[1]):
        mu_fi = np.mean(X[:, i])
        num = 0
        den = 0

        for c in classes:
            Xc = X[y == c, i]
            ncj = len(Xc)
            mu_cj = np.mean(Xc)

            num += ncj * (mu_cj - mu_fi) ** 2
            den += np.sum((Xc - mu_cj) ** 2)

        rfi = num / den if den != 0 else 0
        max_rfi = max(max_rfi, rfi)

    return 1 / (1 + max_rfi)


def calculate_n3(ds):
    """Compute N3 = LOO error rate of a 1-NN classifier"""
    X = ds.drop("target", axis=1).values
    y = ds["target"].values

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    _, idx = nn.kneighbors(X, n_neighbors=2)

    pred = y[idx[:, 1]]
    return np.mean(pred != y)


def calculate_c1_clf(ds):
    """Compute normalized class entropy"""
    y = ds["target"].values
    counts = Counter(y)
    probs = np.array([v / len(y) for v in counts.values()])
    nc = len(counts)

    if nc <= 1:
        return 0
    return -np.sum(probs * np.log(probs)) / log(nc)


# ============================================================================
#                            REGRESSION METRICS
# ============================================================================

def calculate_c1_regr(ds):
    """Compute C1_regr = maximum Spearman correlation between features and target"""
    X = ds.drop("target", axis=1)
    y = ds["target"]

    max_corr = 0
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        max_corr = max(max_corr, abs(corr))
    return max_corr


def calculate_l1(ds):
    """Compute L1 = mean absolute error of Linear Regression fitted on the dataset"""
    X = ds.drop("target", axis=1).values
    y = ds["target"].values

    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    return np.mean(np.abs(residuals))


def calculate_s3(ds):
    """Compute S3 = LOO mean squared error of a 1-NN regressor"""
    X = ds.drop("target", axis=1).values
    y = ds["target"].values

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    _, idx = nn.kneighbors(X, n_neighbors=2)

    pred = y[idx[:, 1]]
    return np.mean((pred - y) ** 2)


# ============================================================================
#                            SCALING
# ============================================================================

def scale_dataset(X, bin_cols):
    """
    Apply robust scaling to numerical features and passthrough for binary features.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    bin_cols : list
        Columns considered binary.

    Returns
    -------
    (DataFrame)
        Scaled X.
    """

    num_features = [c for c in X.columns if c not in bin_cols]
    print(f"ARRIVED TO SCALER: {len(bin_cols)} BIN COLS")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_features),
            ("bin", "passthrough", bin_cols),
        ]
    )

    X_scaled_arr = preprocessor.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=num_features + bin_cols, index=X.index)

    return X_scaled


# ----------------------------------------------------------------------------
# MAIN BATCH ROUTINE
# ----------------------------------------------------------------------------

def calc_all_metrics(ds_names, problem, output_file):
    """
    Compute all dataset metrics for a list of datasets and save results.

    Parameters
    ----------
    ds_names : list[str]
        Names of datasets to evaluate.
    problem : {'regr', 'clf'}
        Type of problem, determines which meta-features are computed.
    output_file : str
        Path where results are written as CSV.

    Returns
    -------
    DataFrame
        Final accumulated results.
    """
    assert problem in ("regr", "clf")

    from ..preprocessing.preprocessing import single_dataset_preprocessing

    # Initialize result container
    if problem == "clf":
        results = {
            "dataset": [], "row": [], "col": [],
            "T2": [], "T3": [],
            "F1": [], "N3": [], "C1_clf": [],
        }
    else:
        results = {
            "dataset": [], "row": [], "col": [],
            "T2": [], "T3": [],
            "C1_regr": [], "L1": [], "S3": [],
        }

    i = 0

    for name in ds_names:
        i += 1
        try:
            print("=" * 30)
            print(i)

            ds, shape, bin_cols = single_dataset_preprocessing(name, problem, split=False)

            X, y = ds.drop(columns=["target"]), ds["target"].values
            X_scaled = scale_dataset(X, bin_cols)
            X_scaled["target"] = y
            ds = X_scaled

            assert shape == ds.shape
            results["dataset"].append(name)
            results["row"].append(shape[0])
            results["col"].append(shape[1] - 1)
            results["T2"].append(calculate_t2(shape))
            results["T3"].append(calculate_t3(ds))

            if problem == "clf":
                results["F1"].append(calculate_f1(ds))
                results["N3"].append(calculate_n3(ds))
                results["C1_clf"].append(calculate_c1_clf(ds))
            else:
                results["C1_regr"].append(calculate_c1_regr(ds))
                results["L1"].append(calculate_l1(ds))
                results["S3"].append(calculate_s3(ds))

            # Periodic saving
            if i % 25 == 0:
                df_tmp = pd.DataFrame(results)
                df_tmp.to_csv(output_file, mode="a", index=False,
                              header=not os.path.exists(output_file))
                results = {k: [] for k in results}
                print(f"✓ Saved {i} results to '{output_file}'")

        except Exception as e:
            print(f"Error on {name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, mode="a", index=False, header=not os.path.exists(output_file))

    return df
