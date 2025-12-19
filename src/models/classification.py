"""
Classification model-level training, prediction and complexity
measurement routines. 
This module does not manage experiment orchestration, which is
handled separately by the evaluator.

This module implements:

- model-specific training/evaluation wrappers
- structural complexity measures for interpretable models
- a unified metric computation interface

Functions here return dictionaries of evaluation metrics and (when applicable)
structural complexity values to allow uniform comparison across models in
the paper.
"""

import math
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from numpy import log2
from gosdt._tree import Leaf
import multiprocessing as mp

from ..config.config import *
from .utils_model_io import save_model


# ============================================================================
#                            METRIC COMPUTATION
# ============================================================================

def classification_metric(y_true, y_pred, model=None, model_file_name=None):
    """
    Compute evaluation metrics for classification models.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    model : sklearn-like estimator, optional
        Model instance to store on disk (if requested).
    model_file_name : str, optional
        Filename used when saving the model.

    Returns
    -------
    dict
        Dictionary containing macro-averaged precision, recall, F-beta,
        accuracy, and model complexity (if available).
    """
    # Main performance metrics (macro-averaged)
    precision, recall, fbeta, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=True
    )

    accuracy = accuracy_score(y_true, y_pred)

    results = {
        "Precision": precision,
        "Recall": recall,
        "Fbeta": fbeta,
        "Accuracy": accuracy,
        "complexity": 0
    }

    # Optionally save the model to disk
    if model:
        save_model(model, model_file_name)
        

    logger.info(results)
    return results


# ============================================================================
#                              MODEL WRAPPERS
# ============================================================================

def knn_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """Train and evaluate a KNN classifier."""
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(**hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if save:
        return classification_metric(y_test, y_pred, model, save)
    return classification_metric(y_test, y_pred)


def logistic_regression_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """Train and evaluate Logistic Regression."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(**hyperparams).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if save:
        return classification_metric(y_test, y_pred, model, save)
    return classification_metric(y_test, y_pred)


def decision_tree_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """Train and evaluate a Decision Tree with a depth rule based on feature count."""
    from sklearn.tree import DecisionTreeClassifier

    # Set depth using log2(#features) + 2 as heuristic
    hyperparams['max_depth'] = math.ceil(log2(X_train.shape[1])) + 2

    model = DecisionTreeClassifier(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = classification_metric(y_test, y_pred, model, save) if save else classification_metric(y_test, y_pred)

    results["complexity"] = model.tree_.node_count
    return results


# ============================================================================
#                       GOSDT STRUCTURAL COMPLEXITY
# ============================================================================

def _get_gosdt_tree_complexity(node):
    """
    Recursively compute structural complexity of a GOSDT tree,
    defined as total number of nodes.
    """
    if isinstance(node, Leaf):
        return 1
    left = _get_gosdt_tree_complexity(node.left_child)
    right = _get_gosdt_tree_complexity(node.right_child)
    return 1 + left + right


def gosdt_fit_predict(X_train_guessed, y_train, X_test_guessed, GOSDT_PARAMS, warm_labels):
    """
    Fit GOSDT in an isolated subprocess and return predictions and tree root.

    Used to avoid residual memory accumulation from C++ bindings.
    """
    from gosdt import GOSDTClassifier

    clf = GOSDTClassifier(**GOSDT_PARAMS, verbose=False, debug=False)
    clf.fit(X_train_guessed, y_train, y_ref=warm_labels)

    y_pred = clf.predict(X_test_guessed)
    tree_root = clf.trees_[0].tree
    return y_pred, tree_root, clf


def gosdt_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """
    Train and evaluate GOSDT using threshold guessing and GBDT warm start,
    executed in a forked subprocess for memory isolation.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from gosdt import ThresholdGuessBinarizer
    import gc

    # Separate GDBT and GOSDT parameters based on prefix
    GDBT_PARAMS = {k.replace("GDBT_", ""): v for k, v in hyperparams.items() if k.startswith("GDBT_")}
    GOSDT_PARAMS = {k.replace("GOSDT_", ""): v for k, v in hyperparams.items() if k.startswith("GOSDT_")}

    # Binarization step
    binarizer = ThresholdGuessBinarizer(**GDBT_PARAMS)
    X_train_bin = binarizer.fit_transform(X_train, y_train)
    X_test_bin = binarizer.transform(X_test)

    # Warm-start labels via gradient boosting
    warm_model = GradientBoostingClassifier(**GDBT_PARAMS)
    warm_model.fit(X_train_bin, y_train)
    warm_labels = warm_model.predict(X_train_bin)

    # Fit GOSDT inside a separate process
    ctx = mp.get_context('fork')
    with ctx.Pool(1) as pool:
        result = pool.apply(gosdt_fit_predict, (X_train_bin, y_train, X_test_bin, GOSDT_PARAMS, warm_labels))

    y_pred, tree_root, clf = result

    results = classification_metric(y_test, y_pred, clf, save) if save else classification_metric(y_test, y_pred)

    results["complexity"] = _get_gosdt_tree_complexity(tree_root)

    gc.collect()
    return results


# ============================================================================
#                    NAIVE BAYES AND STRUCTURAL COMPLEXITY
# ============================================================================

def _gaussian_nb_complexity(model, X):
    """Structural complexity = 2 * (#classes) * (#features)."""
    n_classes = len(model.classes_)
    n_features = X.shape[1]
    return 2 * n_classes * n_features


def naive_bayes_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """Train and evaluate Gaussian Naive Bayes."""
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB(**hyperparams).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = classification_metric(y_test, y_pred, model, save) if save else classification_metric(y_test, y_pred)

    results["complexity"] = _gaussian_nb_complexity(model, X_train)
    return results


# ============================================================================
#                                EBM MODEL
# ============================================================================

def _ebm_structural_complexity(model):
    """
    Structural complexity of EBM.    
    """
    return sum(len(score) for score in model.term_scores_)


def ebm_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """Train and evaluate an Explainable Boosting Machine (EBM)."""
    from interpret.glassbox import ExplainableBoostingClassifier

    model = ExplainableBoostingClassifier(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = classification_metric(y_test, y_pred, model, save) if save else classification_metric(y_test, y_pred)

    results['complexity'] = _ebm_structural_complexity(model)
    return results


# ============================================================================
#                                 IGANN
# ============================================================================

def igann_performances(hyperparams, X_train, X_test, y_train, y_test, save=False):
    """Train and evaluate an IGANN classifier."""
    from igann import IGANNClassifier

    model = IGANNClassifier(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if save:
        return classification_metric(y_test, y_pred, model, save)
    return classification_metric(y_test, y_pred)
