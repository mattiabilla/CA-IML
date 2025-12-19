"""
Hyperparameter search configuration and Optuna optimization utilities.

This module defines:

- the full hyperparameter grid specification for all models used in the study
- dynamically adjusted search ranges depending on dataset size
- a unified Optuna-based optimizer (`optuna_hyp_opt`) that returns
  the best hyperparameter set based on the selected scoring metric

No training or evaluation happens here: the optimizer interacts with external
model-performance functions passed as callables.
"""

import optuna
from sklearn.model_selection import train_test_split

from ..config.config import *


# ============================================================================
#                 MODEL-SPECIFIC HYPERPARAMETER CONFIGURATION
# ============================================================================

def get_models_optuna_config(n_samples, n_features, SEED=42):
    """
    Return the hierarchical Optuna search space for all models.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of input features.
    SEED : int
        Random seed for reproducibility at model level.

    Returns
    -------
    dict
        A nested dictionary specifying parameter types and ranges
        for each model under 'clf' and 'regr'.
    """
    return {
        "clf": {
            "knn": {
                "n_neighbors": {
                    "range": (3, min(100, max(3, int(0.2 * n_samples)))),
                    "type": "int"
                }
            },
            "decision_tree": {
                "min_samples_leaf": {"range": (0.01, 0.3), "type": "float"},
                "criterion": {"range": ["gini", "entropy"], "type": "categorical"},
                "max_depth": {"range": (2, max(2, n_features / 2)), "type": "int"},
                "random_state": {"fixed": SEED},
            },
            "logistic_regression": {
                "solver": {"range": ["lbfgs", "newton-cholesky"], "type": "categorical"},
                "C": {"range": (0.01, 10.0), "type": "logfloat"},
                "penalty": {"range": ["l2", None], "type": "categorical"},
                "max_iter": {"fixed": 1000},
                "random_state": {"fixed": SEED},
            },
            "gosdt": {
                "GDBT_n_estimators": {"range": (20, 40), "type": "int"},
                "GDBT_max_depth": {"range": (2, 6), "type": "int"},
                "GDBT_random_state": {"fixed": SEED},
                "GOSDT_regularization": {"range": (0.01, 0.2), "type": "logfloat"},
                "GOSDT_similar_support": {"range": [False, True], "type": "categorical"},
                "GOSDT_allow_small_reg": {"fixed": True},
                "GOSDT_time_limit": {"fixed": 120},
                "GOSDT_depth_budget": {"range": (2, 20), "type": "int"},
            },
            "naive_bayes": {
                "var_smoothing": {"range": (1e-12, 1e-6), "type": "logfloat"}
            },
            "ebm": {
                "interactions": {"range": (0, 3), "type": "int"},
                "learning_rate": {"range": (0.01, 0.05), "type": "logfloat"},
                "max_rounds": {"range": (100, 200), "type": "int"},
                "min_samples_leaf": {"range": (2, 5), "type": "int"},
                "n_jobs": {"fixed": -1},
                "random_state": {"fixed": SEED},
            },
            "igann": {
                "n_hid": {"range": (5, 50), "type": "int"},
                "n_estimators": {"range": (100, 5000), "type": "int"},
                "boost_rate": {"range": (0.001, 0.5), "type": "float"},
                "random_state": {"fixed": SEED},
            },
        },

        "regr": {
            "linear_regression": {
                "n_jobs": {"fixed": 4}
            },
            "decision_tree_regressor": {
                "random_state": {"fixed": SEED},
                "max_depth": {"range": (3, max(3, n_features / 2)), "type": "int"},
            },
            "symbolic_regression": {
                "select_k_features": {
                    "range": (min(2, n_features), min(n_features, 12)),
                    "type": "int"
                },
                "niterations": {"range": (100, 500), "type": "int"},
                "populations": {"range": (20, 100), "type": "int"},
                "population_size": {"range": (32, 96), "type": "int"},
                "model_selection": {"fixed": "best"},
                "weight_optimize": {"range": (0.001, 0.1), "type": "logfloat"},
                "random_state": {"fixed": SEED},
                "timeout_in_seconds": {"fixed": 120},
                "deterministic": {"fixed": True},
                "procs": {"fixed": 0},
                "temp_equation_file": {"fixed": True},
                "parallelism": {"fixed": "serial"},
            },
            "ebm": {
                "interactions": {"range": (0, 3), "type": "int"},
                "learning_rate": {"range": (0.01, 0.05), "type": "logfloat"},
                "max_rounds": {"range": (100, 200), "type": "int"},
                "min_samples_leaf": {"range": (2, 5), "type": "int"},
                "n_jobs": {"fixed": -1},
                "random_state": {"fixed": SEED},
            },
            "knn": {
                "n_neighbors": {
                    "range": (3, min(100, max(3, int(0.2 * n_samples)))),
                    "type": "int"
                }
            },
            "lasso": {
                "random_state": {"fixed": SEED},
                "max_iter": {"fixed": 10000},
                "alpha": {
                    "range": (1e-4, 10.0),
                    "type": "float",
                    "scale": "log"
                }
            },
            "glm": {
                "max_iter": {"fixed": 10000},
                "alpha": {
                    "range": (1e-4, 1.0),
                    "type": "float",
                    "scale": "log"
                }
            },
            "poly_lasso": {
                "poly__degree": {"range": (2, 5), "type": "int"},
                "poly__include_bias": {"fixed": False},
                "lasso__criterion": {"fixed": "bic"},
                "lasso__max_iter": {"fixed": 500},
            },
            "igann": {
                "n_hid": {"range": (5, 50), "type": "int"},
                "n_estimators": {"range": (100, 5000), "type": "int"},
                "boost_rate": {"range": (0.001, 0.5), "type": "float"},
                "random_state": {"fixed": SEED},
            },
        }
    }


# ============================================================================
#                           OPTUNA OPTIMIZATION LOGIC
# ============================================================================

def optuna_hyp_opt(model, problem, function, X, y):
    """
    Run Optuna hyperparameter optimization using the model-specific search space.

    Parameters
    ----------
    model : str
        Model identifier (e.g., 'knn', 'ebm', 'symbolic_regression').
    problem : {'clf', 'regr'}
        Specifies task type.
    function : callable
        A function with signature f(params, X_train, X_val, y_train, y_val)
        returning a score dictionary.
    X, y : array-like or DataFrame
        Dataset used for optimization.

    Returns
    -------
    dict
        Best hyperparameters found by Optuna.
    """

    def objective(trial):
        # Construct parameter dictionary from config
        params = {}
        param_config = get_models_optuna_config(
            X.shape[0], X.shape[1]
        )[problem][model]

        for key, cfg in param_config.items():
            if "fixed" in cfg:
                params[key] = cfg["fixed"]

            elif "range" in cfg and "type" in cfg:
                if cfg["type"] == "int":
                    params[key] = trial.suggest_int(key, *cfg["range"])

                elif cfg["type"] == "float":
                    params[key] = trial.suggest_float(key, *cfg["range"])

                elif cfg["type"] == "logfloat":
                    params[key] = trial.suggest_float(
                        key, *cfg["range"], log=True
                    )

                elif cfg["type"] == "categorical":
                    params[key] = trial.suggest_categorical(
                        key, cfg["range"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported parameter type for '{key}': {cfg['type']}"
                    )

            else:
                raise ValueError(
                    f"Invalid parameter configuration: {key} -> {cfg}"
                )

        # Train/validation split
        if problem == "clf":
            Xtr, Xval, ytr, yval = train_test_split(
                X, y, test_size=OPT_SINGLE_TEST_SIZE,
                random_state=SEED, stratify=y
            )
        else:
            Xtr, Xval, ytr, yval = train_test_split(
                X, y, test_size=OPT_SINGLE_TEST_SIZE,
                random_state=SEED
            )

        # Evaluate
        score_dict = function(params, Xtr, Xval, ytr, yval)

        # Select appropriate metric
        score = score_dict["Fbeta"] if problem == "clf" else score_dict["R2"]

        trial.set_user_attr("full_params", params)
        return score

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    return study.best_trial.user_attrs["full_params"]
