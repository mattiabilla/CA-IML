"""
End-to-end evaluator for all datasets, models, and splits.

This module orchestrates:

- preprocessing (via `single_dataset_preprocessing`)
- scaling (via `scaler`)
- hyperparameter search (via Optuna wrapper `optuna_hyp_opt`)
- model training / loading
- inference and metric computation
- multi-split aggregation
- optional CSV persistence of split-level results

The logic is intentionally sequential and explicit for reproducibility and
alignment with the experimental protocol described in the paper.
"""

import time
import gc
import pandas as pd
from pathlib import Path

from ..config.config import *
from ..models import regression
from ..models import classification
from ..preprocessing.preprocessing import single_dataset_preprocessing, scaler
from ..models.hyper_opt import optuna_hyp_opt


# ============================================================================
#                                MAIN EVALUATOR
# ============================================================================

def complete_evaluator(
    problem,
    evaluation,
    models_name,
    ds_names,
    save,
    which_split_evaluate,
    fit=True
):
    """
    Execute the full evaluation pipeline over multiple datasets and models.

    Parameters
    ----------
    problem : {'clf', 'regr'}
        Task type.
    evaluation : {'IS', 'OOS'}
        Evaluation strategy.
    models_name : list[str]
        Models to evaluate (identified by name prefix used in classification/regression).
    ds_names : list[str]
        List of dataset names.
    save : bool
        Whether to save per-split results in CSV files.
    which_split_evaluate : list[int]
        List of split IDs to evaluate (e.g., [0, 1, 2, 3]).
    fit : bool
        If True, train new models; if False, load saved model files.

    Returns
    -------
    (DataFrame, DataFrame)
        - df: mean performance per (Dataset, Model)
        - split_separated_df: performance per (Dataset, Model, Split)
    """

    # Aggregators
    multi_ds_results = []
    rows_separated_for_split = []

    # ========================================================================
    #                     LOOP OVER ALL DATASETS
    # ========================================================================
    for ds_name in ds_names:
        try:
            data, _, bin_cols = single_dataset_preprocessing(
                ds_name, problem, evaluation
            )
            splitted_data = data
        except Exception as e:
            logger.error(f"{ds_name}  {e}")
            continue

        model_counter = 0

        # ====================================================================
        #                   LOOP OVER ALL MODELS FOR THIS DATASET
        # ====================================================================
        for model in models_name:
            model_counter += 1
            function_name = model + "_performances"

            if problem == "clf":
                function = getattr(classification, function_name)
            else:
                function = getattr(regression, function_name)

            results = []

            # ----------------------------------------------------------------
            #               LOOP OVER ALL SPLITS FOR THIS MODEL
            # ----------------------------------------------------------------
            try:
                for i, split in enumerate(splitted_data):

                    # Skip unwanted splits
                    if i not in which_split_evaluate:
                        continue

                    X_train = split["X_train"]
                    y_train = split["y_train"]
                    X_test = split["X_test"]
                    y_test = split["y_test"]

                    # Scaling (feature + optional target)
                    X_train, X_test, y_train, y_test, y_transformer = scaler(
                        X_train, X_test, y_train, y_test, problem, bin_cols
                    )

                    start_time = time.time()

                    # --------------------------------------------------------
                    #              Hyperparameter search (if required)
                    # --------------------------------------------------------
                    if fit:
                        hyperparams = optuna_hyp_opt(
                            model, problem, function, X_train, y_train
                        )

                        # Extended time limits for computationally heavy models
                        if model == "gosdt":
                            hyperparams["GOSDT_time_limit"] = 300
                        if model == "symbolic_regression":
                            hyperparams["timeout_in_seconds"] = 300
                    else:
                        hyperparams = False

                    model_file_name = f"{ds_name}_{model}_{i}"

                    # --------------------------------------------------------
                    #                Model fit or load + inference
                    # --------------------------------------------------------
                    if problem == "regr":
                        if fit:
                            result = function(
                                hyperparams,
                                X_train, X_test, y_train, y_test,
                                model_file_name,
                                fit
                            )
                        else:
                            load_params = {
                                "ds_name": ds_name,
                                "evaluation": evaluation,
                                "split": i,
                            }
                            result = function(
                                hyperparams,
                                X_train, X_test, y_train, y_test,
                                model_file_name,
                                fit,
                                load_params,
                            )
                    else:
                        # classification always fits or evaluates directly
                        result = function(
                            hyperparams,
                            X_train, X_test, y_train, y_test,
                            model_file_name
                        )

                    # --------------------------------------------------------
                    #                Timing and aggregator update
                    # --------------------------------------------------------
                    duration = time.time() - start_time
                    result["Duration"] = duration

                    row_split = {
                        "Dataset": ds_name,
                        "Model": model,
                        "Split": f"split{i}",
                        **result
                    }

                    rows_separated_for_split.append(row_split)

                    # --------------------------------------------------------
                    #                     Optional CSV persistence
                    # --------------------------------------------------------
                    if save:
                        try:
                            output_dir = (
                                Path(__file__).resolve().parent / "output/separated"
                            )
                            output_dir.mkdir(parents=True, exist_ok=True)

                            file_path = (
                                output_dir
                                / f"{ds_name}_{problem}_{evaluation}_split{i}.csv"
                            )
                            df_row = pd.DataFrame([row_split])

                            # First model writes header, others append
                            if model_counter == 1:
                                df_row.to_csv(
                                    file_path, mode="w", header=True, index=False
                                )
                            else:
                                df_row.to_csv(
                                    file_path, mode="a", header=False, index=False
                                )

                        except Exception as e:
                            logger.error(
                                f"Error while saving split-level CSV: {e}"
                            )

                    # Memory cleanup
                    del X_train, X_test, y_train, y_test
                    gc.collect()

                    # Store raw split result
                    results.append(result)

            except Exception as e:
                logger.error(f"ds-name: {ds_name} | {e}")
                continue

            # ----------------------------------------------------------------
            #              Compute mean metrics across splits for this model
            # ----------------------------------------------------------------
            results_df = pd.DataFrame(results)
            mean_results = results_df.mean().to_dict()

            del results, results_df
            gc.collect()

            row_summary = {
                "Dataset": ds_name,
                "Model": model,
                **mean_results
            }
            multi_ds_results.append(row_summary)

    # ============================================================================
    #                       BUILD FINAL SUMMARY DATAFRAMES
    # ============================================================================
    df = pd.DataFrame(multi_ds_results)
    df.set_index(["Dataset", "Model"], inplace=True)
    pd.set_option("display.float_format", "{:.4f}".format)

    split_separated_df = pd.DataFrame(rows_separated_for_split)
    split_separated_df.set_index(["Dataset", "Model", "Split"], inplace=True)

    return df, split_separated_df
