"""
Main entry point for running experiments.

This script routes command-line arguments to the evaluation framework,
translates dataset indexes into dataset names, and triggers the full
evaluation procedure (training, testing, saving results).

It acts as a reproducibility interface for the experiments included in the paper.
"""

import argparse

from .config.config import *
from .evaluation.evaluation import complete_evaluator


def main(problem: str,
         evaluation: str,
         save: int,
         splits: list[int],
         indexes: list[int] | None = None,
         models_name: list[str] | None = None,
         fit: bool = True):
    """
    Execute the full evaluation workflow for the specified task.

    Parameters
    ----------
    problem : str
        Type of task, either "clf" (classification) or "regr" (regression).

    evaluation : str
        Evaluation strategy: in-sample ("IS") or out-of-sample ("OOS").

    save : int
        Whether to save output files (1) or not (0).

    splits : list of int
        List of split values controlling cross-validation.

    indexes : list of int, optional
        Dataset indexes to select. If None, all datasets for the given
        problem type are used.

    models_name : list of str, optional
        List of model identifiers. If None, all models are used.

    fit : bool
        If False, attempts to reload previously trained models from disk. Default is True.

    Returns
    -------
    dict or None
        Output from `complete_evaluator`.
    """

    print("Problem:", problem)
    print("Evaluation:", evaluation)
    print("Indexes:", indexes)
    print("Save:", save)
    print("Splits:", splits)
    print("Models:", models_name)
    print("Fit:", fit)

    # Map dataset indexes to actual dataset names.
    if problem == "regr":
        ds_names = (regr_dataset_names if not indexes else [regr_dataset_names[idx] for idx in indexes])

    elif problem == "clf":
        ds_names = (clf_dataset_names if not indexes else [clf_dataset_names[idx] for idx in indexes])
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    # Fallback to default models if none are provided.
    if not models_name:
        models_name = MODELS_NAME[problem]

    # Delegate execution to the main evaluation pipeline.
    return complete_evaluator(
        problem=problem,
        evaluation=evaluation,
        models_name=models_name,
        ds_names=ds_names,
        save=save,
        splits=splits,
        fit=fit,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line interface for running experiments.")

    parser.add_argument('--problem', type=str, required=True, help='Problem type: "clf" for classification, "regr" for regression.')
    parser.add_argument('--evaluation', type=str, required=True, help='Evaluation strategy: "IS" for in-sample or "OOS" for out-of-sample.')
    parser.add_argument('--indexes', type=int, nargs='+', required=False, help='Dataset indexes. If omitted, all datasets are used.')
    parser.add_argument('--save', type=int, required=True, help='Whether to save output files: 1 = yes, 0 = no.')
    parser.add_argument('--splits', type=int, nargs='+', required=True, help='List of split values for CV.')
    parser.add_argument('--modelsname', type=str, nargs='+', required=False, help='Custom list of models to run. If omitted, all models are run.')
    parser.add_argument('--fit', type=int, required=False, default=1, help='Set to 0 to load existing trained models instead of fitting anew.')

    args = parser.parse_args()

    main(
        problem=args.problem,
        evaluation=args.evaluation,
        save=args.save,
        splits=args.splits,
        indexes=args.indexes,
        models_name=args.modelsname,
        fit=bool(args.fit),
    )


