"""
Centralized utilities for saving and loading trained models.
Handles KNN special case (text file with n_neighbors) and binary serialization for all other models.
"""

import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


# ============================================================================
#                               UTILITY FUNCTIONS
# ============================================================================

def _get_filepath(load_params, model_name):
    """
    Construct the file path used to load a previously trained model.

    Parameters
    ----------
    load_params : dict
        Must contain:
            - 'ds_name'    : str, dataset name
            - 'evaluation' : str, evaluation scheme ("IS" or "OOS")
            - 'split'      : int, split index
    model_name : str
        Name of the model being loaded.

    Returns
    -------
    str
        Fully constructed file path.
    """
    return (
        "models/" +
        load_params["ds_name"] + "_" +
        load_params["evaluation"] + "_" +
        model_name + "_" +
        str(load_params["split"]) + ".pkl"
    )


# ============================================================================
#                               SAVE / LOAD MODELS
# ============================================================================

def save_model(model, model_file_name):
    """
    Save a trained model to disk.

    Notes
    -----
    • KNN models are stored as a simple text file containing only
      the number of neighbors (n_neighbors).
    • All other models are serialized using pickle.

    Parameters
    ----------
    model : estimator
        Fitted model instance.
    model_file_name : str
        Base filename (without extension) used when saving.
    """
    output_dir = Path(__file__).resolve().parent / "output/models"
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, (KNeighborsRegressor, KNeighborsClassifier)):
        # KNN: store only the number of neighbors
        file_path = output_dir / f"{model_file_name}.txt"
        with open(file_path, "w") as f:
            f.write(str(model.n_neighbors))
    else:
        file_path = output_dir / f"{model_file_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)


def load_model(model_name, load_params):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    model_name : str
        Identifier of the model to load.
    load_params : dict
        Dictionary containing dataset name, evaluation type and split index.

    Returns
    -------
    estimator
        The deserialized model instance.
    """
    file_path = _get_filepath(load_params, model_name)
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model
