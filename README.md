# Interpretable ML Benchmarking Framework

This repository contains the complete codebase used for a large-scale study on interpretable machine learning models.  
It supports both **classification** and **regression** and provides a unified pipeline to preprocess datasets, train models, evaluate them under different strategies and analyze their performance and structural complexity.

The project includes:

- End-to-end dataset preprocessing (NaN handling, encoding, scaling, IS/OOS splitting)
- A wide suite of interpretable models:
  - Decision trees
  - Logistic regression
  - KNN
  - Naive Bayes
  - Explainable Boosting Machines (EBM)
  - IGANN
  - Symbolic Regression
  - GLM, Lasso, Polynomial Lasso
- Unified wrappers for training, loading/saving models and computing metrics
- Automatic hyperparameter optimization via Optuna
- Evaluation engine supporting:
  - In-Sample evaluation (k-fold)
  - Out-of-Sample evaluation
- Dataset complexity metrics (F1, N3, C1, L1, S3, T2, T3)
- Plotting and statistical testing utilities (correlations, ranking, Friedman/Nemenyi)

---

## Project Structure

```
.
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── main.py
    |
    ├── config
    │   ├── __init__.py
    │   └── config.py
    |
    ├── preprocessing
    │   ├── __init__.py
    │   ├── preprocessing.py
    │   └── splitting.py
    |
    ├── models
    │   ├── __init__.py
    │   ├── classification.py
    │   ├── regression.py
    │   ├── hyper_opt.py
    │   └── utils_model_io.py
    |
    ├── metrics
    │   ├── __init__.py
    │   └── dataset_metrics.py
    |
    ├── evaluation
    │   ├── __init__.py
    │   └── evaluation.py
    |
    └── analysis
        ├── __init__.py
        └── plotting_utils.py
```


---

## Installation

### 1. Clone the repository
```bash
git clone git@github.com:mattiabilla/CA-IML.git
```

### 2. Create a Python 3.11 virtual environment
Using virtualenvwrapper:
```bash
mkvirtualenv CA-IML -p /usr/bin/python3.11
```

or using venv:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install PMLB datasets
```bash
git clone https://github.com/EpistasisLab/pmlb
cd pmlb
pip install .
cd ..
```

### 4. Install project dependencies
```bash
pip install -r requirements.txt
```

---

## Running Experiments

Always execute the project as a Python package:

```bash
python -m src.main \
    --problem clf \
    --evaluation IS \
    --save 1 \
    --splits 0 1 2 3
```

### Main arguments
| Argument | Description |
|---------|-------------|
| `--problem` | `clf` or `regr` |
| `--evaluation` | `IS` (k-fold) or `OOS` (distribution-shift split) |
| `--indexes` | Optional dataset indices (otherwise all) |
| `--modelsname` | Optional list of models to evaluate |
| `--save` | 1 to save results, 0 otherwise |
| `--splits` | Which splits to evaluate |
| `--fit` | 1 to train new models, 0 to load them from disk |

Examples:

### Run classification, OOS scenario
```bash
python -m src.main --problem clf --evaluation OOS --save 1 --splits 0 1 2 3
```

### Run regression, training only on split 0
```bash
python -m src.main --problem regr --evaluation IS --splits 0
```

---

## Analysis

The `analysis/plotting_utils.py` module provides:

- Model performance boxplots
- Complexity/performance heatmaps
- Correlation analyses
- Model ranking (global + per-stratum)
- Friedman + Nemenyi statistical tests

Example usage:

```python
from src.analysis.plotting_utils import plot_model_boxplots
plot_model_boxplots(df, metrics=["Accuracy", "Fbeta"])
```

---

## Dataset Metadata & Metrics

The project computes multiple dataset-level complexity metrics (T2, T3, F1, N3, C1_clf, C1_regr, L1, S3).  
See `metrics/dataset_metrics.py` for full definitions.

---

## Extending the Project

To add a new model:

1. Implement a `*_performances` function in `models/classification.py` or `models/regression.py`.
2. Add hyperparameter ranges to `models/hyper_opt.py`.
3. Register the model name in `config/config.py`.
4. Run:
   ```bash
   python -m src.main --modelsname newmodel
   ```
