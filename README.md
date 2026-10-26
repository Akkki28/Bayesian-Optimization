# Bayesian Optimization for Hyperparameter Tuning

This project demonstrates Bayesian optimization for selecting hyperparameters
of several regression models on the
[California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
It uses a Gaussian process surrogate model and a lower-confidence-bound
acquisition function to minimize validation mean squared error (MSE).

The experiment compares each model's default configuration with a
Bayesian-optimized configuration:

- **XGBoost** (`XGBRegressor`)
- **K-nearest neighbors** (`KNeighborsRegressor`)
- **Support vector regression** (`SVR`)

For every model, the script reports validation and test MSE for both
configurations, the improvement from tuning, and diagnostics from the final
Gaussian process.

## How it works

1. `model.py` downloads the California Housing data through scikit-learn.
2. A reproducible permutation (seed `42`) selects 8,000 examples.
3. The selected data is split into:
   - 4,000 training examples
   - 2,000 validation examples
   - 2,000 test examples
4. Features and targets are standardized using statistics computed only from
   the training split.
5. Each model's hyperparameters are represented by a three-dimensional point
   in the unit cube `[0, 1]^3`.
6. `GP.py` fits a Matérn 5/2 Gaussian process to the observed validation
   losses and proposes new points with a lower-confidence-bound acquisition
   function.
7. The best observed hyperparameters are used to fit a final model, which is
   evaluated on the validation and test splits.

The objective optimizes `log(MSE)` for numerical stability, while the reported
metrics are in MSE units on the standardized target.

## Hyperparameter search spaces

The three normalized optimization variables are decoded into model-specific
parameters:

| Model | Parameters searched |
| --- | --- |
| XGBoost | `n_estimators`: 50–300; `max_depth`: 2–10; `learning_rate`: 0.01–approximately 0.316 |
| KNN | `n_neighbors`: 2–50; `weights`: `uniform` or `distance`; `p`: 1–2 |
| SVM | `C`: 0.01–100; `gamma`: 0.0001–0.1; `epsilon`: 0.001–approximately 0.005 |

## Requirements

Python 3.9 or newer is recommended. Install the required packages with:

```bash
python -m pip install numpy scipy scikit-learn xgboost
```

## Usage

Run the default experiment from the repository root:

```bash
python main.py
```

By default, each model receives six initial random evaluations followed by
24 Bayesian optimization iterations. Adjust the optimization trade-off and
number of iterations with:

```bash
python main.py --kappa 2.0 --iters 24
```

### Command-line options

| Option | Default | Description |
| --- | ---: | --- |
| `--kappa` | `2.0` | Exploration weight in the lower-confidence-bound acquisition function. Larger values encourage more exploration. |
| `--iters` | `24` | Number of Bayesian optimization iterations per model. |

The command prints a separate report for each model, including:

- best Bayesian-optimized parameters;
- default and optimized validation MSE;
- validation improvement (`default - optimized`);
- default and optimized test MSE;
- test improvement (`default - optimized`);
- learned Gaussian-process lengthscales; and
- the fraction of leave-one-out z-scores within `+/-2`.

