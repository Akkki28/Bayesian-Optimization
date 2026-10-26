import argparse

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from GP import bayes_opt, loo_zscores
from model import load_data

D = 3


def model_specs():
    return {
        "XGBoost": {
            "default": lambda: XGBRegressor(objective="reg:squarederror"),
            "decode": lambda u: {
                "n_estimators": int(round(50 + 250 * u[0])),
                "max_depth": int(round(2 + 8 * u[1])),
                "learning_rate": 10 ** (-2 + 1.5 * u[2]),
            },
        },
        "KNN": {
            "default": lambda: KNeighborsRegressor(),
            "decode": lambda u: {
                "n_neighbors": int(round(2 + 48 * u[0])),
                "weights": "distance" if u[1] >= 0.5 else "uniform",
                "p": int(round(1 + u[2])),
            },
        },
        "SVM": {
            "default": lambda: SVR(),
            "decode": lambda u: {
                "C": 10 ** (-2 + 4 * u[0]),
                "gamma": 10 ** (-4 + 3 * u[1]),
                "epsilon": 10 ** (-3 + 0.7 * u[2]),
            },
        },
    }


def tune_model(name, spec, train, val, test, kappa, n_iter):
    def estimator_mse(estimator, data):
        X, y = data
        return mean_squared_error(y, estimator.predict(X))

    def objective(u):
        params = spec["decode"](u)
        estimator = spec["default"]()
        estimator.set_params(**params)
        estimator.fit(*train)
        value = estimator_mse(estimator, val)
        return np.log(value) if np.isfinite(value) else np.log(10.0)

    X_bo, y_bo, gp = bayes_opt(
        objective, D, n_iter=n_iter, kappa=kappa, verbose=False
    )
    best_u = X_bo[np.argmin(y_bo)]
    best_params = spec["decode"](best_u)

    default_model = spec["default"]()
    default_model.fit(*train)
    tuned_model = spec["default"]()
    tuned_model.set_params(**best_params)
    tuned_model.fit(*train)

    default_val = estimator_mse(default_model, val)
    tuned_val = estimator_mse(tuned_model, val)
    return {
        "params": best_params,
        "default_val": default_val,
        "tuned_val": tuned_val,
        "default_test": estimator_mse(default_model, test),
        "tuned_test": estimator_mse(tuned_model, test),
        "gp": gp,
    }


def print_report(name, result):
    print(f"\n{name}")
    print("-" * len(name))
    print("BO parameters:", result["params"])
    print(f"default validation MSE: {result['default_val']:.4f}")
    print(f"BO validation MSE:      {result['tuned_val']:.4f}")
    print(
        "validation improvement (default - BO): "
        f"{result['default_val'] - result['tuned_val']:.4f}"
    )
    print(f"default test MSE:      {result['default_test']:.4f}")
    print(f"BO test MSE:           {result['tuned_test']:.4f}")
    print(
        "test improvement (default - BO): "
        f"{result['default_test'] - result['tuned_test']:.4f}"
    )
    gp = result["gp"]
    print("BO GP learned lengthscales:", np.round(gp["ls"], 3))
    print(f"BO GP LOO fraction within +-2: {np.mean(np.abs(loo_zscores(gp)) < 2):.2f}")


def main(args):
    train, val, test_data = load_data()
    for name, spec in model_specs().items():
        result = tune_model(name, spec, train, val, test_data, args.kappa, args.iters)
        print_report(name, result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=24)
    args = ap.parse_args()
    main(args)
