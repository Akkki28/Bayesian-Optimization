import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

from GP import (bayes_opt, random_search, fit_gp, predict, matern52,
                   loo_zscores)
from model import load_data, make_objective, decode, train_mlp, mse

D = 3 


def plot_prior_samples():
    xs = np.linspace(0, 1, 200)[:, None]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, ls in zip(axes, [0.05, 0.2, 0.6]):
        K = matern52(xs, xs, np.array([ls]), 1.0) + 1e-6 * np.eye(len(xs))
        L = cholesky(K, lower=True)
        ax.plot(xs, L @ rng.standard_normal((len(xs), 5)))
        ax.set_title(f"lengthscale = {ls}")
    plt.tight_layout()
    plt.savefig("gp_prior_samples.png", dpi=120)
    plt.close()


def plot_toy_gp():
    f = lambda x: np.sin(6 * x) + 0.5 * x
    rng = np.random.default_rng(1)
    X = rng.random((6, 1))
    y = f(X[:, 0]) + 0.05 * rng.standard_normal(6)
    gp = fit_gp(X, y)
    grid = np.linspace(0, 1, 300)[:, None]
    mu, sd = predict(gp, grid)

    plt.figure(figsize=(7, 4))
    plt.plot(grid, f(grid[:, 0]), "k--", label="true f")
    plt.plot(grid, mu, label="GP mean")
    plt.fill_between(grid[:, 0], mu - 2 * sd, mu + 2 * sd, alpha=0.2, label="+-2 sigma")
    plt.scatter(X, y, c="r", zorder=3, label="observations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gp_toy_fit.png", dpi=120)
    plt.close()
    print("toy GP learned ls, sf2, sn2:", gp["ls"], gp["sf2"], gp["sn2"])


def single_run(train, val, test, kappa, n_iter):
    f = make_objective(train, val)
    X_bo, y_bo, gp = bayes_opt(f, D, n_iter=n_iter, kappa=kappa)
    _, y_rs = random_search(f, D, n=len(y_bo))

    plt.figure(figsize=(7, 4))
    plt.plot(np.minimum.accumulate(y_bo), label=f"BO (GP + LCB, kappa={kappa})")
    plt.plot(np.minimum.accumulate(y_rs), label="Random search")
    plt.xlabel("evaluations"); plt.ylabel("best log val-MSE so far")
    plt.legend(); plt.tight_layout()
    plt.savefig("bo_vs_random.png", dpi=120)
    plt.close()

    lr, wd, hidden = decode(X_bo[np.argmin(y_bo)])
    P = train_mlp(*train, hidden=hidden, lr=lr, wd=wd)
    print(f"\nbest: lr={lr:.2e} wd={wd:.2e} hidden={hidden}")
    print(f"val MSE (std. units): {mse(P, *val):.4f} | test MSE: {mse(P, *test):.4f}")
    print("learned lengthscales (lr, wd, hidden):", np.round(gp["ls"], 3))
    print(f"learned signal var {gp['sf2']:.3f}, noise var {gp['sn2']:.2e}")

    z = loo_zscores(gp)
    print(f"LOO calibration: fraction within +-2 = {np.mean(np.abs(z) < 2):.2f} (target ~0.95)")


def sweep_kappa(train, val, n_seeds, n_iter):
    f = make_objective(train, val)
    settings = {"kappa=0": 0.0, "kappa=1": 1.0, "kappa=2": 2.0,
                "kappa=4": 4.0, "schedule": None}
    plt.figure(figsize=(8, 5))
    for name, kappa in settings.items():
        curves = []
        for s in range(n_seeds):
            _, y, _ = bayes_opt(f, D, n_iter=n_iter, kappa=kappa, seed=s, verbose=False)
            curves.append(np.minimum.accumulate(y))
        curves = np.array(curves)
        med = np.median(curves, 0)
        lo, hi = np.percentile(curves, [25, 75], axis=0)
        plt.plot(med, label=name)
        plt.fill_between(range(len(med)), lo, hi, alpha=0.15)
        print(f"{name:10s} final median best: {med[-1]:.4f}")
    plt.xlabel("evaluations"); plt.ylabel("best log val-MSE so far")
    plt.legend(); plt.tight_layout()
    plt.savefig("kappa_sweep.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-kappa", action="store_true")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    plot_prior_samples()
    plot_toy_gp()

    train, val, test = load_data()
    if args.sweep_kappa:
        sweep_kappa(train, val, args.seeds, args.iters)
    else:
        single_run(train, val, test, args.kappa, args.iters)