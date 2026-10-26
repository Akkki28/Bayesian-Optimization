import numpy as np
from sklearn.datasets import fetch_california_housing 

def load_data(n_total=8000, seed=42):
    data = fetch_california_housing()
    return split_and_standardize(data.data, data.target, n_total, seed)


def split_and_standardize(X, y, n_total, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:n_total]
    X, y = X[idx], y[idx]
    a, b = n_total // 2, (3 * n_total) // 4
    Xtr, Xva, Xte = X[:a], X[a:b], X[b:]
    ytr, yva, yte = y[:a], y[a:b], y[b:]

    mx, sx = Xtr.mean(0), Xtr.std(0)
    my, sy = ytr.mean(), ytr.std()
    Xtr, Xva, Xte = [(A - mx) / sx for A in (Xtr, Xva, Xte)]
    ytr, yva, yte = [(v - my) / sy for v in (ytr, yva, yte)]
    return (Xtr, ytr), (Xva, yva), (Xte, yte)


def mse(P, X, y):
    W1, b1, w2, b2 = P
    return np.mean((np.maximum(0, X @ W1 + b1) @ w2 + b2 - y) ** 2)


def decode(u):
    lr = 10 ** (-4 + 3 * u[0])                  # 1e-4 .. 1e-1
    wd = 10 ** (-6 + 5 * u[1])                  # 1e-6 .. 1e-1
    hidden = int(round(2 ** (2 + 6 * u[2])))    # 4 .. 256
    return lr, wd, hidden


def make_objective(train, val, noisy=False):
    counter = {"n": 0}

    def f(u):
        lr, wd, hidden = decode(u)
        seed = counter["n"] if noisy else 0
        counter["n"] += 1
        P = train_mlp(*train, hidden=hidden, lr=lr, wd=wd, seed=seed)
        v = mse(P, *val)
        return np.log(v) if np.isfinite(v) else np.log(10.0)

    return f