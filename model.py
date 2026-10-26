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


def train_mlp(X, y, hidden, lr, wd, epochs=30, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(1 / hidden), hidden); b2 = np.zeros(1)
    P = [W1, b1, w2, b2]
    M = [np.zeros_like(p) for p in P]
    V = [np.zeros_like(p) for p in P]
    beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0

    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            ids = perm[i:i + batch]
            Xb, yb = X[ids], y[ids]
            # forward
            H = np.maximum(0, Xb @ W1 + b1)
            err = H @ w2 + b2 - yb
            # backward (MSE + L2 weight decay)
            dp = 2 * err / len(yb)
            dH = np.outer(dp, w2) * (H > 0)
            grads = [Xb.T @ dH + wd * W1, dH.sum(0),
                     H.T @ dp + wd * w2, np.array([dp.sum()])]
            # Adam
            t += 1
            for p, g, m, v in zip(P, grads, M, V):
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * g**2
                p -= lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
    return P


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