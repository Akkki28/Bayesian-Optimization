import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist



def matern52(X1, X2, ls, sf2):
    """KERNEL FUNCTION"""
    r = cdist(X1 / ls, X2 / ls)
    s = np.sqrt(5) * r
    return sf2 * (1 + s + s**2 / 3) * np.exp(-s)


def unpack(theta, d):
    return np.exp(theta[:d]), np.exp(theta[d]), np.exp(theta[d + 1])


def build_gp(X, ys, theta, mu_y, sd_y):
    d = X.shape[1]
    ls, sf2, sn2 = unpack(theta, d)
    K = matern52(X, X, ls, sf2) + (sn2 + 1e-8) * np.eye(len(X))
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), ys)  # alpha = K^{-1} y
    return dict(X=X, L=L, alpha=alpha, ls=ls, sf2=sf2, sn2=sn2,
                mu_y=mu_y, sd_y=sd_y)


def predict(gp, Xs):
    Ks = matern52(Xs, gp["X"], gp["ls"], gp["sf2"])     
    mu = Ks @ gp["alpha"]
    v = solve_triangular(gp["L"], Ks.T, lower=True)       
    var = gp["sf2"] - np.sum(v**2, axis=0)                
    sd = np.sqrt(np.maximum(var, 1e-12))
    return mu * gp["sd_y"] + gp["mu_y"], sd * gp["sd_y"] 



def nll(theta, X, ys):
    """Negative log marginal likelihood"""
    
    d = X.shape[1]
    ls, sf2, sn2 = unpack(theta, d)
    K = matern52(X, X, ls, sf2) + (sn2 + 1e-8) * np.eye(len(X))
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = cho_solve((L, True), ys)
    return (0.5 * ys @ alpha
            + np.sum(np.log(np.diag(L)))     
            + 0.5 * len(ys) * np.log(2 * np.pi))


def fit_gp(X, y, n_restarts=8, rng=None):
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]
    mu_y, sd_y = y.mean(), y.std() + 1e-9
    ys = (y - mu_y) / sd_y

    bounds = ([(np.log(0.05), np.log(5.0))] * d
              + [(np.log(0.05), np.log(5.0)), (np.log(1e-6), np.log(1.0))])

    best = None
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(nll, x0, args=(X, ys), bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    gp = build_gp(X, ys, best.x, mu_y, sd_y)
    gp["nll"] = best.fun
    return gp


def lcb(x, gp, kappa):
    """LOWER CONFIDENCE BOUND"""
    mu, sd = predict(gp, np.atleast_2d(x))
    return (mu - kappa * sd)[0]


def kappa_schedule(t, d, delta=0.1):
    return np.sqrt(2 * np.log(t ** (d / 2 + 2) * np.pi**2 / (3 * delta)))


def propose(gp, kappa, d, rng, n_scan=3000, n_refine=5):
    cand = rng.random((n_scan, d))
    mu, sd = predict(gp, cand)
    starts = cand[np.argsort(mu - kappa * sd)[:n_refine]]

    best_x, best_v = None, np.inf
    for x0 in starts:
        res = minimize(lambda x: lcb(x, gp, kappa), x0,
                       bounds=[(0, 1)] * d, method="L-BFGS-B")
        if res.fun < best_v:
            best_x, best_v = res.x, res.fun
    return best_x


def bayes_opt(f, d, n_init=6, n_iter=24, kappa=None, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    X = rng.random((n_init, d))
    y = np.array([f(x) for x in X])

    for t in range(1, n_iter + 1):
        gp = fit_gp(X, y, rng=rng)                         # 1. update surrogate
        k = kappa if kappa is not None else kappa_schedule(t, d)
        x_next = propose(gp, k, d, rng)                    # 2. optimize acquisition
        X = np.vstack([X, x_next])
        y = np.append(y, f(x_next))                        # 3. query black box
        if verbose:
            print(f"iter {t:2d} | kappa={k:.2f} | new {y[-1]:.3f} | best {y.min():.3f}")

    return X, y, fit_gp(X, y, rng=rng)


def random_search(f, d, n, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    return X, np.array([f(x) for x in X])


def loo_zscores(gp):
    Kinv = cho_solve((gp["L"], True), np.eye(len(gp["X"])))
    return gp["alpha"] / np.sqrt(np.diag(Kinv))