"""Bayesian optimization: Gaussian-process regression + expected improvement.

Everything works in the unit hypercube [0, 1]^d (see config.Space).  The
pieces are deliberately small and explicit:

* ``initial_design``  - space-filling (Latin hypercube + greedy maximin) points
                        that satisfy the constraints.
* ``fit_gp``          - scikit-learn GP with an anisotropic Matern-5/2 kernel
                        (one length scale per parameter = automatic relevance)
                        plus a white-noise term (absorbs mesh noise).
* ``propose``         - maximizes expected improvement over many feasible
                        random + local candidates, then polishes the best ones
                        with L-BFGS-B.  Batches (parallel workers) use the
                        "constant liar" trick.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .config import Space


# --------------------------------------------------------------------------
# design of experiments
# --------------------------------------------------------------------------
def feasible_samples(space: Space, n: int, rng: np.random.Generator, max_tries: int = 200) -> np.ndarray:
    """n feasible uniform random points in the unit cube"""
    out = []
    got = 0
    for _ in range(max_tries):
        U = rng.random((max(4 * n, 256), space.dim))
        U = U[space.feasible_unit(U)]
        out.append(U)
        got += len(U)
        if got >= n:
            break
    if got == 0:
        raise RuntimeError("no feasible point found - check the constraints in the YAML file")
    return np.vstack(out)[:n]


def initial_design(space: Space, n: int, rng: np.random.Generator,
                   existing: np.ndarray | None = None) -> np.ndarray:
    """Latin hypercube pool -> keep feasible -> greedy maximin selection."""
    try:
        sampler = qmc.LatinHypercube(d=space.dim, rng=rng)      # SciPy >= 1.15
    except TypeError:
        sampler = qmc.LatinHypercube(d=space.dim, seed=rng)
    pool = sampler.random(max(60 * n, 2000))
    pool = pool[space.feasible_unit(pool)]
    if len(pool) < n:
        pool = np.vstack([pool, feasible_samples(space, 20 * n, rng)])
    chosen = [] if existing is None or len(existing) == 0 else list(np.atleast_2d(existing))
    dmin = np.full(len(pool), np.inf)
    for c in chosen:
        dmin = np.minimum(dmin, np.linalg.norm(pool - c, axis=1))
    picks = []
    for _ in range(n):
        if not chosen and not picks:
            idx = int(rng.integers(len(pool)))
        else:
            idx = int(np.argmax(dmin))
        picks.append(pool[idx])
        dmin = np.minimum(dmin, np.linalg.norm(pool - pool[idx], axis=1))
    return np.array(picks)


# --------------------------------------------------------------------------
# Gaussian process
# --------------------------------------------------------------------------
def make_gp(dim: int, seed: int = 0, noise: float = 1e-3) -> GaussianProcessRegressor:
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(dim, 0.3), length_scale_bounds=(1e-2, 1e1), nu=2.5)
              + WhiteKernel(noise, (1e-6, 1e-1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=8, random_state=seed)


def fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GaussianProcessRegressor:
    gp = make_gp(X.shape[1], seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def refit_fixed(gp: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """same hyper-parameters, new data (used for constant-liar batches)"""
    g = GaussianProcessRegressor(kernel=gp.kernel_, normalize_y=True, optimizer=None)
    g.fit(X, y)
    return g


class GPModel:
    """GP + its training data (so a batch can add constant-liar points)"""

    def __init__(self, gp: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray):
        self.gp, self.X, self.y = gp, np.atleast_2d(X), np.asarray(y, dtype=float)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "GPModel":
        return cls(fit_gp(X, y, seed), X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.gp.predict(np.atleast_2d(X), return_std=True)
        return mu, np.maximum(sd, 1e-9)

    def with_lies(self, X_lie: np.ndarray, y_lie: np.ndarray) -> "GPModel":
        X = np.vstack([self.X, X_lie])
        y = np.concatenate([self.y, y_lie])
        return GPModel(refit_fixed(self.gp, X, y), X, y)


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------
def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    imp = best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def propose(model, space: Space, X_obs: np.ndarray, y_best: float, n_points: int,
            rng: np.random.Generator, n_candidates: int = 20000, xi: float = 0.01,
            min_dist: float = 1e-3, penalty_radius: float = 0.05) -> np.ndarray:
    """Return n_points new unit-cube points maximizing EI.

    Batches (n_points > 1) use the constant-liar trick, plus a local penalty
    that suppresses EI within ~penalty_radius of points already picked in
    this batch.
    """
    lo = np.zeros(space.dim)
    hi = np.ones(space.dim)

    picks: list[np.ndarray] = []
    cur = model
    X_all = np.atleast_2d(X_obs) if len(X_obs) else np.zeros((0, space.dim))
    for _ in range(n_points):
        cand = _candidates(space, X_all, rng, n_candidates, lo, hi)
        if len(X_all):
            d = np.min(np.linalg.norm(cand[:, None, :] - X_all[None, :, :], axis=2), axis=1)
            cand = cand[d > min_dist]
        if len(cand) == 0:
            raise RuntimeError("no feasible candidate left in the search region")
        pen = _penalty(picks, penalty_radius)
        mu, sd = cur.predict(cand)
        ei = expected_improvement(mu, sd, y_best, xi) * pen(cand)
        # polish the top few with L-BFGS-B
        order = np.argsort(-ei)[:5]
        best_x, best_ei = cand[order[0]], ei[order[0]]
        for i in order:
            x, v = _polish(cur, space, cand[i], y_best, xi, lo, hi, pen)
            if v > best_ei:
                best_x, best_ei = x, v
        picks.append(best_x)
        # constant liar: pretend the point returned the current best value
        X_all = np.vstack([X_all, best_x])
        cur = cur.with_lies(np.atleast_2d(best_x), np.array([y_best]))
    return np.array(picks)


def _candidates(space, X_obs, rng, n, lo, hi):
    U = lo + rng.random((n, space.dim)) * (hi - lo)
    parts = [U]
    if len(X_obs):
        # local perturbations around the observed points (cheap exploitation)
        base = X_obs[rng.integers(len(X_obs), size=n // 2)]
        P = base + rng.normal(scale=0.05, size=base.shape)
        parts.append(np.clip(P, lo, hi))
    C = np.vstack(parts)
    return C[space.feasible_unit(C)]


def _penalty(picks, r):
    P = np.array(picks) if picks else None

    def f(X):
        X = np.atleast_2d(X)
        if P is None:
            return np.ones(len(X))
        d2 = np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=2)
        return np.prod(1.0 - np.exp(-d2 / (2 * r * r)), axis=1)
    return f


def _polish(model, space, x0, y_best, xi, lo, hi, pen):
    def neg_ei(x):
        mu, sd = model.predict(x[None, :])
        return -float(expected_improvement(mu, sd, y_best, xi)[0] * pen(x)[0])

    try:
        r = minimize(neg_ei, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options={"maxiter": 50})
        x = np.clip(r.x, lo, hi)
        if space.feasible(space.from_unit(x)):
            return x, -r.fun
    except Exception:
        pass
    return x0, -neg_ei(x0)
