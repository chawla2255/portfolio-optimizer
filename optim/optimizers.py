"""Long-only portfolio optimisers: minimum variance, maximum Sharpe, the efficient frontier
and minimum CVaR.

All portfolios are fully invested (weights sum to 1) and long-only, with an optional cap on
any single weight.

CVaR (conditional value at risk, or expected shortfall) at level alpha is the average loss on
the worst (1 - alpha) share of days. Rockafellar & Uryasev (2000) showed that minimising it
over historical scenarios r_1..r_T is a linear program:

    minimise    zeta + 1 / ((1 - alpha) T) * sum_t u_t
    subject to  u_t >= -r_t . w - zeta,   u_t >= 0,   sum(w) = 1,   0 <= w <= w_max

and at the optimum zeta is the portfolio's value at risk.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize


def _solve_qp(objective, n, w_max, extra_constraints=(), x0=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    res = minimize(objective, x0, method="SLSQP", bounds=[(0.0, w_max)] * n, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"optimiser failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def min_variance(cov: np.ndarray, w_max: float = 1.0) -> np.ndarray:
    """Fully invested, long-only portfolio with the lowest variance."""
    cov = np.asarray(cov, float)
    return _solve_qp(lambda w: w @ cov @ w, len(cov), w_max)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, w_max: float = 1.0) -> np.ndarray:
    """Long-only tangency portfolio: highest (expected return - rf) per unit of volatility."""
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    return _solve_qp(lambda w: -(mu @ w - rf) / np.sqrt(w @ cov @ w), len(mu), w_max)


def target_return(mu: np.ndarray, cov: np.ndarray, target: float, w_max: float = 1.0) -> np.ndarray:
    """Lowest-variance portfolio with expected return equal to `target`."""
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    con = {"type": "eq", "fun": lambda w: mu @ w - target}
    return _solve_qp(lambda w: w @ cov @ w, len(mu), w_max, extra_constraints=(con,))


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 30, w_max: float = 1.0):
    """(volatility, expected return, weights) along the long-only efficient frontier."""
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    w_min = min_variance(cov, w_max)
    lo = float(mu @ w_min)
    # With a weight cap the highest attainable return fills the best assets up to the cap.
    order = np.argsort(mu)[::-1]
    w_top, left = np.zeros_like(mu), 1.0
    for i in order:
        w_top[i] = min(w_max, left)
        left -= w_top[i]
        if left <= 0:
            break
    hi = float(mu @ w_top)
    points = []
    for target in np.linspace(lo, hi * 0.999, n_points):
        w = target_return(mu, cov, target, w_max)
        points.append((float(np.sqrt(w @ cov @ w)), float(mu @ w), w))
    return points


def cvar(portfolio_returns: np.ndarray, alpha: float = 0.95) -> float:
    """Historical CVaR as a positive loss: the average loss on the worst (1 - alpha) of days."""
    r = np.sort(np.asarray(portfolio_returns, float))
    k = max(1, int(np.ceil((1 - alpha) * len(r))))
    return float(-r[:k].mean())


def min_cvar(scenarios: np.ndarray, alpha: float = 0.95, w_max: float = 1.0, min_return: float | None = None) -> np.ndarray:
    """Long-only portfolio minimising historical CVaR, via the Rockafellar-Uryasev LP.

    `scenarios` is (days x assets). `min_return`, if given, is a floor on the mean daily return.
    Variables are stacked as [w (n), zeta (1), u (T)].
    """
    r = np.asarray(scenarios, float)
    t, n = r.shape
    c = np.r_[np.zeros(n), 1.0, np.full(t, 1.0 / ((1 - alpha) * t))]
    # -r_t . w - zeta - u_t <= 0
    a_ub = np.hstack([-r, -np.ones((t, 1)), -np.eye(t)])
    b_ub = np.zeros(t)
    if min_return is not None:
        a_ub = np.vstack([a_ub, np.r_[-r.mean(axis=0), 0.0, np.zeros(t)]])
        b_ub = np.r_[b_ub, -min_return]
    a_eq = np.r_[np.ones(n), 0.0, np.zeros(t)][None, :]
    bounds = [(0.0, w_max)] * n + [(None, None)] + [(0.0, None)] * t
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    w = np.clip(res.x[:n], 0.0, None)
    return w / w.sum()
