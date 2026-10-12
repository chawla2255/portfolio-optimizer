import numpy as np
import pytest

from optim.optimizers import cvar, efficient_frontier, max_sharpe, min_cvar, min_variance, target_return


def test_min_variance_two_uncorrelated_assets_weights_inverse_variance():
    cov = np.diag([0.04, 0.01])
    w = min_variance(cov)
    np.testing.assert_allclose(w, np.array([0.01, 0.04]) / 0.05, atol=1e-6)


def test_weights_are_fully_invested_long_only_and_capped():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T / 6 + np.eye(6) * 0.01
    w = min_variance(cov, w_max=0.3)
    assert np.isclose(w.sum(), 1) and (w >= -1e-12).all() and (w <= 0.3 + 1e-9).all()


def test_max_sharpe_matches_the_analytic_tangency_portfolio():
    mu = np.array([0.08, 0.05])
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    w = max_sharpe(mu, cov)
    analytic = np.linalg.solve(cov, mu)
    analytic /= analytic.sum()          # interior solution, so no long-only constraint binds
    np.testing.assert_allclose(w, analytic, atol=1e-4)


def test_frontier_is_increasing_in_both_risk_and_return():
    mu = np.array([0.03, 0.06, 0.10])
    cov = np.diag([0.002, 0.02, 0.06])
    pts = efficient_frontier(mu, cov, n_points=10)
    vols, rets = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    assert (np.diff(rets) > 0).all() and (np.diff(vols) >= -1e-9).all()


def test_target_return_hits_the_target():
    mu = np.array([0.03, 0.06, 0.10])
    w = target_return(mu, np.diag([0.002, 0.02, 0.06]), 0.07)
    assert mu @ w == pytest.approx(0.07, abs=1e-8)


def test_cvar_is_the_mean_of_the_worst_tail():
    r = np.arange(-10, 10) / 100.0          # 20 days from -10% to +9%
    assert cvar(r, alpha=0.9) == pytest.approx(0.095)   # worst 2 days: -10%, -9%


def test_min_cvar_beats_every_point_on_a_weight_grid():
    rng = np.random.default_rng(1)
    days = 500
    crash = rng.random(days) < 0.03
    a = rng.normal(0.0005, 0.01, days) - 0.05 * crash          # risky, crash-prone
    b = rng.normal(0.0002, 0.004, days) + 0.01 * crash          # defensive, rallies in crashes
    scen = np.column_stack([a, b])
    w = min_cvar(scen, alpha=0.95)
    best = min(cvar(scen @ np.array([x, 1 - x])) for x in np.linspace(0, 1, 1001))
    assert cvar(scen @ w) == pytest.approx(best, rel=1e-3)
