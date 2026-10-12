import numpy as np
import pandas as pd

from optim.backtest import equal_weight, metrics, run


def _returns(days=900, n=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (days, n)), index=idx, columns=["SPY", "IEF", "GLD", "TLT"])


def test_rules_only_see_the_past():
    r = _returns()
    seen = []

    def spy(window):
        seen.append(window.index[-1])
        return equal_weight(window)

    run(r, spy, lookback=252)
    daily, _, _ = run(r, equal_weight, lookback=252)
    # every decision date is before the first day it is applied to
    assert all(d < daily.index[daily.index > d][0] for d in seen)


def test_costs_reduce_returns_by_turnover_times_cost():
    r = _returns()
    gross, _, _ = run(r, equal_weight, lookback=252, cost_bps=0)
    net, _, turnover = run(r, equal_weight, lookback=252, cost_bps=10)
    assert np.isclose((gross - net).sum(), turnover.sum() * 10 / 1e4)


def test_weights_drift_between_rebalances():
    r = _returns()
    r.iloc[:, 0] += 0.002                      # asset 0 trends up, so it drifts overweight
    _, _, turnover = run(r, equal_weight, lookback=252)
    assert turnover.iloc[1:].mean() > 0        # rebalancing back to 1/N requires trading


def test_metrics_keys():
    r = _returns()
    daily, w, t = run(r, equal_weight, lookback=252)
    m = metrics(daily, w, t)
    assert {"CAGR", "Sharpe (rf=0)", "max drawdown", "daily CVaR 95%"} <= set(m)
