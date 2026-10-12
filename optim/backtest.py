"""Monthly walk-forward evaluation of allocation rules.

At each month-end the rule sees only the trailing `lookback` days of returns and sets target
weights. The portfolio trades to them at that close (paying `cost_bps` per dollar traded) and
then holds, letting weights drift with prices, until the next month-end.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .estimators import ledoit_wolf, sample_cov
from .optimizers import cvar, max_sharpe, min_cvar, min_variance

TRADING_DAYS = 252
Rule = Callable[[pd.DataFrame], np.ndarray]


def equal_weight(window: pd.DataFrame) -> np.ndarray:
    return np.full(window.shape[1], 1.0 / window.shape[1])


def inverse_vol(window: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / window.std().to_numpy()
    return inv / inv.sum()


def sixty_forty(window: pd.DataFrame) -> np.ndarray:
    w = np.zeros(window.shape[1])
    w[list(window.columns).index("SPY")] = 0.6
    w[list(window.columns).index("IEF")] = 0.4
    return w


def min_var_sample(window: pd.DataFrame) -> np.ndarray:
    return min_variance(sample_cov(window.to_numpy()))


def min_var_shrunk(window: pd.DataFrame) -> np.ndarray:
    return min_variance(ledoit_wolf(window.to_numpy())[0])


def max_sharpe_shrunk(window: pd.DataFrame) -> np.ndarray:
    return max_sharpe(window.mean().to_numpy(), ledoit_wolf(window.to_numpy())[0])


def min_cvar_95(window: pd.DataFrame) -> np.ndarray:
    return min_cvar(window.to_numpy(), alpha=0.95)


RULES: dict[str, Rule] = {
    "Equal weight (1/N)": equal_weight,
    "60/40 (SPY/IEF)": sixty_forty,
    "Inverse volatility": inverse_vol,
    "Min variance (sample)": min_var_sample,
    "Min variance (Ledoit-Wolf)": min_var_shrunk,
    "Max Sharpe (Ledoit-Wolf)": max_sharpe_shrunk,
    "Min CVaR 95%": min_cvar_95,
}


def run(returns: pd.DataFrame, rule: Rule, lookback: int = 756, cost_bps: float = 5.0):
    """Daily net returns, the target weights at each rebalance, and turnover per rebalance."""
    dates = returns.index
    month_end = dates[np.r_[dates.month[1:] != dates.month[:-1], True]]
    rebalance = [d for d in month_end if dates.get_loc(d) >= lookback - 1][:-1]
    r = returns.to_numpy()
    w_held = np.zeros(returns.shape[1])
    daily, targets, turnover = {}, {}, {}
    for i, d in enumerate(rebalance):
        loc = dates.get_loc(d)
        target = rule(returns.iloc[loc - lookback + 1 : loc + 1])
        trade = np.abs(target - w_held).sum()
        targets[d], turnover[d] = target, trade
        end = dates.get_loc(rebalance[i + 1]) if i + 1 < len(rebalance) else len(dates) - 1
        w = target.copy()
        first = True
        for j in range(loc + 1, end + 1):
            port = float(w @ r[j])
            if first:
                port -= trade * cost_bps / 1e4
                first = False
            daily[dates[j]] = port
            w = w * (1 + r[j])
            w = w / w.sum()
        w_held = w
    return pd.Series(daily), pd.DataFrame(targets, index=returns.columns).T, pd.Series(turnover)


def metrics(daily: pd.Series, weights: pd.DataFrame, turnover: pd.Series) -> dict:
    ann_ret = (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1
    ann_vol = daily.std() * np.sqrt(TRADING_DAYS)
    wealth = (1 + daily).cumprod()
    return {
        "CAGR": ann_ret,
        "volatility": ann_vol,
        "Sharpe (rf=0)": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS),
        "max drawdown": (wealth / wealth.cummax() - 1).min(),
        "daily CVaR 95%": cvar(daily.to_numpy(), 0.95),
        "annual turnover": turnover.mean() * 12,
        "avg holdings >1%": (weights > 0.01).sum(axis=1).mean(),
        "max single weight": weights.max(axis=1).mean(),
    }
