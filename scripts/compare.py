"""Run every allocation rule walk-forward and write the comparison table to docs/."""
from pathlib import Path

import pandas as pd

from optim.backtest import RULES, metrics, run
from optim.data import daily_returns, load_prices

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main():
    r = daily_returns(load_prices())
    rows, curves = {}, {}
    for name, rule in RULES.items():
        daily, weights, turnover = run(r, rule)
        rows[name] = metrics(daily, weights, turnover)
        curves[name] = daily
        weights.to_csv(DOCS / f"weights_{name.split(' (')[0].lower().replace(' ', '_').replace('/', '')}.csv")
    table = pd.DataFrame(rows).T
    table.to_csv(DOCS / "comparison.csv", float_format="%.4f")
    first = min(c.index[0] for c in curves.values())
    print("evaluation period:", first.date(), "to", r.index[-1].date())
    fmt = table.copy()
    for c in ("CAGR", "volatility", "max drawdown", "daily CVaR 95%", "annual turnover", "max single weight"):
        fmt[c] = (fmt[c] * 100).round(2)
    print(fmt.round(2).to_markdown())


if __name__ == "__main__":
    main()
