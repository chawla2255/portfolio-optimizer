"""Charts for the README: frontier, equity curves, drawdowns and weights over time."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optim.backtest import RULES, run
from optim.data import ASSETS, daily_returns, load_prices
from optim.estimators import ledoit_wolf
from optim.optimizers import cvar, efficient_frontier, max_sharpe, min_cvar, min_variance

DOCS = Path(__file__).resolve().parent.parent / "docs"
plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"Equal weight (1/N)": "#8A9BAD", "60/40 (SPY/IEF)": "#495A6B", "Inverse volatility": "#C0717A",
          "Min variance (sample)": "#9DD9D2", "Min variance (Ledoit-Wolf)": "#2EC4B6",
          "Max Sharpe (Ledoit-Wolf)": "#F5A623", "Min CVaR 95%": "#6A4C93"}


def main():
    r = daily_returns(load_prices())

    # 1. Efficient frontier on the last 3 years, with the assets and the optimal portfolios.
    win = r.iloc[-756:]
    mu, cov = win.mean().to_numpy() * 252, ledoit_wolf(win.to_numpy())[0] * 252
    pts = efficient_frontier(mu, cov, n_points=40)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([p[0] * 100 for p in pts], [p[1] * 100 for p in pts], color="#F5A623", lw=2, label="efficient frontier (long-only)")
    vols = np.sqrt(np.diag(cov))
    ax.scatter(vols * 100, mu * 100, color="#495A6B", s=14)
    for name, v, m in zip(win.columns, vols, mu):
        ax.annotate(name, (v * 100, m * 100), textcoords="offset points", xytext=(4, 2), fontsize=7)
    for label, w, c in (("min variance", min_variance(cov), "#2EC4B6"), ("max Sharpe", max_sharpe(mu, cov), "#F5A623"),
                        ("min CVaR 95%", min_cvar(win.to_numpy()), "#6A4C93"), ("1/N", np.full(len(mu), 1 / len(mu)), "#8A9BAD")):
        ax.scatter(np.sqrt(w @ cov @ w) * 100, mu @ w * 100, s=60, marker="D", color=c, edgecolor="white", zorder=5, label=label)
    ax.set_xlabel("annual volatility %"); ax.set_ylabel("annual expected return % (trailing mean)")
    ax.set_title(f"Efficient frontier, {win.index[0].date()} to {win.index[-1].date()} (Ledoit-Wolf covariance)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(DOCS / "frontier.png"); plt.close(fig)

    # 2. Equity curves and drawdowns of every rule.
    results = {name: run(r, rule) for name, rule in RULES.items()}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, (daily, _, _) in results.items():
        wealth = (1 + daily).cumprod()
        a1.plot(wealth, color=COLORS[name], lw=1.4 if "Sharpe" in name or "CVaR" in name or "1/N" in name else 1.0, label=name)
        a2.plot((wealth / wealth.cummax() - 1) * 100, color=COLORS[name], lw=0.9)
    a1.set_ylabel("growth of $1, after costs"); a1.legend(frameon=False, fontsize=7, ncol=2)
    a1.set_title("Monthly walk-forward, 3-year lookback, 5 bp costs")
    a2.set_ylabel("drawdown %")
    fig.tight_layout(); fig.savefig(DOCS / "equity_curves.png"); plt.close(fig)

    # 3. Weights through time: min CVaR vs max Sharpe.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    cmap = plt.get_cmap("tab20")
    for ax, name in zip(axes, ("Min CVaR 95%", "Max Sharpe (Ledoit-Wolf)")):
        w = results[name][1]
        ax.stackplot(w.index, w.T.to_numpy(), labels=w.columns, colors=[cmap(i) for i in range(w.shape[1])], alpha=0.95)
        ax.set_title(f"{name}: target weights"); ax.set_ylim(0, 1)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(DOCS / "weights.png"); plt.close(fig)

    # Numbers quoted in the README.
    for name in ("Min variance (Ledoit-Wolf)", "Min CVaR 95%", "60/40 (SPY/IEF)", "Equal weight (1/N)"):
        d = results[name][0]
        wealth = (1 + d).cumprod()
        dd = wealth / wealth.cummax() - 1
        print(name, "worst drawdown trough:", dd.idxmin().date(), round(dd.min() * 100, 1), "%",
              "| 2022 return:", round(((1 + d.loc['2022']).prod() - 1) * 100, 1), "%")
    w = results["Min CVaR 95%"][1]
    print("avg min-CVaR weights:", (w.mean() * 100).round(1).sort_values(ascending=False).head(5).to_dict())
    w = results["Max Sharpe (Ledoit-Wolf)"][1]
    print("avg max-Sharpe weights:", (w.mean() * 100).round(1).sort_values(ascending=False).head(5).to_dict())


if __name__ == "__main__":
    main()
