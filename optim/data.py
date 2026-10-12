"""ETF price data across asset classes, downloaded once and cached locally."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ASSETS = {
    "SPY": "US large-cap equity",
    "QQQ": "US tech / Nasdaq-100",
    "IWM": "US small-cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging-market equity",
    "TLT": "Long-term Treasuries",
    "IEF": "Intermediate Treasuries",
    "LQD": "Investment-grade credit",
    "HYG": "High-yield credit",
    "GLD": "Gold",
    "VNQ": "US real estate",
    "DBC": "Broad commodities",
}
START, END = "2010-01-01", "2025-09-30"
CACHE = Path(__file__).resolve().parent.parent / "data" / "etf_prices.csv.gz"


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices, one column per ETF."""
    if refresh or not CACHE.exists():
        import yfinance as yf

        raw = yf.download(list(ASSETS), start=START, end=END, auto_adjust=True, progress=False)
        CACHE.parent.mkdir(exist_ok=True)
        raw["Close"][list(ASSETS)].dropna().to_csv(CACHE)
    return pd.read_csv(CACHE, index_col=0, parse_dates=True)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna()
