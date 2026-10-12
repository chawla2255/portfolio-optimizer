"""Covariance estimators: the sample covariance and Ledoit-Wolf shrinkage.

Ledoit & Wolf (2004) shrink the sample covariance S toward a scaled identity F = m I, with the
blend that minimises expected squared (Frobenius) error:

    Sigma = s F + (1 - s) S,   s = min(b2, d2) / d2

where d2 measures how far S is from F and b2 how noisy S is. Few observations relative to the
number of assets push s toward 1; optimisers then see a better-conditioned matrix and stop
exploiting estimation noise.
"""
from __future__ import annotations

import numpy as np


def sample_cov(returns: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(returns, float), rowvar=False, ddof=1)


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrunk covariance and the shrinkage intensity in [0, 1]."""
    x = np.asarray(returns, float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    m = np.trace(s) / n
    f = m * np.eye(n)
    d2 = np.sum((s - f) ** 2) / n
    b2_bar = sum(np.sum((np.outer(row, row) - s) ** 2) for row in x) / (t * t * n)
    b2 = min(b2_bar, d2)
    shrink = 0.0 if d2 == 0 else b2 / d2
    sigma = shrink * f + (1 - shrink) * s
    return sigma * t / (t - 1), float(shrink)
