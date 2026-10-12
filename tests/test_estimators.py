import numpy as np

from optim.estimators import ledoit_wolf, sample_cov


def _data(t, n=10, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) * 0.3
    true = a @ a.T + np.eye(n)
    return rng.multivariate_normal(np.zeros(n), true, size=t), true


def test_shrinkage_is_between_zero_and_one_and_matrix_is_positive_definite():
    x, _ = _data(60)
    sigma, s = ledoit_wolf(x)
    assert 0 <= s <= 1
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_fewer_observations_mean_more_shrinkage():
    _, s_short = ledoit_wolf(_data(30)[0])
    _, s_long = ledoit_wolf(_data(3000)[0])
    assert s_short > s_long


def test_shrinkage_beats_the_sample_covariance_on_short_samples():
    errs_lw, errs_s = [], []
    for seed in range(20):
        x, true = _data(25, seed=seed)
        errs_lw.append(np.sum((ledoit_wolf(x)[0] - true) ** 2))
        errs_s.append(np.sum((sample_cov(x) - true) ** 2))
    assert np.mean(errs_lw) < np.mean(errs_s)


def test_conditioning_improves():
    x, _ = _data(30)
    assert np.linalg.cond(ledoit_wolf(x)[0]) < np.linalg.cond(sample_cov(x))
