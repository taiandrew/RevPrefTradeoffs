"""Sanity checks for the QUAIDS estimation.

Run with ``python tests/test_quaids.py`` (from ``code/``) or ``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quaids import quaids, _price_indices

GOODS = ['a', 'b', 'c']

# True parameters satisfying adding-up, homogeneity and symmetry
ALPHA = np.array([0.4, 0.35, 0.25])
BETA = np.array([0.05, -0.02, -0.03])
LAMBDA = np.array([0.02, -0.03, 0.01])
GAMMA = np.array([
    [0.08, -0.05, -0.03],
    [-0.05, 0.09, -0.04],
    [-0.03, -0.04, 0.07],
])


def simulate(n=4000, lam=LAMBDA, noise_sd=0.005, seed=7):
    """Draw (quantities, prices) from a known QUAIDS.

    Prices and expenditure are rescaled to have median exactly 1, so the
    estimator's median normalisation leaves the true parameters unchanged.
    """
    rng = np.random.default_rng(seed)
    k = len(GOODS)

    p = rng.lognormal(0, 0.25, size=(n, k))
    p /= np.median(p, axis=0)
    x = rng.lognormal(0, 0.5, size=n)
    x /= np.median(x)

    lnp, lnx = np.log(p), np.log(x)
    lnx_real, b_tilde = _price_indices(lnp, lnx, ALPHA, GAMMA, BETA)
    w = (ALPHA[None, :] + lnp @ GAMMA.T + np.outer(lnx_real, BETA)
         + np.outer(lnx_real ** 2 / b_tilde, lam))

    # Adding-up-consistent errors: noise on the first K-1 shares, last absorbs
    e = rng.normal(0, noise_sd, size=(n, k - 1))
    w[:, :-1] += e
    w[:, -1] -= e.sum(axis=1)
    assert (w > 0).all(), "simulated shares must stay positive"

    q = w * x[:, None] / p
    idx = pd.RangeIndex(n)
    return (pd.DataFrame(q, columns=GOODS, index=idx),
            pd.DataFrame(p, columns=GOODS, index=idx))


def test_recovers_true_parameters():
    quantities, prices = simulate()
    res = quaids(quantities, prices)
    assert res.converged
    np.testing.assert_allclose(res.alpha, ALPHA, atol=0.01)
    np.testing.assert_allclose(res.beta, BETA, atol=0.01)
    np.testing.assert_allclose(res.lambda_, LAMBDA, atol=0.01)
    np.testing.assert_allclose(res.gamma, GAMMA, atol=0.02)
    assert np.isfinite(res.alpha_se).all() and (res.alpha_se > 0).all()


def test_restrictions_hold():
    quantities, prices = simulate(n=500, seed=21)
    res = quaids(quantities, prices)
    assert abs(res.alpha.sum() - 1) < 1e-10
    assert abs(res.beta.sum()) < 1e-10
    assert abs(res.lambda_.sum()) < 1e-10
    g = res.gamma.to_numpy()
    np.testing.assert_allclose(g, g.T, atol=1e-10)          # symmetry
    np.testing.assert_allclose(g.sum(axis=1), 0, atol=1e-10)  # homogeneity
    # Adding-up: fitted shares sum to 1 row-wise
    np.testing.assert_allclose(res.fitted.sum(axis=1), 1, atol=1e-10)


def test_aids_special_case():
    # Data generated with lambda = 0: the quadratic terms estimate near zero
    quantities, prices = simulate(lam=np.zeros(3), seed=3)
    res = quaids(quantities, prices)
    np.testing.assert_allclose(res.lambda_, 0, atol=0.01)


def test_validation_errors():
    quantities, prices = simulate(n=50, seed=5)
    with pytest.raises(ValueError):
        quaids(quantities.iloc[:, :-1], prices)  # shape mismatch
    with pytest.raises(ValueError):
        quaids(quantities, -prices)  # non-positive prices
    zero_bundle = quantities.copy()
    zero_bundle.iloc[0] = 0.0
    with pytest.raises(ValueError):
        quaids(zero_bundle, prices)  # zero total expenditure
    never_bought = quantities.copy()
    never_bought.iloc[:, 0] = 0.0
    with pytest.raises(ValueError):
        quaids(never_bought, prices)  # constant (all-zero) budget share


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
