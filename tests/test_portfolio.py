import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from portfolio.mean_variance import MeanVarianceOptimizer, EfficientFrontier
from portfolio.risk_parity import RiskParity
from portfolio.kelly import KellyCriterion


def make_sample_returns(n_assets=4, n_periods=500):
    np.random.seed(42)
    mean = np.array([0.0008, 0.0006, 0.0004, 0.0002])
    cov = np.array([
        [0.0004, 0.0002, 0.0001, 0.0000],
        [0.0002, 0.0003, 0.0001, 0.0000],
        [0.0001, 0.0001, 0.0005, 0.0001],
        [0.0000, 0.0000, 0.0001, 0.0006],
    ])
    return pd.DataFrame(np.random.multivariate_normal(mean[:n_assets], cov[:n_assets, :n_assets], n_periods))


def test_mean_variance_shapes():
    returns = make_sample_returns()
    opt = MeanVarianceOptimizer(returns)
    w = opt.max_sharpe()
    assert len(w) == 4, f"Expected 4 weights, got {len(w)}"
    assert abs(sum(w) - 1) < 1e-6, "Weights should sum to 1"
    ret, vol, sr = opt.portfolio_stats(w)
    assert ret > 0, "Expected positive return"
    assert vol > 0, "Expected positive volatility"
    print("  ✓ Mean-variance optimization")


def test_efficient_frontier():
    returns = make_sample_returns()
    opt = MeanVarianceOptimizer(returns)
    ef = EfficientFrontier(opt, num_portfolios=20)
    frontier = ef.compute()
    assert len(frontier) > 0, "Frontier should have points"
    assert all(frontier["vol"] > 0), "All vols should be positive"
    print("  ✓ Efficient frontier computation")


def test_risk_parity_weights():
    returns = make_sample_returns()
    cov = returns.cov() * 252
    rp = RiskParity(cov.values)
    weights, rc = rp.equal_risk_contribution()
    assert abs(sum(weights) - 1) < 1e-6, "Weights should sum to 1"
    assert all(w >= 0 for w in weights), "No negative weights"
    rc_std = np.std(rc)
    equal_rc_std = np.std([0.25] * 4)
    assert rc_std < equal_rc_std, "Risk parity should equalize contributions"
    print("  ✓ Risk parity allocation")


def test_kelly():
    kc = KellyCriterion()
    f = kc.from_binary_outcome(win_prob=0.6, odds=1)
    assert 0 < f < 1, f"Expected fraction between 0 and 1, got {f}"
    returns = np.random.normal(0.001, 0.02, 1000)
    f_cont = kc.from_continuous_returns(returns)
    assert f_cont >= 0, "Kelly fraction should be non-negative"
    print("  ✓ Kelly criterion")


if __name__ == "__main__":
    test_mean_variance_shapes()
    test_efficient_frontier()
    test_risk_parity_weights()
    test_kelly()
    print("\nAll portfolio optimization tests passed!")
