import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from portfolio.mean_variance import MeanVarianceOptimizer
from portfolio.risk_parity import RiskParity
from portfolio.backtest import PortfolioBacktest

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    print(f"Downloading 5 years of data for {tickers}...")
    data = yf.download(tickers, start="2019-01-01", end="2023-12-31")["Close"]
    returns = data.pct_change().dropna()

    def max_sharpe_weights(window):
        opt = MeanVarianceOptimizer(window)
        return opt.max_sharpe()

    def min_vol_weights(window):
        opt = MeanVarianceOptimizer(window)
        return opt.min_volatility()

    def risk_parity_weights(window):
        cov = window.cov() * 252
        rp = RiskParity(cov.values)
        return rp.risk_parity_weights()

    def equal_weights(window):
        return np.array([1.0 / window.shape[1]] * window.shape[1])

    bt = PortfolioBacktest(data)

    strategies = {
        "Equal Weight": lambda w: equal_weights(w),
        "Min Volatility": lambda w: min_vol_weights(w),
        "Max Sharpe": lambda w: max_sharpe_weights(w),
        "Risk Parity": lambda w: risk_parity_weights(w),
    }

    results = {}
    plt.figure(figsize=(12, 6))

    print("\n" + "=" * 65)
    print("   PORTFOLIO STRATEGY COMPARISON (2019-2023)")
    print("=" * 65)
    print(f"  {'Strategy':<20s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s} {'Max DD':>8s}")
    print("-" * 65)

    for name, wf in strategies.items():
        eq, metrics = bt.run(wf, name)
        results[name] = eq
        s = metrics.summary()
        print(f"  {name:<20s} {s['return']:>8.2%} {s['vol']:>8.2%} "
              f"{s['sharpe']:>8.2f} {s['max_dd']:>8.2%}")
        plt.plot(eq.index, eq.values, label=name, linewidth=2)

    print("=" * 65)

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Portfolio Strategy Comparison (Monthly Rebalance)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("strategy_comparison.png", dpi=150)
    print("\nPlot saved to strategy_comparison.png")
