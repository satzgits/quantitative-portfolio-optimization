import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from portfolio.mean_variance import MeanVarianceOptimizer, EfficientFrontier

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    print(f"Downloading 3 years of data for {tickers}...")
    data = yf.download(tickers, start="2021-01-01", end="2023-12-31")["Close"]
    returns = data.pct_change().dropna()

    optimizer = MeanVarianceOptimizer(returns)
    ef = EfficientFrontier(optimizer, num_portfolios=50)
    frontier = ef.compute()

    max_sharpe_w = optimizer.max_sharpe()
    min_vol_w = optimizer.min_volatility()
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = optimizer.portfolio_stats(max_sharpe_w)
    min_vol_ret, min_vol_vol, min_vol_sr = optimizer.portfolio_stats(min_vol_w)

    print("\n" + "=" * 55)
    print("   EFFICIENT FRONTIER ANALYSIS")
    print("=" * 55)
    print(f"\n  Assets: {', '.join(tickers)}")
    print(f"  Period: 2021-01-01 to 2023-12-31")
    print(f"  Risk-free rate: {optimizer.risk_free_rate:.1%}")
    print("-" * 55)
    print(f"\n  Max Sharpe Portfolio:")
    print(f"    Return: {max_sharpe_ret:.2%}   Vol: {max_sharpe_vol:.2%}   Sharpe: {max_sharpe_sr:.2f}")
    for name, w in zip(tickers, max_sharpe_w):
        print(f"    {name:<6s}: {w:.2%}")
    print(f"\n  Min Volatility Portfolio:")
    print(f"    Return: {min_vol_ret:.2%}   Vol: {min_vol_vol:.2%}   Sharpe: {min_vol_sr:.2f}")
    for name, w in zip(tickers, min_vol_w):
        print(f"    {name:<6s}: {w:.2%}")
    print("=" * 55)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(frontier["vol"], frontier["return"], c=frontier["sharpe"],
               cmap="viridis", s=30, alpha=0.8)
    ax.scatter(min_vol_vol, min_vol_ret, c="blue", marker="D", s=100,
               label="Min Volatility")
    ax.scatter(max_sharpe_vol, max_sharpe_ret, c="red", marker="*", s=200,
               label="Max Sharpe (Tangency)")

    for name, w in zip(tickers, max_sharpe_w):
        r = optimizer.mean_returns[list(tickers).index(name)]
        v = np.sqrt(optimizer.cov_matrix.loc[name, name])
        ax.scatter(v, r, alpha=0.5, s=80)
        ax.annotate(name, (v, r), fontsize=8, alpha=0.7)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier — Max Sharpe & Min Volatility Portfolios")
    ax.legend()
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Sharpe Ratio")

    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150)
    print("\nPlot saved to efficient_frontier.png")
