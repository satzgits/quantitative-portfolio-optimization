# Quantitative Portfolio Optimization

Mean-variance optimization, efficient frontier, Kelly criterion, risk parity, and strategy comparison.

## Overview

This project implements the core mathematical frameworks for portfolio construction and allocation optimization. It moves beyond "pick good stocks" to "how much of each should I hold?" — the fundamental question of quantitative risk management.

### Why Portfolio Optimization?

Finding alpha (a profitable strategy) is only half the battle. The other half is position sizing, risk management, and portfolio construction. Even a mediocre strategy with excellent risk management can outperform a great strategy with poor position sizing.

This project covers four major allocation frameworks:

1. **Mean-Variance (Markowitz)** — maximize return for a given risk level
2. **Efficient Frontier** — the set of optimal portfolios
3. **Kelly Criterion** — maximize long-term growth rate
4. **Risk Parity** — equal risk contribution from each asset

## Features

- **Mean-Variance Optimization** — find the tangency portfolio that maximizes Sharpe ratio
- **Efficient Frontier** — compute and visualize the full risk-return frontier
- **Kelly Criterion** — optimal position sizing for maximum geometric growth
- **Risk Parity** — allocate so each asset contributes equally to portfolio risk
- **Strategy Comparison** — backtest all four approaches on historical data and compare
- **Performance Metrics** — Sharpe ratio, volatility, max drawdown, Calmar ratio

## Project Structure

```
quantitative-portfolio-optimization/
├── portfolio/
│   ├── __init__.py
│   ├── mean_variance.py # Markowitz optimization & efficient frontier
│   ├── kelly.py         # Kelly criterion position sizing
│   ├── risk_parity.py   # Risk parity allocation
│   ├── backtest.py      # Strategy backtesting & comparison
│   └── metrics.py       # Performance statistics
├── examples/
│   ├── efficient_frontier.py   # Compute and plot efficient frontier
│   └── compare_strategies.py   # Backtest all 4 allocation methods
├── tests/
│   └── test_portfolio.py
├── requirements.txt
└── README.md
```

## The Math

### Mean-Variance Optimization (Markowitz 1952)

The classic framework. Find weights w that minimize portfolio variance for a given target return:

```
min     wᵀ Σ w
s.t.    wᵀ μ = μ_target
        Σ wᵢ = 1
        wᵢ ≥ 0 (no shorting constraint, optional)
```

Where:
- w = vector of asset weights
- Σ = covariance matrix of returns
- μ = vector of expected returns

Solution: Quadratic programming (using `scipy.optimize`).

### Efficient Frontier

The set of portfolios that offer the highest expected return for each level of risk. Every rational investor should hold a portfolio on this curve.

**Key portfolios:**
- **Minimum Variance Portfolio**: Lowest possible risk
- **Maximum Sharpe Portfolio** (Tangency): Best risk-adjusted return
- **Maximum Return Portfolio**: Concentrated in the best asset

### Kelly Criterion (1956)

Maximizes long-term geometric growth rate. For a binary outcome with probability p of winning b:

```
f* = (bp - q) / b
```

Where f* is the fraction of capital to bet.

For continuous returns (normal distribution):

```
f* = μ / σ²
```

Kelly is aggressive — it maximizes growth but can lead to 100% drawdowns in practice. Most traders use fractional Kelly (e.g., half-Kelly).

### Risk Parity

Instead of equal capital allocation, equalize risk contribution from each asset:

```
RCᵢ = wᵢ × (Σw)ᵢ / √(wᵀΣw)

Goal: RC₁ = RC₂ = ... = RCₙ
```

This creates naturally diversified portfolios that aren't dominated by high-volatility assets.

## How It Works (Step by Step)

### 1. Mean-Variance Module (`portfolio/mean_variance.py`)

```
Input:  historical returns (DataFrame), risk-free rate
Process:
  1. Compute mean returns vector μ
  2. Compute covariance matrix Σ
  3. Solve quadratic program for target returns
  4. Trace out efficient frontier from min-var to max-return
  5. Identify tangency portfolio (max Sharpe)
Output: optimal weights, frontier curve, portfolio stats
```

### 2. Kelly Module (`portfolio/kelly.py`)

```
Input:  historical returns or win/loss statistics
Process:
  1. Estimate μ (expected return) and σ² (variance)
  2. Compute f* = μ / σ²
  3. Apply fractional Kelly scaling
  4. Compute implied growth rate
Output: optimal fraction, growth rate
```

### 3. Risk Parity Module (`portfolio/risk_parity.py`)

```
Input:  covariance matrix
Process:
  1. Start with equal weights
  2. Iteratively adjust to equalize risk contributions
  3. Converge when all RCᵢ are within tolerance
Output: risk parity weights
```

### 4. Backtest Module (`portfolio/backtest.py`)

```
Input:  historical prices, rebalance frequency
Process:
  1. Compute rolling windows
  2. For each window, compute optimal weights via each method
  3. Rebalance portfolio to target weights
  4. Track portfolio value over time
Output: equity curves for each strategy
```

### 5. Metrics Module (`portfolio/metrics.py`)

Common performance statistics applied to any equity curve:
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Calmar ratio (return / max drawdown)
- Win rate (if trade-level data available)

## Example Output

```
=== Strategy Comparison (2015-2024) ===
Strategy                Return    Vol     Sharpe   Max DD
─────────────────────────────────────────────────────────
Equal Weight            12.3%    18.5%    0.66    -25.4%
Mean-Variance (Max S)   15.7%    14.2%    1.11    -16.8%
Risk Parity             13.1%    11.8%    1.11    -12.2%
Kelly (half)            18.2%    20.1%    0.91    -35.7%

Efficient Frontier:
  Min Variance:  Vol=10.5%, Return=11.2%
  Max Sharpe:    Vol=14.2%, Return=15.7%
  Max Return:    Vol=24.8%, Return=19.3%
```

## Getting Started

```bash
pip install -r requirements.txt
python examples/efficient_frontier.py
python examples/compare_strategies.py
```

## Why This Matters for Quant Trading

Portfolio optimization is the bridge between raw strategies and real money management:
- **Know what to trade** ← strategy research (backtesting engine)
- **Know how to size** ← Kelly criterion (this project)
- **Know how to diversify** ← risk parity (this project)
- **Know your risk budget** ← mean-variance (this project)

A quant trader who only knows how to find signals but not how to size and diversify them will blow up. This project proves you understand the full picture.
