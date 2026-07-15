# Quantitative Portfolio Optimization

Mean-variance optimization, efficient frontier, and advanced portfolio allocation strategies.

## Features

- Mean-Variance Optimization (Markowitz)
- Efficient Frontier computation and visualization
- Kelly Criterion for optimal position sizing
- Risk Parity allocation
- Backtest comparisons across strategies
- Sharpe ratio, volatility, drawdown analysis

## Motivation

Portfolio optimization is the foundation of systematic risk management. Demonstrates understanding of modern portfolio theory and practical allocation methods.

## Getting Started

```bash
pip install -r requirements.txt
python examples/efficient_frontier.py
```

## Project Structure

```
├── portfolio/
│   ├── mean_variance.py # Markowitz optimization
│   ├── kelly.py         # Kelly criterion
│   ├── risk_parity.py   # Risk parity allocation
│   ├── backtest.py      # Strategy comparison
│   └── metrics.py       # Performance metrics
├── examples/
│   ├── efficient_frontier.py
│   └── compare_strategies.py
├── tests/
├── requirements.txt
└── README.md
```

## Example Output

```
Strategy          Return   Vol    Sharpe
───────────────────────────────────────
Equal Weight      12.3%   18.5%   0.66
Mean-Variance     15.7%   14.2%   1.11
Risk Parity       13.1%   11.8%   1.11
Kelly             18.2%   20.1%   0.91
```
