import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MeanVarianceOptimizer:
    def __init__(self, returns, risk_free_rate=0.05):
        self.returns = returns
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)

    def portfolio_stats(self, weights):
        weights = np.array(weights)
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def min_volatility(self):
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        result = minimize(
            self._neg_sharpe, self.n_assets * [1. / self.n_assets],
            args=args, method="SLSQP", bounds=bounds,
            constraints=constraints
        )
        return result.x

    def max_sharpe(self):
        def neg_sharpe(weights):
            return -self.portfolio_stats(weights)[2]

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        result = minimize(
            neg_sharpe, self.n_assets * [1. / self.n_assets],
            method="SLSQP", bounds=bounds,
            constraints=constraints
        )
        return result.x

    def efficient_return(self, target_return):
        def portfolio_vol(weights):
            return self.portfolio_stats(weights)[1]

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x: self.portfolio_stats(x)[0] - target_return},
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        result = minimize(
            portfolio_vol, self.n_assets * [1. / self.n_assets],
            method="SLSQP", bounds=bounds,
            constraints=constraints
        )
        return result.x


class EfficientFrontier:
    def __init__(self, optimizer, num_portfolios=50):
        self.optimizer = optimizer
        self.num_portfolios = num_portfolios

    def compute(self):
        mean_ret = self.optimizer.mean_returns
        min_ret = mean_ret.min()
        max_ret = mean_ret.max()
        target_returns = np.linspace(min_ret, max_ret, self.num_portfolios)

        frontier = []
        for target in target_returns:
            try:
                weights = self.optimizer.efficient_return(target)
                ret, vol, sharpe = self.optimizer.portfolio_stats(weights)
                frontier.append({"return": ret, "vol": vol, "sharpe": sharpe, "weights": weights})
            except Exception:
                continue

        return pd.DataFrame(frontier)
