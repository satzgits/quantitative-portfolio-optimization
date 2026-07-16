import numpy as np
import pandas as pd


class PortfolioMetrics:
    def __init__(self, equity_curve, risk_free_rate=0.05):
        self.equity = equity_curve
        self.risk_free_rate = risk_free_rate
        self.returns = equity_curve.pct_change().dropna()

    @property
    def annualized_return(self):
        total = self.equity.iloc[-1] / self.equity.iloc[0]
        years = len(self.returns) / 252
        return total ** (1 / years) - 1 if years > 0 else 0

    @property
    def annualized_vol(self):
        return float(self.returns.std() * np.sqrt(252))

    @property
    def sharpe_ratio(self):
        if self.annualized_vol == 0:
            return 0
        return (self.annualized_return - self.risk_free_rate) / self.annualized_vol

    @property
    def max_drawdown(self):
        roll_max = self.equity.expanding().max()
        dd = (self.equity - roll_max) / roll_max
        return float(dd.min())

    @property
    def calmar_ratio(self):
        if self.max_drawdown == 0:
            return 0
        return self.annualized_return / abs(self.max_drawdown)

    def summary(self):
        return {
            "return": self.annualized_return,
            "vol": self.annualized_vol,
            "sharpe": self.sharpe_ratio,
            "max_dd": self.max_drawdown,
            "calmar": self.calmar_ratio,
        }


class PortfolioBacktest:
    def __init__(self, prices, rebalance_freq=21):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.rebalance_freq = rebalance_freq
        self.n_assets = prices.shape[1]
        self.asset_names = prices.columns

    def run(self, weight_func, name="Strategy"):
        eq = [1.0]
        current_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        for i in range(len(self.returns)):
            if i % self.rebalance_freq == 0:
                window = self.returns.iloc[max(0, i - 252):i]
                if len(window) > 20:
                    current_weights = weight_func(window)

            r = self.returns.iloc[i].values
            port_return = np.dot(current_weights, r)
            eq.append(eq[-1] * (1 + port_return))

        eq_series = pd.Series(eq, index=[self.prices.index[0]] + list(self.returns.index))
        metrics = PortfolioMetrics(eq_series)
        return eq_series, metrics
