import numpy as np


class KellyCriterion:
    def __init__(self, fraction=0.25):
        self.fraction = fraction

    def from_binary_outcome(self, win_prob, odds):
        f = (odds * win_prob - (1 - win_prob)) / odds
        return max(0, f)

    def from_continuous_returns(self, returns):
        mu = returns.mean()
        sigma2 = returns.var()
        f = mu / sigma2 if sigma2 > 0 else 0
        return f

    def fractional_kelly(self, f_full):
        return self.fraction * f_full

    def growth_rate(self, f, mu, sigma2):
        return f * mu - 0.5 * f * f * sigma2

    @staticmethod
    def simulate_kelly(returns, fractions=None):
        if fractions is None:
            fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

        results = []
        for f in fractions:
            portfolio = [1.0]
            for r in returns:
                portfolio.append(portfolio[-1] * (1 + f * r))
            results.append({
                "fraction": f,
                "final_value": portfolio[-1],
                "cagr": portfolio[-1] ** (252 / len(returns)) - 1,
            })

        return pd.DataFrame(results)
