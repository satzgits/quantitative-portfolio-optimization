import numpy as np
from scipy.optimize import minimize


class RiskParity:
    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix
        self.n_assets = cov_matrix.shape[0]

    def risk_contribution(self, weights):
        weights = np.array(weights)
        portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        marginal_contrib = np.dot(self.cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)
        return risk_contrib

    def risk_parity_weights(self):
        def risk_parity_objective(weights):
            weights = np.array(weights)
            rc = self.risk_contribution(weights)
            target = np.mean(rc)
            return np.sum((rc - target) ** 2)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        result = minimize(
            risk_parity_objective,
            self.n_assets * [1. / self.n_assets],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return result.x

    def equal_risk_contribution(self):
        weights = self.risk_parity_weights()
        rc = self.risk_contribution(weights)
        return weights, rc
