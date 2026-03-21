"""
Monte Carlo simulation for BTC price probability estimation.
Generates thousands of possible price paths to estimate the probability
of BTC being above/below a strike price at expiry.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Uses geometric Brownian motion calibrated to historical volatility
    to simulate future BTC price paths.
    """

    def __init__(self, n_simulations=5000):
        self.n_simulations = n_simulations

    def estimate_probability(self, current_price, strike_price, prices,
                              hours_to_expiry=24, direction="higher"):
        """
        Estimate probability that BTC will be above/below strike at expiry.

        Uses historical returns to calibrate drift and volatility,
        then simulates n_simulations price paths.
        """
        if len(prices) < 20:
            return 0.5

        # Calculate log returns
        log_returns = np.diff(np.log(prices[-200:]))

        if len(log_returns) < 5:
            return 0.5

        # Calibrate drift (mu) and volatility (sigma)
        mu = np.mean(log_returns)
        sigma = np.std(log_returns)

        if sigma == 0:
            return 0.5

        # Scale to forecast horizon
        # Assume each data point is ~1 hour apart
        steps = max(1, int(hours_to_expiry))

        # Simulate paths using geometric Brownian motion
        dt = 1.0  # 1 step = 1 data interval
        random_walks = np.random.normal(
            loc=(mu - 0.5 * sigma**2) * dt,
            scale=sigma * np.sqrt(dt),
            size=(self.n_simulations, steps),
        )

        # Cumulative sum to get log price paths
        log_paths = np.cumsum(random_walks, axis=1)

        # Final prices
        final_log_prices = np.log(current_price) + log_paths[:, -1]
        final_prices = np.exp(final_log_prices)

        # Count outcomes
        if direction == "higher":
            prob = np.mean(final_prices >= strike_price)
        else:
            prob = np.mean(final_prices < strike_price)

        return round(float(np.clip(prob, 0.01, 0.99)), 4)

    def get_price_distribution(self, current_price, prices,
                                hours_to_expiry=24):
        """
        Return distribution statistics for future price.
        Useful for identifying which strike prices are interesting.
        """
        if len(prices) < 20:
            return None

        log_returns = np.diff(np.log(prices[-200:]))
        mu = np.mean(log_returns)
        sigma = np.std(log_returns)

        if sigma == 0:
            return None

        steps = max(1, int(hours_to_expiry))
        dt = 1.0

        random_walks = np.random.normal(
            loc=(mu - 0.5 * sigma**2) * dt,
            scale=sigma * np.sqrt(dt),
            size=(self.n_simulations, steps),
        )

        log_paths = np.cumsum(random_walks, axis=1)
        final_prices = np.exp(np.log(current_price) + log_paths[:, -1])

        percentiles = np.percentile(final_prices, [5, 10, 25, 50, 75, 90, 95])

        return {
            "mean": round(float(np.mean(final_prices)), 2),
            "median": round(float(percentiles[3]), 2),
            "std": round(float(np.std(final_prices)), 2),
            "p5": round(float(percentiles[0]), 2),
            "p10": round(float(percentiles[1]), 2),
            "p25": round(float(percentiles[2]), 2),
            "p50": round(float(percentiles[3]), 2),
            "p75": round(float(percentiles[4]), 2),
            "p90": round(float(percentiles[5]), 2),
            "p95": round(float(percentiles[6]), 2),
            "prob_up": round(float(np.mean(final_prices > current_price)), 4),
            "prob_down": round(float(np.mean(final_prices < current_price)), 4),
            "expected_move_pct": round(
                float((np.mean(final_prices) - current_price) / current_price * 100), 3
            ),
        }
