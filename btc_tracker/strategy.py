"""
Combined strategy engine: merges momentum, mean reversion, and Monte Carlo
into a unified probability estimate for Kalshi contract trading.
"""

import numpy as np
import logging
from btc_tracker.signals import generate_signals, compute_rsi, compute_bollinger
from btc_tracker.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Multi-strategy ensemble that combines:
    1. Momentum (MACD + trend following)
    2. Mean reversion (RSI + Bollinger Band extremes)
    3. Monte Carlo simulation
    4. Bayesian weighting based on recent accuracy

    Each strategy votes on direction with a probability, then results are
    combined using weighted average. Weights adapt based on recent performance.
    """

    def __init__(self):
        self.mc = MonteCarloSimulator(n_simulations=3000)
        # Track accuracy of each strategy for adaptive weighting
        self.strategy_scores = {
            "momentum": {"correct": 10, "total": 20},  # Start with prior
            "mean_reversion": {"correct": 10, "total": 20},
            "monte_carlo": {"correct": 10, "total": 20},
        }

    def _momentum_probability(self, prices, strike, current_price):
        """Momentum strategy: follow the trend."""
        if len(prices) < 30:
            return 0.5

        signals = generate_signals(prices)

        # Base: is trend up or down?
        if signals.direction == "higher":
            base_prob = 0.5 + signals.confidence * 0.3
        else:
            base_prob = 0.5 - signals.confidence * 0.3

        # Adjust for distance to strike
        distance_pct = (strike - current_price) / current_price
        # Harder to reach further strikes
        adjustment = -distance_pct * 5  # Each 1% distance reduces prob by 5%
        prob = base_prob + adjustment

        return float(np.clip(prob, 0.02, 0.98))

    def _mean_reversion_probability(self, prices, strike, current_price):
        """Mean reversion: extreme RSI/BB values tend to revert."""
        if len(prices) < 20:
            return 0.5

        rsi = compute_rsi(prices)
        bb = compute_bollinger(prices)

        prob = 0.5

        if rsi is not None:
            # RSI < 30 -> likely to go up, RSI > 70 -> likely to go down
            if rsi < 30:
                prob += 0.15 * (30 - rsi) / 30
            elif rsi > 70:
                prob -= 0.15 * (rsi - 70) / 30

        if bb is not None:
            # Near lower band -> likely to revert up
            if bb < 0.2:
                prob += 0.1 * (0.2 - bb) / 0.2
            elif bb > 0.8:
                prob -= 0.1 * (bb - 0.8) / 0.2

        # Adjust for strike distance
        distance_pct = (strike - current_price) / current_price
        prob -= distance_pct * 3

        return float(np.clip(prob, 0.02, 0.98))

    def _monte_carlo_probability(self, prices, strike, current_price,
                                  hours_to_expiry=24):
        """Monte Carlo: simulation-based probability."""
        return self.mc.estimate_probability(
            current_price=current_price,
            strike_price=strike,
            prices=prices,
            hours_to_expiry=hours_to_expiry,
            direction="higher",
        )

    def get_strategy_weights(self):
        """Get adaptive weights based on historical accuracy."""
        weights = {}
        for name, scores in self.strategy_scores.items():
            # Bayesian estimate of accuracy: (correct + 1) / (total + 2)
            accuracy = (scores["correct"] + 1) / (scores["total"] + 2)
            weights[name] = accuracy
        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def update_accuracy(self, strategy_name, was_correct):
        """Update strategy accuracy tracking."""
        if strategy_name in self.strategy_scores:
            self.strategy_scores[strategy_name]["total"] += 1
            if was_correct:
                self.strategy_scores[strategy_name]["correct"] += 1

    def estimate_probability(self, prices, strike, current_price,
                              hours_to_expiry=24):
        """
        Ensemble probability estimate combining all strategies.
        Returns probability that BTC will be ABOVE strike at expiry.
        """
        weights = self.get_strategy_weights()

        # Get individual strategy estimates
        probs = {
            "momentum": self._momentum_probability(prices, strike, current_price),
            "mean_reversion": self._mean_reversion_probability(
                prices, strike, current_price
            ),
            "monte_carlo": self._monte_carlo_probability(
                prices, strike, current_price, hours_to_expiry
            ),
        }

        # Weighted average
        ensemble_prob = sum(probs[k] * weights[k] for k in probs)

        logger.debug(
            f"Strike ${strike:,.0f}: momentum={probs['momentum']:.2f} "
            f"mean_rev={probs['mean_reversion']:.2f} "
            f"mc={probs['monte_carlo']:.2f} -> ensemble={ensemble_prob:.2f}"
        )

        return {
            "ensemble": round(ensemble_prob, 4),
            "strategies": {k: round(v, 4) for k, v in probs.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
        }

    def get_distribution(self, prices, current_price, hours_to_expiry=24):
        """Get Monte Carlo price distribution for dashboard display."""
        return self.mc.get_price_distribution(
            current_price, prices, hours_to_expiry
        )
