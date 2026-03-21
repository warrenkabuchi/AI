"""
Kalshi BTC contract analyzer.
Uses Bayesian probability estimation and Expected Value (EV) calculations
to find mispriced contracts and generate trade recommendations.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KalshiContract:
    """Represents a Kalshi BTC price contract."""
    contract_id: str
    title: str
    direction: str  # "higher" or "lower"
    strike_price: float  # BTC price threshold
    expiry: str  # ISO timestamp
    yes_price: float  # 0-1 (cents on the dollar)
    no_price: float  # 0-1
    volume: int = 0


@dataclass
class TradeRecommendation:
    contract: KalshiContract
    side: str  # "YES" or "NO"
    model_probability: float  # Our estimated probability
    market_probability: float  # Market implied probability (yes_price)
    edge: float  # model_prob - market_prob
    ev: float  # Expected value per dollar
    confidence: str  # "low", "medium", "high"
    reasoning: str


class BayesianPredictor:
    """
    Uses Bayesian updating with historical price movements to estimate
    the probability that BTC will be above/below a strike price.
    """

    def __init__(self):
        # Prior: start with 50/50
        self.prior_higher = 0.5

    def estimate_probability(self, current_price, strike_price, prices,
                             signal_direction, signal_confidence, hours_to_expiry=24):
        """
        Estimate probability that BTC will be above strike_price at expiry.

        Uses:
        1. Historical volatility to model price distribution
        2. Current trend/signal as Bayesian prior update
        3. Distance from strike as key factor
        """
        if len(prices) < 10:
            return 0.5

        # Step 1: Calculate historical volatility (annualized -> period)
        returns = np.diff(np.log(prices[-100:]))  # Log returns
        if len(returns) == 0:
            return 0.5

        # Estimate data frequency (assume data interval from history)
        hourly_vol = np.std(returns) * np.sqrt(1)  # Per-period vol
        period_vol = hourly_vol * np.sqrt(max(hours_to_expiry, 1))

        if period_vol == 0:
            period_vol = 0.01  # Prevent division by zero

        # Step 2: Calculate z-score (how many std devs is strike from current)
        price_ratio = np.log(strike_price / current_price)
        z_score = price_ratio / period_vol

        # Base probability from normal distribution (prob of being above strike)
        from scipy.stats import norm
        base_prob_higher = 1 - norm.cdf(z_score)

        # Step 3: Bayesian update with signal
        # Convert signal to likelihood ratio
        if signal_direction == "higher":
            # Signal says higher -> increase probability of higher
            signal_lr = 1 + signal_confidence * 0.5  # Likelihood ratio
        else:
            signal_lr = 1 / (1 + signal_confidence * 0.5)

        # Bayesian update: posterior = prior * LR / normalizer
        odds = (base_prob_higher / (1 - base_prob_higher + 1e-10)) * signal_lr
        posterior = odds / (1 + odds)

        # Clamp to reasonable range
        return round(np.clip(posterior, 0.02, 0.98), 4)

    def estimate_probability_simple(self, current_price, strike_price, prices,
                                     signal_direction, signal_confidence,
                                     hours_to_expiry=24):
        """
        Simplified probability estimation without scipy dependency.
        Uses empirical distribution from historical data.
        """
        if len(prices) < 10:
            return 0.5

        # Calculate historical price changes over similar periods
        lookback = min(len(prices) - 1, 100)
        changes = []
        step = max(1, int(hours_to_expiry))

        for i in range(lookback):
            if i + step < len(prices):
                pct_change = (prices[i + step] - prices[i]) / prices[i]
                changes.append(pct_change)

        if not changes:
            # Fallback: use simple returns
            changes = [(prices[i+1] - prices[i]) / prices[i]
                       for i in range(len(prices) - 1)]

        if not changes:
            return 0.5

        # What % change is needed to reach strike?
        needed_change = (strike_price - current_price) / current_price

        # Count how often historical changes exceeded needed change
        count_above = sum(1 for c in changes if c >= needed_change)
        base_prob = count_above / len(changes)

        # Bayesian update with signal
        if signal_direction == "higher":
            adjustment = signal_confidence * 0.15
            adjusted = base_prob + adjustment * (1 - base_prob)
        else:
            adjustment = signal_confidence * 0.15
            adjusted = base_prob - adjustment * base_prob

        return round(np.clip(adjusted, 0.02, 0.98), 4)


class EVCalculator:
    """Calculate Expected Value for Kalshi contract trades."""

    @staticmethod
    def calculate_ev(buy_price, model_probability, payout=1.0):
        """
        Calculate EV of buying a contract.

        buy_price: cost in dollars (0-1 scale, e.g., 0.35 = 35 cents)
        model_probability: our estimated probability of winning
        payout: payout if correct (always $1 on Kalshi)
        """
        ev_win = model_probability * (payout - buy_price)
        ev_lose = (1 - model_probability) * (-buy_price)
        return round(ev_win + ev_lose, 4)

    @staticmethod
    def calculate_kelly(model_probability, market_price):
        """
        Kelly criterion for optimal bet sizing.
        Returns fraction of bankroll to risk.
        """
        if market_price <= 0 or market_price >= 1:
            return 0

        b = (1 - market_price) / market_price  # Odds ratio
        q = 1 - model_probability

        kelly = (model_probability * b - q) / b
        # Half-Kelly for safety
        return round(max(kelly * 0.5, 0), 4)


class ContractAnalyzer:
    """Analyzes Kalshi contracts and generates trade recommendations."""

    def __init__(self):
        self.predictor = BayesianPredictor()
        self.ev_calc = EVCalculator()

    def analyze_contract(self, contract, current_price, prices,
                         signal_direction, signal_confidence):
        """Analyze a single contract and return a recommendation if +EV."""

        # Estimate probability of BTC being above strike
        prob_higher = self.predictor.estimate_probability_simple(
            current_price=current_price,
            strike_price=contract.strike_price,
            prices=prices,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
        )

        if contract.direction == "higher":
            model_prob = prob_higher
        else:
            model_prob = 1 - prob_higher

        market_prob = contract.yes_price

        # Calculate EV for YES and NO sides
        ev_yes = self.ev_calc.calculate_ev(contract.yes_price, model_prob)
        ev_no = self.ev_calc.calculate_ev(contract.no_price, 1 - model_prob)

        # Pick the better side
        if ev_yes > ev_no and ev_yes > 0:
            side = "YES"
            ev = ev_yes
            edge = model_prob - market_prob
        elif ev_no > 0:
            side = "NO"
            ev = ev_no
            edge = (1 - model_prob) - contract.no_price
        else:
            return None  # No +EV trade

        # Confidence based on edge size and signal confidence
        if abs(edge) > 0.15 and signal_confidence > 0.5:
            conf = "high"
        elif abs(edge) > 0.08:
            conf = "medium"
        else:
            conf = "low"

        # Build reasoning
        distance_pct = abs(current_price - contract.strike_price) / current_price * 100
        reasoning = (
            f"BTC ${current_price:,.0f} vs strike ${contract.strike_price:,.0f} "
            f"({distance_pct:.1f}% away). "
            f"Model: {model_prob:.0%} vs Market: {market_prob:.0%} "
            f"({edge:+.1%} edge). "
            f"Signal: {signal_direction} ({signal_confidence:.0%} conf). "
            f"EV: {ev:+.2f} per contract."
        )

        return TradeRecommendation(
            contract=contract,
            side=side,
            model_probability=model_prob,
            market_probability=market_prob,
            edge=edge,
            ev=ev,
            confidence=conf,
            reasoning=reasoning,
        )

    def scan_opportunities(self, contracts, current_price, prices,
                           signal_direction, signal_confidence, min_ev=0.02):
        """Scan all contracts and return sorted +EV opportunities."""
        recommendations = []
        for contract in contracts:
            rec = self.analyze_contract(
                contract, current_price, prices,
                signal_direction, signal_confidence,
            )
            if rec and rec.ev >= min_ev:
                recommendations.append(rec)

        # Sort by EV descending
        recommendations.sort(key=lambda r: r.ev, reverse=True)
        return recommendations


def generate_simulated_contracts(current_price):
    """
    Generate simulated Kalshi-style BTC contracts for demo/testing.
    In production, these would come from the Kalshi API.
    """
    contracts = []
    now = datetime.now(timezone.utc)

    # Generate contracts at various strike prices around current price
    offsets = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]

    for offset in offsets:
        strike = round(current_price + offset, -2)  # Round to nearest 100

        # Simulate market prices based on distance from current
        distance_pct = offset / current_price
        # Simple logistic for demo - further away = cheaper YES for higher
        import math
        base_prob = 1 / (1 + math.exp(distance_pct * 20))

        # Add some noise and spread
        noise = np.random.uniform(-0.05, 0.05)
        yes_price = round(np.clip(base_prob + noise, 0.03, 0.97), 2)
        no_price = round(1 - yes_price, 2)

        contracts.append(KalshiContract(
            contract_id=f"BTC-{strike:.0f}-HIGHER-24H",
            title=f"BTC above ${strike:,.0f} at end of day?",
            direction="higher",
            strike_price=strike,
            expiry=now.isoformat(),
            yes_price=yes_price,
            no_price=no_price,
            volume=np.random.randint(50, 5000),
        ))

    return contracts
