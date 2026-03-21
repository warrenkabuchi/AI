"""
Backtesting engine for BTC Kalshi trading strategies.
Validates signal accuracy and EV calculations against historical data.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from btc_tracker.signals import generate_signals
from btc_tracker.kalshi import ContractAnalyzer, KalshiContract

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_ev: float = 0.0
    avg_edge: float = 0.0
    equity_curve: list = field(default_factory=list)
    direction_accuracy: float = 0.0
    direction_correct: int = 0
    direction_total: int = 0


class Backtester:
    """
    Run strategy against historical price data to validate performance.
    Simulates Kalshi contracts at each time step and evaluates outcomes.
    """

    def __init__(self, lookback=50, forecast_horizon=24):
        """
        lookback: number of data points to feed into signal generator
        forecast_horizon: how many periods ahead the contract expires
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.analyzer = ContractAnalyzer()

    def run(self, prices, min_ev=0.02):
        """
        Run backtest over price series.
        At each step, generate signals, create contracts, find opportunities,
        then check if the trade would have won after forecast_horizon periods.
        """
        result = BacktestResult()
        equity = 0.0
        peak_equity = 0.0
        max_dd = 0.0

        if len(prices) < self.lookback + self.forecast_horizon + 10:
            logger.warning("Insufficient data for backtest")
            return result

        step = max(1, self.forecast_horizon // 4)  # Don't overlap too much

        for i in range(self.lookback, len(prices) - self.forecast_horizon, step):
            window = prices[i - self.lookback:i]
            current_price = prices[i]
            future_price = prices[i + self.forecast_horizon]

            # Generate signals
            signals = generate_signals(window)

            # Check directional accuracy
            actual_direction = "higher" if future_price > current_price else "lower"
            result.direction_total += 1
            if signals.direction == actual_direction:
                result.direction_correct += 1

            # Create simulated contracts around current price
            strikes = [
                current_price - 1000,
                current_price - 500,
                current_price,
                current_price + 500,
                current_price + 1000,
            ]

            for strike in strikes:
                # Simulate reasonable market prices
                distance_pct = (strike - current_price) / current_price
                import math
                base_yes = 1 / (1 + math.exp(distance_pct * 15))
                base_yes = np.clip(base_yes, 0.05, 0.95)

                contract = KalshiContract(
                    contract_id=f"BT-{i}-{strike:.0f}",
                    title=f"BTC above ${strike:,.0f}",
                    direction="higher",
                    strike_price=strike,
                    expiry="",
                    yes_price=round(base_yes, 2),
                    no_price=round(1 - base_yes, 2),
                )

                rec = self.analyzer.analyze_contract(
                    contract, current_price, window,
                    signals.direction, signals.confidence,
                )

                if rec and rec.ev >= min_ev:
                    # Evaluate outcome
                    btc_above = future_price >= strike
                    if rec.side == "YES":
                        won = btc_above
                        pnl = (1.0 - rec.contract.yes_price) if won else -rec.contract.yes_price
                    else:
                        won = not btc_above
                        pnl = (1.0 - rec.contract.no_price) if won else -rec.contract.no_price

                    result.total_trades += 1
                    if won:
                        result.wins += 1
                    else:
                        result.losses += 1

                    result.total_pnl += pnl
                    result.avg_ev += rec.ev
                    result.avg_edge += abs(rec.edge)

                    equity += pnl
                    result.equity_curve.append(round(equity, 4))
                    peak_equity = max(peak_equity, equity)
                    dd = peak_equity - equity
                    max_dd = max(max_dd, dd)

        # Finalize stats
        if result.total_trades > 0:
            result.win_rate = round(result.wins / result.total_trades * 100, 1)
            result.avg_ev = round(result.avg_ev / result.total_trades, 4)
            result.avg_edge = round(result.avg_edge / result.total_trades, 4)
            result.total_pnl = round(result.total_pnl, 4)
            result.max_drawdown = round(max_dd, 4)

            # Simplified Sharpe: mean return / std return
            if len(result.equity_curve) > 1:
                returns = np.diff(result.equity_curve)
                if np.std(returns) > 0:
                    result.sharpe_ratio = round(
                        np.mean(returns) / np.std(returns) * np.sqrt(252), 2
                    )

        if result.direction_total > 0:
            result.direction_accuracy = round(
                result.direction_correct / result.direction_total * 100, 1
            )

        return result
