"""
Auto-trading execution engine.
Takes trade recommendations and executes them on Kalshi
with risk management controls and safety limits.
"""

import logging
import threading
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from btc_tracker import config
from btc_tracker.kalshi_client import KalshiClient
from btc_tracker.paper_trader import PaperTrader

logger = logging.getLogger(__name__)

CONFIDENCE_RANKS = {"low": 0, "medium": 1, "high": 2}


@dataclass
class ExecutionResult:
    success: bool
    order_id: str = ""
    message: str = ""
    ticker: str = ""
    side: str = ""
    count: int = 0
    price: float = 0.0


class TradingExecutor:
    """
    Executes trades on Kalshi with safety controls.
    Enforces: position limits, daily loss limits, EV thresholds,
    confidence filters, and bet sizing via Kelly criterion.
    """

    def __init__(self):
        self.client = KalshiClient()
        self.paper = PaperTrader(starting_balance=config.PAPER_STARTING_BALANCE)
        self._lock = threading.Lock()
        self.daily_pnl = 0.0
        self.daily_exposure = 0.0
        self.open_position_count = 0
        self.last_reset = datetime.now(timezone.utc).date()
        self.execution_log = []

    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset:
            self.daily_pnl = 0.0
            self.daily_exposure = 0.0
            self.last_reset = today
            logger.info("Daily trading counters reset")

    def _check_safety(self, recommendation):
        """Run all safety checks before trading. Returns (ok, reason)."""
        self._reset_daily_if_needed()

        # Check if any trading mode is enabled
        if not config.AUTO_TRADE_ENABLED and not config.PAPER_TRADE_ENABLED:
            return False, "Auto-trading and paper trading are both disabled"

        # For live trading, require Kalshi API
        if not config.PAPER_TRADE_ENABLED and not self.client.is_configured:
            return False, "Kalshi API not configured and paper trading disabled"

        # Check minimum EV
        if recommendation.ev < config.MIN_EV_THRESHOLD:
            return False, f"EV {recommendation.ev:.3f} below threshold {config.MIN_EV_THRESHOLD}"

        # Check minimum confidence
        rec_rank = CONFIDENCE_RANKS.get(recommendation.confidence, -1)
        min_rank = CONFIDENCE_RANKS.get(config.MIN_CONFIDENCE, 1)
        if rec_rank < min_rank:
            return False, f"Confidence '{recommendation.confidence}' below minimum '{config.MIN_CONFIDENCE}'"

        # Check daily loss limit
        if self.daily_pnl <= config.DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"

        # Check daily exposure limit
        if self.daily_exposure >= config.MAX_DAILY_EXPOSURE:
            return False, f"Daily exposure limit reached: ${self.daily_exposure:.2f}"

        # Check open positions
        if self.open_position_count >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached: {self.open_position_count}"

        return True, "OK"

    def calculate_bet_size(self, recommendation):
        """
        Calculate number of contracts to buy using Kelly criterion
        with configurable fraction.
        """
        if recommendation.side == "YES":
            price = recommendation.contract.yes_price
        else:
            price = recommendation.contract.no_price

        if price <= 0 or price >= 1:
            return 1

        # Kelly fraction
        model_prob = recommendation.model_probability
        if recommendation.side == "NO":
            model_prob = 1 - model_prob

        b = (1 - price) / price  # Odds
        q = 1 - model_prob
        kelly = (model_prob * b - q) / b

        # Apply configured Kelly fraction
        adjusted_kelly = max(kelly * config.KELLY_FRACTION, 0)

        # Convert to dollar amount, capped at max bet
        bet_dollars = min(adjusted_kelly * config.MAX_BET_SIZE * 10, config.MAX_BET_SIZE)

        # Convert to number of contracts
        contracts = max(1, int(bet_dollars / price))

        # Ensure we don't exceed remaining daily exposure
        remaining = config.MAX_DAILY_EXPOSURE - self.daily_exposure
        max_contracts = max(1, int(remaining / price))
        contracts = min(contracts, max_contracts)

        return contracts

    def execute(self, recommendation, btc_price=None):
        """
        Execute a trade recommendation.
        Routes to paper trader or live Kalshi based on config.
        Returns ExecutionResult.
        """
        with self._lock:
            # Safety checks
            ok, reason = self._check_safety(recommendation)
            if not ok:
                logger.info(f"Trade blocked: {reason}")
                return ExecutionResult(success=False, message=reason)

            ticker = recommendation.contract.contract_id
            side = recommendation.side.lower()
            count = self.calculate_bet_size(recommendation)
            price = (recommendation.contract.yes_price
                     if side == "yes"
                     else recommendation.contract.no_price)

            # Route: paper trading or live
            if config.PAPER_TRADE_ENABLED:
                return self._execute_paper(recommendation, ticker, side,
                                           count, price, btc_price or 0)
            else:
                return self._execute_live(recommendation, ticker, side,
                                          count, price)

    def _execute_paper(self, recommendation, ticker, side, count, price, btc_price):
        """Execute via paper trader."""
        success, info = self.paper.place_order(
            ticker=ticker,
            title=recommendation.contract.title,
            side=side,
            direction=recommendation.contract.direction,
            strike_price=recommendation.contract.strike_price,
            count=count,
            price=price,
            btc_price=btc_price,
            ev=recommendation.ev,
            edge=recommendation.edge,
            confidence=recommendation.confidence,
        )

        if not success:
            return ExecutionResult(success=False, message=info.get("error", "Paper order failed"))

        order_id = info["order_id"]
        cost = price * count
        self.daily_exposure += cost
        self.open_position_count += 1

        result = ExecutionResult(
            success=True,
            order_id=order_id,
            message=f"[PAPER] {side.upper()} {count}x {ticker}",
            ticker=ticker,
            side=side.upper(),
            count=count,
            price=price,
        )

        self.execution_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "side": side.upper(),
            "count": count,
            "price": price,
            "cost": cost,
            "order_id": order_id,
            "ev": recommendation.ev,
            "edge": recommendation.edge,
            "confidence": recommendation.confidence,
            "mode": "paper",
        })

        logger.info(
            f"PAPER TRADE: {side.upper()} {count}x {ticker} "
            f"@ ${price:.2f} (EV: {recommendation.ev:+.3f})"
        )
        return result

    def _execute_live(self, recommendation, ticker, side, count, price):
        """Execute via live Kalshi API."""
        try:
            order = self.client.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=count,
                order_type="market",
            )

            order_id = order.get("order_id", "unknown")
            cost = price * count
            self.daily_exposure += cost
            self.open_position_count += 1

            result = ExecutionResult(
                success=True,
                order_id=order_id,
                message=f"Placed {side.upper()} {count}x {ticker}",
                ticker=ticker,
                side=side.upper(),
                count=count,
                price=price,
            )

            self.execution_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "side": side.upper(),
                "count": count,
                "price": price,
                "cost": cost,
                "order_id": order_id,
                "ev": recommendation.ev,
                "edge": recommendation.edge,
                "confidence": recommendation.confidence,
                "mode": "live",
            })

            logger.info(
                f"LIVE TRADE: {side.upper()} {count}x {ticker} "
                f"@ ${price:.2f} (EV: {recommendation.ev:+.3f})"
            )
            return result

        except Exception as e:
            msg = f"Order failed: {e}"
            logger.error(msg)
            return ExecutionResult(success=False, message=msg)

    def update_pnl(self, pnl_change):
        """Update daily P&L tracking."""
        with self._lock:
            self.daily_pnl += pnl_change

    def settle_paper_positions(self, current_btc_price):
        """Settle paper trading positions based on current BTC price."""
        if config.PAPER_TRADE_ENABLED:
            settled = self.paper.settle_positions(current_btc_price)
            for pos in settled:
                self.update_pnl(pos.pnl)
                if pos.status != "open":
                    self.open_position_count = max(0, self.open_position_count - 1)
            return settled
        return []

    def get_status(self):
        """Get executor status for dashboard."""
        self._reset_daily_if_needed()
        paper_stats = self.paper.get_full_stats() if config.PAPER_TRADE_ENABLED else None
        trading_mode = "paper" if config.PAPER_TRADE_ENABLED else (
            "live" if self.client.is_configured else "disabled"
        )
        return {
            "trading_mode": trading_mode,
            "auto_trade_enabled": config.AUTO_TRADE_ENABLED,
            "paper_trade_enabled": config.PAPER_TRADE_ENABLED,
            "kalshi_configured": self.client.is_configured,
            "kalshi_env": config.KALSHI_ENV,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_exposure": round(self.daily_exposure, 2),
            "max_daily_exposure": config.MAX_DAILY_EXPOSURE,
            "open_positions": self.open_position_count,
            "max_positions": config.MAX_OPEN_POSITIONS,
            "max_bet_size": config.MAX_BET_SIZE,
            "min_ev": config.MIN_EV_THRESHOLD,
            "min_confidence": config.MIN_CONFIDENCE,
            "daily_loss_limit": config.DAILY_LOSS_LIMIT,
            "recent_executions": self.execution_log[-10:],
            "paper_stats": paper_stats,
        }

    def get_portfolio(self):
        """Fetch portfolio data (paper or live)."""
        if config.PAPER_TRADE_ENABLED:
            return {
                "mode": "paper",
                "balance": self.paper.get_balance(),
                "positions": self.paper.get_positions(),
                "settled": self.paper.get_settled(20),
                "fills": self.paper.get_fills(30),
                "stats": self.paper.get_full_stats(),
            }
        if not self.client.is_configured:
            return None
        try:
            balance = self.client.get_balance()
            positions = self.client.get_positions()
            return {
                "mode": "live",
                "balance": balance,
                "positions": positions,
            }
        except Exception as e:
            logger.error(f"Portfolio fetch failed: {e}")
            return None
