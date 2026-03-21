"""
Profitability and trade tracker.
Records all recommendations, tracks which would have won/lost,
and maintains running P&L statistics.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class TradeRecord:
    timestamp: str
    contract_id: str
    title: str
    side: str  # YES or NO
    entry_price: float
    model_probability: float
    market_probability: float
    edge: float
    ev: float
    confidence: str
    btc_price_at_entry: float
    strike_price: float
    direction: str
    outcome: str = "pending"  # "win", "loss", "pending"
    pnl: float = 0.0


class ProfitTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.trades = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.pending_count = 0
        DATA_DIR.mkdir(exist_ok=True)
        self._load()

    def _load(self):
        path = DATA_DIR / "trades.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.trades = [TradeRecord(**t) for t in data]
                self._recalculate()
                logger.info(f"Loaded {len(self.trades)} historical trades")
            except Exception as e:
                logger.warning(f"Failed to load trades: {e}")

    def _save(self):
        path = DATA_DIR / "trades.json"
        try:
            with open(path, "w") as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save trades: {e}")

    def _recalculate(self):
        self.win_count = sum(1 for t in self.trades if t.outcome == "win")
        self.loss_count = sum(1 for t in self.trades if t.outcome == "loss")
        self.pending_count = sum(1 for t in self.trades if t.outcome == "pending")
        self.total_pnl = sum(t.pnl for t in self.trades)

    def record_trade(self, recommendation, btc_price):
        """Record a new trade recommendation."""
        with self._lock:
            trade = TradeRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                contract_id=recommendation.contract.contract_id,
                title=recommendation.contract.title,
                side=recommendation.side,
                entry_price=(recommendation.contract.yes_price
                             if recommendation.side == "YES"
                             else recommendation.contract.no_price),
                model_probability=recommendation.model_probability,
                market_probability=recommendation.market_probability,
                edge=recommendation.edge,
                ev=recommendation.ev,
                confidence=recommendation.confidence,
                btc_price_at_entry=btc_price,
                strike_price=recommendation.contract.strike_price,
                direction=recommendation.contract.direction,
            )
            self.trades.append(trade)
            self.pending_count += 1
            self._save()
            return trade

    def resolve_trade(self, trade_index, final_btc_price):
        """Resolve a pending trade based on final BTC price."""
        with self._lock:
            if trade_index >= len(self.trades):
                return
            trade = self.trades[trade_index]
            if trade.outcome != "pending":
                return

            # Did BTC end above or below strike?
            btc_above_strike = final_btc_price >= trade.strike_price

            if trade.direction == "higher":
                contract_settled_yes = btc_above_strike
            else:
                contract_settled_yes = not btc_above_strike

            # Determine if our trade won
            if trade.side == "YES":
                won = contract_settled_yes
                trade.pnl = (1.0 - trade.entry_price) if won else -trade.entry_price
            else:
                won = not contract_settled_yes
                trade.pnl = (1.0 - trade.entry_price) if won else -trade.entry_price

            trade.outcome = "win" if won else "loss"
            trade.pnl = round(trade.pnl, 4)

            self._recalculate()
            self._save()

    def resolve_all_pending(self, current_btc_price):
        """Resolve all pending trades (for demo/simulation)."""
        with self._lock:
            for i, trade in enumerate(self.trades):
                if trade.outcome == "pending":
                    pass  # Don't auto-resolve outside lock
        # Resolve outside inner lock calls
        indices = [i for i, t in enumerate(self.trades) if t.outcome == "pending"]
        for idx in indices:
            self.resolve_trade(idx, current_btc_price)

    def get_stats(self):
        """Get summary statistics."""
        with self._lock:
            total = self.win_count + self.loss_count
            win_rate = (self.win_count / total * 100) if total > 0 else 0

            # Stats by confidence level
            by_confidence = {}
            for conf in ["high", "medium", "low"]:
                conf_trades = [t for t in self.trades if t.confidence == conf]
                conf_resolved = [t for t in conf_trades if t.outcome != "pending"]
                conf_wins = sum(1 for t in conf_resolved if t.outcome == "win")
                by_confidence[conf] = {
                    "total": len(conf_trades),
                    "resolved": len(conf_resolved),
                    "wins": conf_wins,
                    "win_rate": (conf_wins / len(conf_resolved) * 100)
                              if conf_resolved else 0,
                    "pnl": round(sum(t.pnl for t in conf_trades), 4),
                }

            # Average edge on winning vs losing trades
            winning = [t for t in self.trades if t.outcome == "win"]
            losing = [t for t in self.trades if t.outcome == "loss"]

            return {
                "total_trades": len(self.trades),
                "resolved": total,
                "pending": self.pending_count,
                "wins": self.win_count,
                "losses": self.loss_count,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(self.total_pnl, 4),
                "avg_edge_winners": round(
                    sum(t.edge for t in winning) / len(winning), 4
                ) if winning else 0,
                "avg_edge_losers": round(
                    sum(t.edge for t in losing) / len(losing), 4
                ) if losing else 0,
                "by_confidence": by_confidence,
            }

    def get_recent_trades(self, n=20):
        """Get the most recent n trades."""
        with self._lock:
            return [asdict(t) for t in self.trades[-n:]]
