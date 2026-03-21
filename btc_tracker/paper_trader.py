"""
Paper trading engine.
Simulates a real brokerage account with virtual money - tracks balance,
positions, fills, settlements, and equity curve identically to live trading.
All state persists to disk so it survives restarts.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class PaperPosition:
    ticker: str
    title: str
    side: str           # "yes" or "no"
    direction: str      # "higher" or "lower"
    strike_price: float
    count: int
    avg_price: float    # entry price per contract (0-1)
    cost: float         # total cost
    opened_at: str
    btc_price_at_entry: float
    ev: float = 0.0
    edge: float = 0.0
    confidence: str = ""
    status: str = "open"     # "open", "settled_win", "settled_loss"
    settled_at: str = ""
    settlement_price: float = 0.0  # 1.0 if won, 0.0 if lost
    pnl: float = 0.0


@dataclass
class PaperFill:
    fill_id: str
    ticker: str
    side: str
    action: str          # "buy" or "sell"
    count: int
    price: float
    cost: float
    timestamp: str
    btc_price: float


@dataclass
class PaperAccount:
    starting_balance: float = 1000.0
    cash_balance: float = 1000.0
    positions: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    total_settled_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0


class PaperTrader:
    """
    Full paper trading simulator. Tracks everything a real account would:
    - Cash balance (decreases on buys, increases on settlements)
    - Open positions with entry prices
    - Fill history (every order)
    - Settlement when BTC crosses/doesn't cross strike
    - Equity curve over time
    - Win/loss stats, P&L tracking
    """

    def __init__(self, starting_balance=1000.0):
        self._lock = threading.Lock()
        self._path = DATA_DIR / "paper_account.json"
        DATA_DIR.mkdir(exist_ok=True)
        self.account = PaperAccount(starting_balance=starting_balance,
                                    cash_balance=starting_balance)
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self.account.starting_balance = data.get("starting_balance", 1000.0)
                self.account.cash_balance = data.get("cash_balance", 1000.0)
                self.account.total_settled_pnl = data.get("total_settled_pnl", 0.0)
                self.account.total_trades = data.get("total_trades", 0)
                self.account.wins = data.get("wins", 0)
                self.account.losses = data.get("losses", 0)
                self.account.equity_curve = data.get("equity_curve", [])
                self.account.positions = [
                    PaperPosition(**p) for p in data.get("positions", [])
                ]
                self.account.fills = [
                    PaperFill(**f) for f in data.get("fills", [])
                ]
                logger.info(
                    f"Paper account loaded: ${self.account.cash_balance:.2f} cash, "
                    f"{len(self._open_positions())} open positions"
                )
            except Exception as e:
                logger.warning(f"Failed to load paper account: {e}")

    def _save(self):
        try:
            data = {
                "starting_balance": self.account.starting_balance,
                "cash_balance": self.account.cash_balance,
                "total_settled_pnl": self.account.total_settled_pnl,
                "total_trades": self.account.total_trades,
                "wins": self.account.wins,
                "losses": self.account.losses,
                "equity_curve": self.account.equity_curve[-500:],
                "positions": [asdict(p) for p in self.account.positions],
                "fills": [asdict(f) for f in self.account.fills[-200:]],
            }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save paper account: {e}")

    def _open_positions(self):
        return [p for p in self.account.positions if p.status == "open"]

    def place_order(self, ticker, title, side, direction, strike_price,
                    count, price, btc_price, ev=0, edge=0, confidence=""):
        """
        Place a paper order. Deducts cost from cash balance.
        Returns (success, order_info).
        """
        with self._lock:
            cost = price * count
            if cost > self.account.cash_balance:
                return False, {"error": f"Insufficient balance: ${self.account.cash_balance:.2f} < ${cost:.2f}"}

            # Deduct cash
            self.account.cash_balance -= cost
            self.account.total_trades += 1

            # Create fill
            fill_id = str(uuid.uuid4())[:8]
            fill = PaperFill(
                fill_id=fill_id,
                ticker=ticker,
                side=side,
                action="buy",
                count=count,
                price=price,
                cost=cost,
                timestamp=datetime.now(timezone.utc).isoformat(),
                btc_price=btc_price,
            )
            self.account.fills.append(fill)

            # Check if we already have an open position on this ticker+side
            existing = None
            for p in self.account.positions:
                if p.ticker == ticker and p.side == side and p.status == "open":
                    existing = p
                    break

            if existing:
                # Average into existing position
                total_count = existing.count + count
                existing.avg_price = (
                    (existing.avg_price * existing.count + price * count) / total_count
                )
                existing.count = total_count
                existing.cost += cost
            else:
                pos = PaperPosition(
                    ticker=ticker,
                    title=title,
                    side=side,
                    direction=direction,
                    strike_price=strike_price,
                    count=count,
                    avg_price=price,
                    cost=cost,
                    opened_at=datetime.now(timezone.utc).isoformat(),
                    btc_price_at_entry=btc_price,
                    ev=ev,
                    edge=edge,
                    confidence=confidence,
                )
                self.account.positions.append(pos)

            self._save()

            logger.info(
                f"PAPER TRADE: BUY {side.upper()} {count}x {ticker} "
                f"@ ${price:.2f} (cost: ${cost:.2f}, balance: ${self.account.cash_balance:.2f})"
            )

            return True, {
                "order_id": f"paper-{fill_id}",
                "fill_id": fill_id,
                "ticker": ticker,
                "side": side,
                "count": count,
                "price": price,
                "cost": cost,
                "balance_after": self.account.cash_balance,
            }

    def settle_positions(self, current_btc_price):
        """
        Settle open positions based on current BTC price.
        For each position, check if BTC is above/below the strike
        and pay out accordingly.

        Returns list of settled positions with P&L.
        """
        with self._lock:
            settled = []
            for pos in self.account.positions:
                if pos.status != "open":
                    continue

                btc_above = current_btc_price >= pos.strike_price

                # Did the contract settle YES or NO?
                if pos.direction == "higher":
                    contract_yes = btc_above
                else:
                    contract_yes = not btc_above

                # Did our side win?
                if pos.side == "yes":
                    won = contract_yes
                else:
                    won = not contract_yes

                # Calculate payout
                if won:
                    payout = pos.count * 1.0  # $1 per contract
                    pos.settlement_price = 1.0
                    pos.pnl = payout - pos.cost
                    pos.status = "settled_win"
                    self.account.wins += 1
                else:
                    payout = 0.0
                    pos.settlement_price = 0.0
                    pos.pnl = -pos.cost
                    pos.status = "settled_loss"
                    self.account.losses += 1

                # Credit payout to cash
                self.account.cash_balance += payout
                self.account.total_settled_pnl += pos.pnl
                pos.settled_at = datetime.now(timezone.utc).isoformat()

                # Record fill for settlement
                self.account.fills.append(PaperFill(
                    fill_id=str(uuid.uuid4())[:8],
                    ticker=pos.ticker,
                    side=pos.side,
                    action="settlement",
                    count=pos.count,
                    price=pos.settlement_price,
                    cost=-payout,
                    timestamp=pos.settled_at,
                    btc_price=current_btc_price,
                ))

                settled.append(pos)
                logger.info(
                    f"PAPER SETTLED: {pos.ticker} {pos.side.upper()} -> "
                    f"{'WIN' if won else 'LOSS'} P&L: ${pos.pnl:+.2f} "
                    f"(BTC: ${current_btc_price:,.0f} vs strike ${pos.strike_price:,.0f})"
                )

            # Record equity snapshot
            total_equity = self.account.cash_balance + sum(
                p.cost for p in self._open_positions()
            )
            self.account.equity_curve.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": round(total_equity, 2),
                "cash": round(self.account.cash_balance, 2),
                "open_value": round(sum(p.cost for p in self._open_positions()), 2),
                "pnl": round(self.account.total_settled_pnl, 2),
            })

            if settled:
                self._save()

            return settled

    def get_balance(self):
        """Get balance like the real Kalshi API would return."""
        with self._lock:
            open_value = sum(p.cost for p in self._open_positions())
            return {
                "balance": round(self.account.cash_balance, 2),
                "portfolio_value": round(self.account.cash_balance + open_value, 2),
            }

    def get_positions(self):
        """Get open positions like the real API."""
        with self._lock:
            return [
                {
                    "ticker": p.ticker,
                    "title": p.title,
                    "side": p.side,
                    "direction": p.direction,
                    "strike_price": p.strike_price,
                    "count": p.count,
                    "avg_price": round(p.avg_price, 4),
                    "cost": round(p.cost, 2),
                    "opened_at": p.opened_at,
                    "btc_at_entry": p.btc_price_at_entry,
                    "ev": p.ev,
                    "edge": p.edge,
                    "confidence": p.confidence,
                }
                for p in self._open_positions()
            ]

    def get_fills(self, limit=50):
        """Get recent fills."""
        with self._lock:
            return [asdict(f) for f in self.account.fills[-limit:]]

    def get_settled(self, limit=50):
        """Get recently settled positions with outcomes."""
        with self._lock:
            settled = [p for p in self.account.positions if p.status != "open"]
            return [
                {
                    "ticker": p.ticker,
                    "title": p.title,
                    "side": p.side,
                    "direction": p.direction,
                    "strike_price": p.strike_price,
                    "count": p.count,
                    "avg_price": round(p.avg_price, 4),
                    "cost": round(p.cost, 2),
                    "pnl": round(p.pnl, 2),
                    "outcome": "WIN" if p.status == "settled_win" else "LOSS",
                    "opened_at": p.opened_at,
                    "settled_at": p.settled_at,
                    "btc_at_entry": p.btc_price_at_entry,
                }
                for p in settled[-limit:]
            ]

    def get_full_stats(self):
        """Complete paper trading statistics."""
        with self._lock:
            open_positions = self._open_positions()
            open_value = sum(p.cost for p in open_positions)
            total_equity = self.account.cash_balance + open_value
            total_resolved = self.account.wins + self.account.losses
            win_rate = (self.account.wins / total_resolved * 100) if total_resolved > 0 else 0

            # Stats by confidence
            all_settled = [p for p in self.account.positions if p.status != "open"]
            by_confidence = {}
            for conf in ["high", "medium", "low"]:
                conf_trades = [p for p in all_settled if p.confidence == conf]
                conf_wins = sum(1 for p in conf_trades if p.status == "settled_win")
                by_confidence[conf] = {
                    "trades": len(conf_trades),
                    "wins": conf_wins,
                    "win_rate": round(conf_wins / len(conf_trades) * 100, 1) if conf_trades else 0,
                    "pnl": round(sum(p.pnl for p in conf_trades), 2),
                }

            # Max drawdown from equity curve
            max_equity = self.account.starting_balance
            max_drawdown = 0.0
            for snap in self.account.equity_curve:
                eq = snap.get("equity", self.account.starting_balance)
                max_equity = max(max_equity, eq)
                dd = max_equity - eq
                max_drawdown = max(max_drawdown, dd)

            # Return on investment
            roi = ((total_equity - self.account.starting_balance)
                   / self.account.starting_balance * 100)

            return {
                "mode": "paper",
                "starting_balance": self.account.starting_balance,
                "cash_balance": round(self.account.cash_balance, 2),
                "open_positions_value": round(open_value, 2),
                "total_equity": round(total_equity, 2),
                "total_pnl": round(self.account.total_settled_pnl, 2),
                "roi": round(roi, 2),
                "total_trades": self.account.total_trades,
                "open_count": len(open_positions),
                "wins": self.account.wins,
                "losses": self.account.losses,
                "win_rate": round(win_rate, 1),
                "max_drawdown": round(max_drawdown, 2),
                "by_confidence": by_confidence,
                "equity_curve": [
                    {"t": s["timestamp"], "eq": s["equity"]}
                    for s in self.account.equity_curve[-100:]
                ],
            }

    def reset(self, starting_balance=None):
        """Reset the paper account (wipe all history)."""
        with self._lock:
            bal = starting_balance or self.account.starting_balance
            self.account = PaperAccount(starting_balance=bal, cash_balance=bal)
            self._save()
            logger.info(f"Paper account reset with ${bal:.2f}")
