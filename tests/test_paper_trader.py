"""
Tests for PaperTrader - virtual brokerage simulation.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from btc_tracker.paper_trader import PaperTrader, PaperPosition


@pytest.fixture
def trader(tmp_path):
    """Create a PaperTrader with temp storage."""
    with patch("btc_tracker.paper_trader.DATA_DIR", tmp_path):
        t = PaperTrader(starting_balance=1000.0)
        yield t


@pytest.fixture
def funded_trader(trader):
    """Trader with some open positions."""
    trader.place_order(
        ticker="BTC-90000-HIGHER-24H",
        title="BTC above $90,000?",
        side="yes",
        direction="higher",
        strike_price=90000,
        count=2,
        price=0.60,
        btc_price=89000,
        ev=0.08,
        edge=0.10,
        confidence="high",
    )
    trader.place_order(
        ticker="BTC-85000-HIGHER-24H",
        title="BTC above $85,000?",
        side="no",
        direction="higher",
        strike_price=85000,
        count=3,
        price=0.25,
        btc_price=89000,
        ev=0.05,
        edge=0.07,
        confidence="medium",
    )
    return trader


class TestPlaceOrder:
    def test_basic_buy(self, trader):
        success, info = trader.place_order(
            ticker="BTC-90000-HIGHER-24H",
            title="BTC above $90,000?",
            side="yes",
            direction="higher",
            strike_price=90000,
            count=5,
            price=0.40,
            btc_price=89000,
        )
        assert success is True
        assert info["order_id"].startswith("paper-")
        assert info["count"] == 5
        assert info["cost"] == 2.0  # 5 * 0.40
        assert info["balance_after"] == 998.0  # 1000 - 2.0

    def test_deducts_cash(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=10, price=0.50, btc_price=89000,
        )
        bal = trader.get_balance()
        assert bal["balance"] == 995.0  # 1000 - 5.0

    def test_insufficient_balance(self, trader):
        success, info = trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=10000, price=0.50, btc_price=89000,
        )
        assert success is False
        assert "Insufficient" in info["error"]

    def test_creates_fill(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.35, btc_price=89000,
        )
        fills = trader.get_fills()
        assert len(fills) == 1
        assert fills[0]["ticker"] == "T1"
        assert fills[0]["action"] == "buy"
        assert fills[0]["price"] == 0.35

    def test_creates_position(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=3, price=0.40, btc_price=89000,
        )
        positions = trader.get_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "T1"
        assert positions[0]["count"] == 3
        assert positions[0]["avg_price"] == 0.40
        assert positions[0]["cost"] == 1.20

    def test_averages_into_existing_position(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.40, btc_price=89000,
        )
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.60, btc_price=89500,
        )
        positions = trader.get_positions()
        assert len(positions) == 1
        assert positions[0]["count"] == 4
        assert positions[0]["avg_price"] == pytest.approx(0.50)
        assert positions[0]["cost"] == pytest.approx(2.0)

    def test_separate_positions_for_different_sides(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.40, btc_price=89000,
        )
        trader.place_order(
            ticker="T1", title="T1", side="no", direction="higher",
            strike_price=90000, count=1, price=0.60, btc_price=89000,
        )
        positions = trader.get_positions()
        assert len(positions) == 2

    def test_increments_trade_count(self, trader):
        for i in range(5):
            trader.place_order(
                ticker=f"T{i}", title=f"T{i}", side="yes", direction="higher",
                strike_price=90000, count=1, price=0.10, btc_price=89000,
            )
        stats = trader.get_full_stats()
        assert stats["total_trades"] == 5


class TestSettlement:
    def test_win_yes_higher_btc_above_strike(self, trader):
        """YES on 'higher' wins when BTC is above strike."""
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.40, btc_price=89000,
        )
        settled = trader.settle_positions(current_btc_price=91000)
        assert len(settled) == 1
        assert settled[0].status == "settled_win"
        assert settled[0].pnl == pytest.approx(1.20)  # 2*1.0 - 2*0.40

    def test_loss_yes_higher_btc_below_strike(self, trader):
        """YES on 'higher' loses when BTC is below strike."""
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.40, btc_price=89000,
        )
        settled = trader.settle_positions(current_btc_price=88000)
        assert len(settled) == 1
        assert settled[0].status == "settled_loss"
        assert settled[0].pnl == pytest.approx(-0.80)  # lost 2*0.40

    def test_win_no_higher_btc_below_strike(self, trader):
        """NO on 'higher' wins when BTC is below strike."""
        trader.place_order(
            ticker="T1", title="T1", side="no", direction="higher",
            strike_price=90000, count=3, price=0.30, btc_price=89000,
        )
        settled = trader.settle_positions(current_btc_price=88000)
        assert len(settled) == 1
        assert settled[0].status == "settled_win"
        assert settled[0].pnl == pytest.approx(2.10)  # 3*1.0 - 3*0.30

    def test_loss_no_higher_btc_above_strike(self, trader):
        """NO on 'higher' loses when BTC is above strike."""
        trader.place_order(
            ticker="T1", title="T1", side="no", direction="higher",
            strike_price=90000, count=3, price=0.30, btc_price=89000,
        )
        settled = trader.settle_positions(current_btc_price=95000)
        assert len(settled) == 1
        assert settled[0].status == "settled_loss"
        assert settled[0].pnl == pytest.approx(-0.90)

    def test_win_yes_lower_btc_below_strike(self, trader):
        """YES on 'lower' wins when BTC is below strike."""
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="lower",
            strike_price=90000, count=1, price=0.50, btc_price=91000,
        )
        settled = trader.settle_positions(current_btc_price=88000)
        assert len(settled) == 1
        assert settled[0].status == "settled_win"
        assert settled[0].pnl == pytest.approx(0.50)

    def test_loss_yes_lower_btc_above_strike(self, trader):
        """YES on 'lower' loses when BTC is above strike."""
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="lower",
            strike_price=90000, count=1, price=0.50, btc_price=91000,
        )
        settled = trader.settle_positions(current_btc_price=92000)
        assert len(settled) == 1
        assert settled[0].status == "settled_loss"

    def test_settlement_credits_cash(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.40, btc_price=89000,
        )
        # Cash after buy: 1000 - 0.80 = 999.20
        trader.settle_positions(current_btc_price=91000)  # WIN
        bal = trader.get_balance()
        # Cash: 999.20 + 2*1.0 = 1001.20
        assert bal["balance"] == pytest.approx(1001.20)

    def test_settlement_loss_no_credit(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=2, price=0.40, btc_price=89000,
        )
        trader.settle_positions(current_btc_price=85000)  # LOSS
        bal = trader.get_balance()
        # Cash: 999.20 + 0 = 999.20 (no payout)
        assert bal["balance"] == pytest.approx(999.20)

    def test_does_not_resettle(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.40, btc_price=89000,
        )
        trader.settle_positions(current_btc_price=91000)
        settled2 = trader.settle_positions(current_btc_price=91000)
        assert len(settled2) == 0  # Already settled

    def test_records_equity_curve(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.40, btc_price=89000,
        )
        trader.settle_positions(current_btc_price=91000)
        stats = trader.get_full_stats()
        assert len(stats["equity_curve"]) >= 1

    def test_creates_settlement_fill(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.40, btc_price=89000,
        )
        trader.settle_positions(current_btc_price=91000)
        fills = trader.get_fills()
        assert len(fills) == 2  # buy + settlement
        assert fills[1]["action"] == "settlement"


class TestMultiplePositions:
    def test_settle_mixed_outcomes(self, funded_trader):
        """Two positions: one wins, one loses."""
        # Position 1: YES higher $90k, BTC at $91k -> WIN
        # Position 2: NO higher $85k, BTC at $91k -> LOSS (NO loses when above)
        settled = funded_trader.settle_positions(current_btc_price=91000)
        assert len(settled) == 2

        wins = [s for s in settled if s.status == "settled_win"]
        losses = [s for s in settled if s.status == "settled_loss"]
        assert len(wins) == 1
        assert len(losses) == 1

    def test_portfolio_value_with_open(self, funded_trader):
        bal = funded_trader.get_balance()
        # Starting: 1000, spent: 2*0.60 + 3*0.25 = 1.20 + 0.75 = 1.95
        assert bal["balance"] == pytest.approx(998.05)
        # Portfolio includes open position cost
        assert bal["portfolio_value"] == pytest.approx(1000.0)

    def test_stats_after_settlement(self, funded_trader):
        funded_trader.settle_positions(current_btc_price=91000)
        stats = funded_trader.get_full_stats()
        assert stats["wins"] + stats["losses"] == 2
        assert stats["total_trades"] == 2
        assert stats["win_rate"] > 0

    def test_confidence_breakdown(self, funded_trader):
        funded_trader.settle_positions(current_btc_price=91000)
        stats = funded_trader.get_full_stats()
        assert "high" in stats["by_confidence"]
        assert "medium" in stats["by_confidence"]
        assert stats["by_confidence"]["high"]["trades"] == 1
        assert stats["by_confidence"]["medium"]["trades"] == 1


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        with patch("btc_tracker.paper_trader.DATA_DIR", tmp_path):
            t1 = PaperTrader(starting_balance=500.0)
            t1.place_order(
                ticker="T1", title="T1", side="yes", direction="higher",
                strike_price=90000, count=2, price=0.30, btc_price=89000,
            )

            # Create new instance - should load from disk
            t2 = PaperTrader(starting_balance=500.0)
            assert t2.get_balance()["balance"] == pytest.approx(499.40)
            positions = t2.get_positions()
            assert len(positions) == 1
            assert positions[0]["ticker"] == "T1"

    def test_persistence_survives_settlement(self, tmp_path):
        with patch("btc_tracker.paper_trader.DATA_DIR", tmp_path):
            t1 = PaperTrader(starting_balance=500.0)
            t1.place_order(
                ticker="T1", title="T1", side="yes", direction="higher",
                strike_price=90000, count=1, price=0.40, btc_price=89000,
            )
            t1.settle_positions(current_btc_price=91000)

            t2 = PaperTrader(starting_balance=500.0)
            stats = t2.get_full_stats()
            assert stats["wins"] == 1
            assert stats["total_pnl"] == pytest.approx(0.60)


class TestReset:
    def test_reset_clears_everything(self, funded_trader):
        funded_trader.settle_positions(current_btc_price=91000)
        funded_trader.reset(starting_balance=2000.0)

        stats = funded_trader.get_full_stats()
        assert stats["cash_balance"] == 2000.0
        assert stats["total_trades"] == 0
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert len(funded_trader.get_positions()) == 0
        assert len(funded_trader.get_fills()) == 0


class TestEdgeCases:
    def test_btc_exactly_at_strike(self, trader):
        """BTC exactly at strike counts as 'above'."""
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.50, btc_price=89000,
        )
        settled = trader.settle_positions(current_btc_price=90000)
        assert settled[0].status == "settled_win"

    def test_very_small_price(self, trader):
        success, _ = trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.01, btc_price=89000,
        )
        assert success is True
        assert trader.get_balance()["balance"] == pytest.approx(999.99)

    def test_max_payout_is_one_dollar(self, trader):
        trader.place_order(
            ticker="T1", title="T1", side="yes", direction="higher",
            strike_price=90000, count=1, price=0.01, btc_price=89000,
        )
        settled = trader.settle_positions(current_btc_price=100000)
        # Payout is $1 per contract, cost was $0.01
        assert settled[0].pnl == pytest.approx(0.99)
