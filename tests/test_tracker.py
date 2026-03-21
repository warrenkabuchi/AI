"""
Tests for ProfitTracker - profitability tracking and statistics.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from btc_tracker.tracker import ProfitTracker, TradeRecord
from btc_tracker.kalshi import KalshiContract, TradeRecommendation


def make_rec(side="YES", direction="higher", strike=90000,
             yes_price=0.40, no_price=0.60, model_prob=0.55,
             ev=0.08, edge=0.10, confidence="high"):
    contract = KalshiContract(
        contract_id=f"BTC-{strike}-{direction.upper()}-24H",
        title=f"BTC {'above' if direction == 'higher' else 'below'} ${strike:,}?",
        direction=direction,
        strike_price=strike,
        expiry="2026-03-22T00:00:00Z",
        yes_price=yes_price,
        no_price=no_price,
        volume=500,
    )
    return TradeRecommendation(
        contract=contract,
        side=side,
        model_probability=model_prob,
        market_probability=yes_price,
        edge=edge,
        ev=ev,
        confidence=confidence,
        reasoning="Test",
    )


@pytest.fixture
def tracker(tmp_path):
    with patch("btc_tracker.tracker.DATA_DIR", tmp_path):
        t = ProfitTracker()
        yield t


@pytest.fixture
def loaded_tracker(tracker):
    """Tracker with a mix of trades recorded."""
    tracker.record_trade(make_rec(confidence="high", ev=0.12, edge=0.15), btc_price=89000)
    tracker.record_trade(make_rec(confidence="medium", ev=0.06, edge=0.08), btc_price=89500)
    tracker.record_trade(make_rec(confidence="low", ev=0.03, edge=0.04), btc_price=88500)
    tracker.record_trade(
        make_rec(side="NO", direction="higher", strike=95000,
                 yes_price=0.80, no_price=0.20, confidence="high",
                 ev=0.10, edge=0.12),
        btc_price=89000,
    )
    return tracker


class TestRecordTrade:
    def test_records_trade(self, tracker):
        rec = make_rec()
        trade = tracker.record_trade(rec, btc_price=89000)
        assert trade.contract_id == rec.contract.contract_id
        assert trade.side == "YES"
        assert trade.entry_price == 0.40
        assert trade.outcome == "pending"
        assert trade.pnl == 0.0

    def test_increments_pending_count(self, tracker):
        tracker.record_trade(make_rec(), btc_price=89000)
        tracker.record_trade(make_rec(), btc_price=89000)
        stats = tracker.get_stats()
        assert stats["pending"] == 2
        assert stats["total_trades"] == 2

    def test_records_btc_price(self, tracker):
        tracker.record_trade(make_rec(), btc_price=87654)
        trades = tracker.get_recent_trades(1)
        assert trades[0]["btc_price_at_entry"] == 87654

    def test_records_no_side_entry_price(self, tracker):
        rec = make_rec(side="NO", yes_price=0.70, no_price=0.30)
        trade = tracker.record_trade(rec, btc_price=89000)
        assert trade.entry_price == 0.30

    def test_persists_to_disk(self, tmp_path):
        with patch("btc_tracker.tracker.DATA_DIR", tmp_path):
            t1 = ProfitTracker()
            t1.record_trade(make_rec(), btc_price=89000)

            t2 = ProfitTracker()
            assert len(t2.trades) == 1
            assert t2.trades[0].side == "YES"


class TestResolveTrade:
    def test_resolve_win_yes_higher_above_strike(self, tracker):
        """YES on 'BTC above $90k' wins when BTC > $90k."""
        tracker.record_trade(
            make_rec(side="YES", direction="higher", strike=90000, yes_price=0.40),
            btc_price=89000,
        )
        tracker.resolve_trade(0, final_btc_price=91000)
        assert tracker.trades[0].outcome == "win"
        assert tracker.trades[0].pnl == pytest.approx(0.60)  # 1.0 - 0.40

    def test_resolve_loss_yes_higher_below_strike(self, tracker):
        """YES on 'BTC above $90k' loses when BTC < $90k."""
        tracker.record_trade(
            make_rec(side="YES", direction="higher", strike=90000, yes_price=0.40),
            btc_price=89000,
        )
        tracker.resolve_trade(0, final_btc_price=88000)
        assert tracker.trades[0].outcome == "loss"
        assert tracker.trades[0].pnl == pytest.approx(-0.40)

    def test_resolve_win_no_higher_below_strike(self, tracker):
        """NO on 'BTC above $90k' wins when BTC < $90k."""
        tracker.record_trade(
            make_rec(side="NO", direction="higher", strike=90000,
                     yes_price=0.70, no_price=0.30),
            btc_price=89000,
        )
        tracker.resolve_trade(0, final_btc_price=88000)
        assert tracker.trades[0].outcome == "win"
        assert tracker.trades[0].pnl == pytest.approx(0.70)  # 1.0 - 0.30

    def test_resolve_loss_no_higher_above_strike(self, tracker):
        """NO on 'BTC above $90k' loses when BTC > $90k."""
        tracker.record_trade(
            make_rec(side="NO", direction="higher", strike=90000,
                     yes_price=0.70, no_price=0.30),
            btc_price=89000,
        )
        tracker.resolve_trade(0, final_btc_price=95000)
        assert tracker.trades[0].outcome == "loss"
        assert tracker.trades[0].pnl == pytest.approx(-0.30)

    def test_resolve_win_yes_lower_below_strike(self, tracker):
        """YES on 'BTC below $90k' wins when BTC < $90k."""
        tracker.record_trade(
            make_rec(side="YES", direction="lower", strike=90000, yes_price=0.50),
            btc_price=91000,
        )
        tracker.resolve_trade(0, final_btc_price=88000)
        assert tracker.trades[0].outcome == "win"

    def test_does_not_re_resolve(self, tracker):
        tracker.record_trade(make_rec(yes_price=0.40), btc_price=89000)
        tracker.resolve_trade(0, final_btc_price=91000)
        original_pnl = tracker.trades[0].pnl
        # Try to resolve again
        tracker.resolve_trade(0, final_btc_price=80000)
        # Should not change
        assert tracker.trades[0].pnl == original_pnl
        assert tracker.trades[0].outcome == "win"


class TestResolveAllPending:
    def test_resolves_all(self, tracker):
        for i in range(5):
            tracker.record_trade(make_rec(), btc_price=89000)
        tracker.resolve_all_pending(current_btc_price=91000)
        stats = tracker.get_stats()
        assert stats["pending"] == 0
        assert stats["resolved"] == 5

    def test_does_not_touch_already_resolved(self, tracker):
        tracker.record_trade(make_rec(yes_price=0.40), btc_price=89000)
        tracker.record_trade(make_rec(yes_price=0.60), btc_price=89000)
        # Resolve first manually
        tracker.resolve_trade(0, final_btc_price=91000)
        pnl_first = tracker.trades[0].pnl

        # Now resolve all pending
        tracker.resolve_all_pending(current_btc_price=85000)

        # First trade unchanged
        assert tracker.trades[0].pnl == pnl_first
        # Second trade resolved
        assert tracker.trades[1].outcome != "pending"


class TestStats:
    def test_empty_stats(self, tracker):
        stats = tracker.get_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0
        assert stats["total_pnl"] == 0

    def test_win_rate_calculation(self, tracker):
        # 3 wins, 2 losses
        for i in range(5):
            tracker.record_trade(make_rec(yes_price=0.40), btc_price=89000)
        # Resolve: first 3 as wins (above strike), last 2 as losses
        for i in range(3):
            tracker.resolve_trade(i, final_btc_price=91000)
        for i in range(3, 5):
            tracker.resolve_trade(i, final_btc_price=85000)

        stats = tracker.get_stats()
        assert stats["wins"] == 3
        assert stats["losses"] == 2
        assert stats["win_rate"] == 60.0

    def test_total_pnl_tracking(self, tracker):
        tracker.record_trade(make_rec(yes_price=0.40), btc_price=89000)
        tracker.record_trade(make_rec(yes_price=0.60), btc_price=89000)
        # Win: +0.60, Loss: -0.60
        tracker.resolve_trade(0, final_btc_price=91000)  # win
        tracker.resolve_trade(1, final_btc_price=85000)  # loss

        stats = tracker.get_stats()
        assert stats["total_pnl"] == pytest.approx(0.0)  # 0.60 + (-0.60)

    def test_confidence_breakdown(self, loaded_tracker):
        loaded_tracker.resolve_all_pending(current_btc_price=91000)
        stats = loaded_tracker.get_stats()

        assert "by_confidence" in stats
        assert stats["by_confidence"]["high"]["total"] == 2
        assert stats["by_confidence"]["medium"]["total"] == 1
        assert stats["by_confidence"]["low"]["total"] == 1

    def test_avg_edge_winners_vs_losers(self, tracker):
        # High edge trade that wins
        tracker.record_trade(
            make_rec(edge=0.20, yes_price=0.30), btc_price=89000
        )
        # Low edge trade that loses
        tracker.record_trade(
            make_rec(edge=0.05, yes_price=0.90), btc_price=89000
        )
        tracker.resolve_trade(0, final_btc_price=91000)  # win
        tracker.resolve_trade(1, final_btc_price=85000)  # loss

        stats = tracker.get_stats()
        assert stats["avg_edge_winners"] == pytest.approx(0.20)
        assert stats["avg_edge_losers"] == pytest.approx(0.05)

    def test_pnl_by_confidence(self, loaded_tracker):
        loaded_tracker.resolve_all_pending(current_btc_price=91000)
        stats = loaded_tracker.get_stats()

        # High confidence trades include a YES higher (win) and a NO higher (loss at 91k)
        high_pnl = stats["by_confidence"]["high"]["pnl"]
        # Just verify it's a number, not the exact value
        assert isinstance(high_pnl, float)


class TestGetRecentTrades:
    def test_returns_recent(self, loaded_tracker):
        trades = loaded_tracker.get_recent_trades(2)
        assert len(trades) == 2

    def test_returns_all_if_fewer(self, tracker):
        tracker.record_trade(make_rec(), btc_price=89000)
        trades = tracker.get_recent_trades(50)
        assert len(trades) == 1

    def test_returns_dicts(self, tracker):
        tracker.record_trade(make_rec(), btc_price=89000)
        trades = tracker.get_recent_trades(1)
        assert isinstance(trades[0], dict)
        assert "contract_id" in trades[0]
        assert "pnl" in trades[0]
        assert "outcome" in trades[0]


class TestPersistence:
    def test_pnl_persists(self, tmp_path):
        with patch("btc_tracker.tracker.DATA_DIR", tmp_path):
            t1 = ProfitTracker()
            t1.record_trade(make_rec(yes_price=0.40), btc_price=89000)
            t1.resolve_trade(0, final_btc_price=91000)

            t2 = ProfitTracker()
            assert t2.total_pnl == pytest.approx(0.60)
            assert t2.win_count == 1

    def test_stats_persist(self, tmp_path):
        with patch("btc_tracker.tracker.DATA_DIR", tmp_path):
            t1 = ProfitTracker()
            for _ in range(3):
                t1.record_trade(make_rec(yes_price=0.40), btc_price=89000)
            t1.resolve_all_pending(91000)

            t2 = ProfitTracker()
            stats = t2.get_stats()
            assert stats["wins"] == 3
            assert stats["total_pnl"] == pytest.approx(1.80)


class TestEdgeCases:
    def test_resolve_out_of_bounds_index(self, tracker):
        tracker.record_trade(make_rec(), btc_price=89000)
        tracker.resolve_trade(999, final_btc_price=91000)  # Should not crash
        assert tracker.trades[0].outcome == "pending"

    def test_pnl_is_rounded(self, tracker):
        tracker.record_trade(make_rec(yes_price=0.33), btc_price=89000)
        tracker.resolve_trade(0, final_btc_price=91000)
        # 1.0 - 0.33 = 0.67, should be rounded to 4 decimal places
        assert tracker.trades[0].pnl == round(0.67, 4)
