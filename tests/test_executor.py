"""
Tests for TradingExecutor - safety checks, bet sizing, routing.
"""

import pytest
from unittest.mock import patch, MagicMock

from btc_tracker.kalshi import KalshiContract, TradeRecommendation
from btc_tracker.executor import TradingExecutor, ExecutionResult


def make_rec(ev=0.10, confidence="high", side="YES", yes_price=0.40,
             no_price=0.60, strike=90000, model_prob=0.55):
    contract = KalshiContract(
        contract_id="BTC-90000-HIGHER-24H",
        title="BTC above $90,000?",
        direction="higher",
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
        edge=model_prob - yes_price,
        ev=ev,
        confidence=confidence,
        reasoning="Test recommendation",
    )


@pytest.fixture
def executor(tmp_path):
    """Executor with paper trading enabled, mocked paths."""
    with patch("btc_tracker.paper_trader.DATA_DIR", tmp_path), \
         patch("btc_tracker.executor.config") as mock_cfg:
        mock_cfg.PAPER_TRADE_ENABLED = True
        mock_cfg.PAPER_STARTING_BALANCE = 1000.0
        mock_cfg.AUTO_TRADE_ENABLED = False
        mock_cfg.MAX_BET_SIZE = 5.00
        mock_cfg.MAX_DAILY_EXPOSURE = 50.00
        mock_cfg.MIN_EV_THRESHOLD = 0.05
        mock_cfg.MIN_CONFIDENCE = "medium"
        mock_cfg.MAX_OPEN_POSITIONS = 10
        mock_cfg.DAILY_LOSS_LIMIT = -25.00
        mock_cfg.KELLY_FRACTION = 0.25
        mock_cfg.KALSHI_ENV = "demo"
        mock_cfg.KALSHI_API_KEY_ID = ""
        mock_cfg.KALSHI_PRIVATE_KEY_PATH = ""
        mock_cfg.KALSHI_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"

        ex = TradingExecutor()
        yield ex


class TestSafetyChecks:
    def test_blocks_when_both_disabled(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.AUTO_TRADE_ENABLED = False
            cfg.PAPER_TRADE_ENABLED = False
            result = executor.execute(make_rec(), btc_price=89000)
            assert result.success is False
            assert "disabled" in result.message

    def test_blocks_low_ev(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.10
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            result = executor.execute(make_rec(ev=0.03), btc_price=89000)
            assert result.success is False
            assert "EV" in result.message

    def test_blocks_low_confidence(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "high"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            result = executor.execute(make_rec(confidence="low"), btc_price=89000)
            assert result.success is False
            assert "Confidence" in result.message

    def test_blocks_at_daily_loss_limit(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -25.0
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            executor.daily_pnl = -30.0
            result = executor.execute(make_rec(), btc_price=89000)
            assert result.success is False
            assert "loss limit" in result.message

    def test_blocks_at_max_exposure(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 10.0
            cfg.MAX_OPEN_POSITIONS = 100
            executor.daily_exposure = 15.0
            result = executor.execute(make_rec(), btc_price=89000)
            assert result.success is False
            assert "exposure" in result.message

    def test_blocks_at_max_positions(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 2
            executor.open_position_count = 3
            result = executor.execute(make_rec(), btc_price=89000)
            assert result.success is False
            assert "positions" in result.message


class TestPaperExecution:
    def test_routes_to_paper(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            cfg.MAX_BET_SIZE = 5.0
            cfg.KELLY_FRACTION = 0.25
            result = executor.execute(make_rec(), btc_price=89000)
            assert result.success is True
            assert "[PAPER]" in result.message

    def test_paper_execution_logs(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            cfg.MAX_BET_SIZE = 5.0
            cfg.KELLY_FRACTION = 0.25
            executor.execute(make_rec(), btc_price=89000)
            assert len(executor.execution_log) == 1
            assert executor.execution_log[0]["mode"] == "paper"

    def test_paper_increments_exposure(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            cfg.MAX_BET_SIZE = 5.0
            cfg.KELLY_FRACTION = 0.25
            executor.execute(make_rec(), btc_price=89000)
            assert executor.daily_exposure > 0
            assert executor.open_position_count == 1


class TestBetSizing:
    def test_returns_at_least_one(self, executor):
        rec = make_rec(model_prob=0.51, yes_price=0.50)
        count = executor.calculate_bet_size(rec)
        assert count >= 1

    def test_caps_at_max_bet(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.MAX_BET_SIZE = 2.0
            cfg.KELLY_FRACTION = 1.0
            cfg.MAX_DAILY_EXPOSURE = 1000
            executor.daily_exposure = 0
            rec = make_rec(model_prob=0.90, yes_price=0.10)
            count = executor.calculate_bet_size(rec)
            # At $0.10/contract, max $2 -> max 20 contracts
            assert count <= 20

    def test_respects_remaining_exposure(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.MAX_BET_SIZE = 100.0
            cfg.KELLY_FRACTION = 1.0
            cfg.MAX_DAILY_EXPOSURE = 5.0
            executor.daily_exposure = 4.5
            rec = make_rec(model_prob=0.90, yes_price=0.10)
            count = executor.calculate_bet_size(rec)
            # Only $0.50 remaining / $0.10 = 5 contracts max
            assert count <= 5


class TestSettlePaper:
    def test_settle_updates_pnl(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            cfg.MAX_BET_SIZE = 5.0
            cfg.KELLY_FRACTION = 0.25
            executor.execute(make_rec(strike=90000), btc_price=89000)
            executor.settle_paper_positions(91000)  # WIN
            assert executor.daily_pnl > 0

    def test_settle_decrements_positions(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.MIN_EV_THRESHOLD = 0.01
            cfg.MIN_CONFIDENCE = "low"
            cfg.DAILY_LOSS_LIMIT = -100
            cfg.MAX_DAILY_EXPOSURE = 1000
            cfg.MAX_OPEN_POSITIONS = 100
            cfg.MAX_BET_SIZE = 5.0
            cfg.KELLY_FRACTION = 0.25
            executor.execute(make_rec(strike=90000), btc_price=89000)
            assert executor.open_position_count == 1
            executor.settle_paper_positions(91000)
            assert executor.open_position_count == 0


class TestGetStatus:
    def test_status_includes_paper_stats(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            cfg.AUTO_TRADE_ENABLED = False
            cfg.KALSHI_ENV = "demo"
            cfg.MAX_DAILY_EXPOSURE = 50
            cfg.MAX_OPEN_POSITIONS = 10
            cfg.MAX_BET_SIZE = 5.0
            cfg.MIN_EV_THRESHOLD = 0.05
            cfg.MIN_CONFIDENCE = "medium"
            cfg.DAILY_LOSS_LIMIT = -25
            status = executor.get_status()
            assert status["trading_mode"] == "paper"
            assert status["paper_trade_enabled"] is True
            assert status["paper_stats"] is not None
            assert status["paper_stats"]["mode"] == "paper"

    def test_portfolio_returns_paper_data(self, executor):
        with patch("btc_tracker.executor.config") as cfg:
            cfg.PAPER_TRADE_ENABLED = True
            portfolio = executor.get_portfolio()
            assert portfolio["mode"] == "paper"
            assert "balance" in portfolio
            assert "positions" in portfolio
            assert "stats" in portfolio
