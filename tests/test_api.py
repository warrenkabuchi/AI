"""
Tests for Flask API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestStatusEndpoint:
    def test_returns_json(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "price" in data
        assert "history_points" in data


class TestPriceEndpoint:
    def test_returns_price(self, client):
        resp = client.get("/api/price")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current" in data
        assert "history" in data


class TestSignalsEndpoint:
    def test_returns_signals(self, client):
        resp = client.get("/api/signals")
        assert resp.status_code == 200
        # May be null if no analysis has run yet
        data = resp.get_json()
        assert data is None or "direction" in data


class TestOpportunitiesEndpoint:
    def test_returns_opportunities(self, client):
        resp = client.get("/api/opportunities")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recommendations" in data
        assert "contracts" in data


class TestTradesEndpoint:
    def test_returns_trades(self, client):
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recent" in data
        assert "stats" in data


class TestConfigEndpoint:
    def test_returns_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "kalshi_configured" in data
        assert "paper_trade_enabled" in data
        assert "max_bet_size" in data


class TestExecutorEndpoint:
    def test_returns_executor_status(self, client):
        resp = client.get("/api/executor")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "trading_mode" in data
        assert "daily_pnl" in data
        assert "recent_executions" in data


class TestPortfolioEndpoint:
    def test_returns_portfolio(self, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 200
        data = resp.get_json()
        # Either paper portfolio or error (if not configured)
        assert "mode" in data or "error" in data


class TestStrategyEndpoint:
    def test_returns_strategy(self, client):
        resp = client.get("/api/strategy")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "probabilities" in data
        assert "weights" in data


class TestChartEndpoint:
    def test_returns_chart_data(self, client):
        resp = client.get("/api/chart")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "prices" in data


class TestDashboard:
    def test_dashboard_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"BTC" in resp.data
