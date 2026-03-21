"""
Tests for config module - env loading and typed access.
"""

import os
import pytest
from unittest.mock import patch


class TestDefaults:
    def test_default_values(self):
        # Clear any env vars that might interfere
        env_overrides = {
            "KALSHI_API_KEY_ID": "",
            "KALSHI_PRIVATE_KEY_PATH": "",
            "KALSHI_ENV": "demo",
            "PAPER_TRADE_ENABLED": "true",
            "PAPER_STARTING_BALANCE": "1000.0",
            "AUTO_TRADE_ENABLED": "false",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            # Re-import to pick up patched env
            import importlib
            import btc_tracker.config as cfg
            importlib.reload(cfg)

            assert cfg.KALSHI_ENV == "demo"
            assert cfg.PAPER_TRADE_ENABLED is True
            assert cfg.PAPER_STARTING_BALANCE == 1000.0
            assert cfg.AUTO_TRADE_ENABLED is False
            assert cfg.MAX_BET_SIZE == 5.0
            assert cfg.MAX_DAILY_EXPOSURE == 50.0
            assert cfg.MIN_EV_THRESHOLD == 0.05
            assert cfg.MIN_CONFIDENCE == "medium"
            assert cfg.MAX_OPEN_POSITIONS == 10
            assert cfg.DAILY_LOSS_LIMIT == -25.0
            assert cfg.KELLY_FRACTION == 0.25
            assert cfg.PORT == 5050
            assert cfg.UPDATE_INTERVAL == 60
            assert cfg.LOG_LEVEL == "INFO"


class TestKalshiConfigured:
    def test_not_configured_with_empty_keys(self):
        with patch.dict(os.environ, {"KALSHI_API_KEY_ID": "", "KALSHI_PRIVATE_KEY_PATH": ""}):
            import importlib
            import btc_tracker.config as cfg
            importlib.reload(cfg)
            assert cfg.is_kalshi_configured() is False

    def test_configured_with_both_keys(self):
        with patch.dict(os.environ, {
            "KALSHI_API_KEY_ID": "some-key",
            "KALSHI_PRIVATE_KEY_PATH": "/path/to/key.pem",
        }):
            import importlib
            import btc_tracker.config as cfg
            importlib.reload(cfg)
            assert cfg.is_kalshi_configured() is True


class TestBaseUrl:
    def test_demo_url(self):
        with patch.dict(os.environ, {"KALSHI_ENV": "demo"}):
            import importlib
            import btc_tracker.config as cfg
            importlib.reload(cfg)
            assert "demo-api" in cfg.KALSHI_BASE_URL

    def test_production_url(self):
        with patch.dict(os.environ, {"KALSHI_ENV": "production"}):
            import importlib
            import btc_tracker.config as cfg
            importlib.reload(cfg)
            assert "api.kalshi.com" in cfg.KALSHI_BASE_URL
            assert "demo" not in cfg.KALSHI_BASE_URL


class TestConfigSummary:
    def test_summary_has_no_secrets(self):
        import importlib
        import btc_tracker.config as cfg
        importlib.reload(cfg)
        summary = cfg.get_config_summary()

        # Should not contain actual key values
        assert "KALSHI_API_KEY_ID" not in str(summary)
        assert "KALSHI_PRIVATE_KEY_PATH" not in str(summary)

        # Should contain status fields
        assert "kalshi_configured" in summary
        assert "paper_trade_enabled" in summary
        assert "auto_trade_enabled" in summary
        assert "max_bet_size" in summary
        assert "kelly_fraction" in summary

    def test_summary_types(self):
        import importlib
        import btc_tracker.config as cfg
        importlib.reload(cfg)
        summary = cfg.get_config_summary()

        assert isinstance(summary["kalshi_configured"], bool)
        assert isinstance(summary["paper_trade_enabled"], bool)
        assert isinstance(summary["max_bet_size"], float)
        assert isinstance(summary["max_open_positions"], int)
        assert isinstance(summary["update_interval"], int)
