"""
Configuration manager.
Loads settings from .env file and provides typed access.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file manually (no python-dotenv dependency)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    logger.info("Loaded .env configuration")


def _get(key, default=None):
    return os.environ.get(key, default)


def _get_float(key, default):
    try:
        return float(_get(key, default))
    except (TypeError, ValueError):
        return default


def _get_bool(key, default=False):
    val = _get(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _get_int(key, default):
    try:
        return int(_get(key, default))
    except (TypeError, ValueError):
        return default


# --- Kalshi API ---
KALSHI_API_KEY_ID = _get("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = _get("KALSHI_PRIVATE_KEY_PATH", "")
KALSHI_ENV = _get("KALSHI_ENV", "demo")

KALSHI_BASE_URL = (
    "https://api.kalshi.com/trade-api/v2"
    if KALSHI_ENV == "production"
    else "https://demo-api.kalshi.co/trade-api/v2"
)

# --- Trading Controls ---
AUTO_TRADE_ENABLED = _get_bool("AUTO_TRADE_ENABLED", False)
MAX_BET_SIZE = _get_float("MAX_BET_SIZE", 5.00)
MAX_DAILY_EXPOSURE = _get_float("MAX_DAILY_EXPOSURE", 50.00)
MIN_EV_THRESHOLD = _get_float("MIN_EV_THRESHOLD", 0.05)
MIN_CONFIDENCE = _get("MIN_CONFIDENCE", "medium")
MAX_OPEN_POSITIONS = _get_int("MAX_OPEN_POSITIONS", 10)

# --- Risk Management ---
DAILY_LOSS_LIMIT = _get_float("DAILY_LOSS_LIMIT", -25.00)
KELLY_FRACTION = _get_float("KELLY_FRACTION", 0.25)

# --- App ---
PORT = _get_int("PORT", 5050)
UPDATE_INTERVAL = _get_int("UPDATE_INTERVAL", 60)
LOG_LEVEL = _get("LOG_LEVEL", "INFO")


def is_kalshi_configured():
    """Check if Kalshi API credentials are set."""
    return bool(KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH)


def get_config_summary():
    """Return config state for dashboard (no secrets)."""
    return {
        "kalshi_configured": is_kalshi_configured(),
        "kalshi_env": KALSHI_ENV,
        "auto_trade_enabled": AUTO_TRADE_ENABLED,
        "max_bet_size": MAX_BET_SIZE,
        "max_daily_exposure": MAX_DAILY_EXPOSURE,
        "min_ev_threshold": MIN_EV_THRESHOLD,
        "min_confidence": MIN_CONFIDENCE,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "daily_loss_limit": DAILY_LOSS_LIMIT,
        "kelly_fraction": KELLY_FRACTION,
        "update_interval": UPDATE_INTERVAL,
    }
