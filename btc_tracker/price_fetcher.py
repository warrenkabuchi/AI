"""
Multi-API BTC price fetcher using CoinGecko, CoinCap, and Binance.
Aggregates prices and maintains historical data for analysis.
"""

import time
import logging
import requests
import threading
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)

# Keep 7 days of minute-level data
MAX_HISTORY = 7 * 24 * 60


class PriceFetcher:
    def __init__(self):
        self.price_history = deque(maxlen=MAX_HISTORY)
        self.current_price = None
        self.last_update = None
        self._lock = threading.Lock()

    def fetch_coingecko(self):
        """Fetch BTC price from CoinGecko free API."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": "bitcoin", "vs_currencies": "usd"}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()["bitcoin"]["usd"]
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
            return None

    def fetch_coincap(self):
        """Fetch BTC price from CoinCap API."""
        try:
            url = "https://api.coincap.io/v2/assets/bitcoin"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return float(resp.json()["data"]["priceUsd"])
        except Exception as e:
            logger.warning(f"CoinCap fetch failed: {e}")
            return None

    def fetch_binance(self):
        """Fetch BTC price from Binance public API."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": "BTCUSDT"}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception as e:
            logger.warning(f"Binance fetch failed: {e}")
            return None

    def fetch_price(self):
        """Fetch from all APIs and return median price for robustness."""
        prices = []
        for fetcher in [self.fetch_coingecko, self.fetch_coincap, self.fetch_binance]:
            p = fetcher()
            if p is not None:
                prices.append(p)

        if not prices:
            logger.error("All price APIs failed")
            return None

        # Use median for robustness against outliers
        prices.sort()
        median = prices[len(prices) // 2]
        return round(median, 2)

    def update(self):
        """Fetch latest price and append to history."""
        price = self.fetch_price()
        if price is None:
            return

        now = datetime.now(timezone.utc)
        with self._lock:
            self.current_price = price
            self.last_update = now
            self.price_history.append({
                "timestamp": now.isoformat(),
                "price": price,
            })
        logger.info(f"BTC price updated: ${price:,.2f}")
        return price

    def get_prices_as_list(self):
        """Return price history as a list of floats."""
        with self._lock:
            return [p["price"] for p in self.price_history]

    def get_history(self, n=None):
        """Return last n data points."""
        with self._lock:
            if n is None:
                return list(self.price_history)
            return list(self.price_history)[-n:]

    def fetch_historical_hourly(self, days=7):
        """Seed history with hourly candles from CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {"vs_currency": "usd", "days": days}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()["prices"]
            with self._lock:
                for ts_ms, price in data:
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    self.price_history.append({
                        "timestamp": dt.isoformat(),
                        "price": round(price, 2),
                    })
                if self.price_history:
                    self.current_price = self.price_history[-1]["price"]
                    self.last_update = datetime.now(timezone.utc)
            logger.info(f"Loaded {len(data)} historical data points")
        except Exception as e:
            logger.warning(f"Historical data fetch failed: {e}")
