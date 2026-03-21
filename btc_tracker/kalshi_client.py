"""
Kalshi API client with RSA key-pair authentication.
Handles market data fetching, order placement, and portfolio management.
"""

import base64
import hashlib
import logging
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from btc_tracker import config
from btc_tracker.kalshi import KalshiContract

logger = logging.getLogger(__name__)


class KalshiClient:
    """Authenticated Kalshi API client."""

    def __init__(self):
        self.base_url = config.KALSHI_BASE_URL
        self.key_id = config.KALSHI_API_KEY_ID
        self._private_key = None
        self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key from PEM file."""
        key_path = config.KALSHI_PRIVATE_KEY_PATH
        if not key_path:
            return

        path = Path(key_path)
        if not path.exists():
            logger.warning(f"Private key file not found: {key_path}")
            return

        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            with open(path, "rb") as f:
                self._private_key = load_pem_private_key(f.read(), password=None)
            logger.info("Kalshi RSA private key loaded")
        except ImportError:
            logger.warning(
                "cryptography package not installed. "
                "Run: pip install cryptography"
            )
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")

    def _sign_request(self, method, path):
        """Generate RSA-PSS signature for Kalshi API authentication."""
        if not self._private_key:
            raise RuntimeError("Private key not loaded")

        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes

        timestamp = str(int(time.time() * 1000))
        # Sign: timestamp + method + path (without query params)
        clean_path = path.split("?")[0]
        message = f"{timestamp}{method.upper()}{clean_path}"

        signature = self._private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    def _request(self, method, path, json_data=None, params=None):
        """Make authenticated request to Kalshi API."""
        url = f"{self.base_url}{path}"
        headers = self._sign_request(method, path)

        try:
            resp = requests.request(
                method, url, headers=headers,
                json=json_data, params=params, timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Kalshi API error: {e} - {resp.text}")
            raise
        except Exception as e:
            logger.error(f"Kalshi request failed: {e}")
            raise

    @property
    def is_configured(self):
        return bool(self.key_id and self._private_key)

    # --- Market Data ---

    def get_btc_markets(self, series_ticker="KXBTC"):
        """Fetch active BTC price prediction markets."""
        params = {"series_ticker": series_ticker, "status": "open", "limit": 100}
        data = self._request("GET", "/markets", params=params)
        markets = data.get("markets", [])

        contracts = []
        for m in markets:
            # Parse strike from market rules/title
            title = m.get("title", "")
            ticker = m.get("ticker", "")
            yes_price = m.get("yes_ask", 0) / 100  # Kalshi prices in cents
            no_price = m.get("no_ask", 0) / 100

            # Extract strike price from title (e.g., "Bitcoin above $85,000?")
            strike = self._parse_strike(title)
            if strike is None:
                continue

            direction = "higher" if "above" in title.lower() else "lower"

            contracts.append(KalshiContract(
                contract_id=ticker,
                title=title,
                direction=direction,
                strike_price=strike,
                expiry=m.get("close_time", ""),
                yes_price=yes_price,
                no_price=no_price,
                volume=m.get("volume", 0),
            ))

        logger.info(f"Fetched {len(contracts)} BTC contracts from Kalshi")
        return contracts

    def get_orderbook(self, ticker):
        """Get order book for a specific market."""
        return self._request("GET", f"/markets/{ticker}/orderbook")

    # --- Portfolio ---

    def get_balance(self):
        """Get account balance in dollars."""
        data = self._request("GET", "/portfolio/balance")
        return {
            "balance": data.get("balance", 0) / 100,  # Cents to dollars
            "portfolio_value": data.get("portfolio_value", 0) / 100,
        }

    def get_positions(self):
        """Get current open positions."""
        data = self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    def get_fills(self, limit=50):
        """Get recent trade fills."""
        params = {"limit": limit}
        data = self._request("GET", "/portfolio/fills", params=params)
        return data.get("fills", [])

    # --- Trading ---

    def place_order(self, ticker, side, action="buy", count=1,
                    order_type="market", price_cents=None):
        """
        Place an order on Kalshi.

        ticker: market ticker (e.g., "KXBTC-25MAR21-B85000")
        side: "yes" or "no"
        action: "buy" or "sell"
        count: number of contracts
        order_type: "market" or "limit"
        price_cents: limit price in cents (required for limit orders)
        """
        order = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action,
            "type": order_type,
            "count": count,
        }
        if order_type == "limit" and price_cents is not None:
            order["yes_price"] = price_cents if side.lower() == "yes" else None
            order["no_price"] = price_cents if side.lower() == "no" else None

        logger.info(f"Placing order: {side.upper()} {count}x {ticker} @ {order_type}")
        data = self._request("POST", "/portfolio/orders", json_data=order)
        return data.get("order", {})

    def cancel_order(self, order_id):
        """Cancel an open order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_orders(self, status="resting"):
        """Get orders by status."""
        params = {"status": status}
        data = self._request("GET", "/portfolio/orders", params=params)
        return data.get("orders", [])

    # --- Helpers ---

    @staticmethod
    def _parse_strike(title):
        """Extract strike price from market title."""
        import re
        # Match patterns like "$85,000", "$84500", "$85,000.00"
        match = re.search(r'\$([0-9,]+(?:\.\d+)?)', title)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                return float(price_str)
            except ValueError:
                pass
        return None
