"""
BTC Tracker for Kalshi - Main Application
Flask web server with background price monitoring and signal generation.
"""

import logging
import json
from datetime import datetime, timezone
from dataclasses import asdict

from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from btc_tracker.price_fetcher import PriceFetcher
from btc_tracker.signals import generate_signals
from btc_tracker.kalshi import ContractAnalyzer, generate_simulated_contracts
from btc_tracker.tracker import ProfitTracker
from btc_tracker.strategy import StrategyEngine
from btc_tracker.backtest import Backtester

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
fetcher = PriceFetcher()
analyzer = ContractAnalyzer()
tracker = ProfitTracker()
strategy = StrategyEngine()

# Cache for latest analysis
latest_analysis = {
    "signals": None,
    "recommendations": [],
    "contracts": [],
    "strategy_probs": None,
    "distribution": None,
    "backtest": None,
    "updated_at": None,
}


def run_analysis():
    """Periodic task: fetch price, run signals, scan for opportunities."""
    fetcher.update()
    prices = fetcher.get_prices_as_list()

    if not prices or len(prices) < 5:
        return

    signals = generate_signals(prices)
    current_price = fetcher.current_price

    # Generate simulated Kalshi contracts (replace with real API in production)
    contracts = generate_simulated_contracts(current_price)

    # Scan for +EV opportunities
    recommendations = analyzer.scan_opportunities(
        contracts=contracts,
        current_price=current_price,
        prices=prices,
        signal_direction=signals.direction,
        signal_confidence=signals.confidence,
    )

    # Record top recommendations
    for rec in recommendations[:3]:  # Top 3 picks
        tracker.record_trade(rec, current_price)

    # Resolve old pending trades for demo
    tracker.resolve_all_pending(current_price)

    # Update cache
    latest_analysis["signals"] = {
        "direction": signals.direction,
        "confidence": signals.confidence,
        "trend": signals.trend,
        "trend_strength": signals.trend_strength,
        "rsi": signals.rsi,
        "rsi_signal": signals.rsi_signal,
        "bb_position": signals.bb_position,
        "bb_signal": signals.bb_signal,
        "macd": asdict(signals.macd) if signals.macd else None,
    }
    latest_analysis["recommendations"] = [
        {
            "contract_id": r.contract.contract_id,
            "title": r.contract.title,
            "side": r.side,
            "strike_price": r.contract.strike_price,
            "yes_price": r.contract.yes_price,
            "no_price": r.contract.no_price,
            "model_probability": r.model_probability,
            "market_probability": r.market_probability,
            "edge": r.edge,
            "ev": r.ev,
            "confidence": r.confidence,
            "reasoning": r.reasoning,
        }
        for r in recommendations
    ]
    latest_analysis["contracts"] = [
        {
            "contract_id": c.contract_id,
            "title": c.title,
            "direction": c.direction,
            "strike_price": c.strike_price,
            "yes_price": c.yes_price,
            "no_price": c.no_price,
            "volume": c.volume,
        }
        for c in contracts
    ]
    # Strategy ensemble probabilities for current price strikes
    strategy_probs = {}
    for offset in [-2000, -1000, -500, 0, 500, 1000, 2000]:
        strike = round(current_price + offset, -2)
        prob = strategy.estimate_probability(prices, strike, current_price)
        strategy_probs[str(strike)] = prob

    latest_analysis["strategy_probs"] = strategy_probs

    # Monte Carlo distribution
    dist = strategy.get_distribution(prices, current_price)
    latest_analysis["distribution"] = dist

    latest_analysis["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"Analysis complete: {signals.direction} ({signals.confidence:.0%}), "
        f"{len(recommendations)} opportunities found"
    )


# --- Routes ---

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "price": fetcher.current_price,
        "last_update": fetcher.last_update.isoformat() if fetcher.last_update else None,
        "history_points": len(fetcher.price_history),
    })


@app.route("/api/price")
def api_price():
    history = fetcher.get_history(n=200)
    return jsonify({
        "current": fetcher.current_price,
        "history": history,
    })


@app.route("/api/signals")
def api_signals():
    return jsonify(latest_analysis["signals"])


@app.route("/api/opportunities")
def api_opportunities():
    return jsonify({
        "recommendations": latest_analysis["recommendations"],
        "contracts": latest_analysis["contracts"],
        "updated_at": latest_analysis["updated_at"],
    })


@app.route("/api/trades")
def api_trades():
    return jsonify({
        "recent": tracker.get_recent_trades(30),
        "stats": tracker.get_stats(),
    })


@app.route("/api/chart")
def api_chart():
    """Return price data formatted for charting."""
    history = fetcher.get_history(n=500)
    prices = [h["price"] for h in history]
    signals = generate_signals(prices) if len(prices) > 15 else None

    return jsonify({
        "prices": history,
        "signals": {
            "direction": signals.direction if signals else "unknown",
            "confidence": signals.confidence if signals else 0,
        } if signals else None,
    })


@app.route("/api/strategy")
def api_strategy():
    """Return ensemble strategy probabilities and distribution."""
    return jsonify({
        "probabilities": latest_analysis.get("strategy_probs"),
        "distribution": latest_analysis.get("distribution"),
        "weights": strategy.get_strategy_weights(),
    })


@app.route("/api/backtest")
def api_backtest():
    """Run backtest on historical data and return results."""
    prices = fetcher.get_prices_as_list()
    if len(prices) < 100:
        return jsonify({"error": "Need more historical data for backtest"})

    bt = Backtester(lookback=50, forecast_horizon=12)
    result = bt.run(prices)
    return jsonify({
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "avg_ev": result.avg_ev,
        "avg_edge": result.avg_edge,
        "direction_accuracy": result.direction_accuracy,
        "equity_curve": result.equity_curve[-100:],  # Last 100 points
    })


# --- Startup ---

def start_scheduler():
    """Start background price monitoring."""
    scheduler = BackgroundScheduler()
    # Fetch price every 60 seconds
    scheduler.add_job(run_analysis, "interval", seconds=60, id="analysis",
                      max_instances=1, misfire_grace_time=30)
    scheduler.start()
    logger.info("Scheduler started - updating every 60 seconds")


if __name__ == "__main__":
    logger.info("Starting BTC Tracker for Kalshi...")

    # Seed with historical data
    logger.info("Fetching historical price data...")
    fetcher.fetch_historical_hourly(days=7)

    # Run initial analysis
    fetcher.update()
    run_analysis()

    # Start periodic updates
    start_scheduler()

    # Run Flask
    app.run(host="0.0.0.0", port=5050, debug=False)
