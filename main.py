import os
import json
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from functools import wraps
import pytz
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import TelegramError, NetworkError, TimedOut
from openai import OpenAI
import time
import secrets

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantSignals")

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


# ============ ERROR TRACKING ============
error_counts = {
    "api_errors": 0,
    "telegram_errors": 0,
    "coinbase_errors": 0,
    "redis_errors": 0,
    "last_error": None,
    "last_error_time": None
}


def log_error(category: str, error: Exception, context: str = ""):
    """Log error with tracking."""
    error_counts[f"{category}_errors"] = error_counts.get(f"{category}_errors", 0) + 1
    error_counts["last_error"] = f"{category}: {str(error)[:100]}"
    error_counts["last_error_time"] = datetime.now().isoformat()
    logger.error(f"[{category.upper()}] {context} - {error}")
    logger.debug(traceback.format_exc())


def safe_async(category: str = "general"):
    """Decorator for safe async execution with error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                log_error(category, e, func.__name__)
                return None
        return wrapper
    return decorator


# ============ REDIS CONNECTION ============
redis_client = None
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL")
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
        redis_client.ping()  # Test connection
        logger.info("✅ Redis connected")
    else:
        logger.warning("⚠️ REDIS_URL not set - using memory storage")
except Exception as e:
    logger.error(f"❌ Redis connection failed: {e}")
    redis_client = None


# ============ STORAGE FUNCTIONS ============
def save_data(key: str, data: dict) -> bool:
    """Save data to Redis with error handling."""
    if redis_client:
        try:
            redis_client.set(f"qs_{key}", json.dumps(data, default=str))
            return True
        except Exception as e:
            log_error("redis", e, f"save_data({key})")
    return False


def load_data(key: str, default: dict = None) -> dict:
    """Load data from Redis with error handling."""
    if redis_client:
        try:
            data = redis_client.get(f"qs_{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            log_error("redis", e, f"load_data({key})")
    return default if default is not None else {}


def save_positions():
    save_data("positions", positions)
    save_data("daily_pnl", daily_pnl)
    save_data("signal_history", signal_history)
    save_data("price_alerts", price_alerts)
    save_data("trade_history", trade_history)
    save_data("strategy_performance", strategy_performance)
    save_data("autopilot_settings", autopilot_settings)
    save_data("watchlist", watchlist)
    save_data("risk_state", risk_state)
    save_data("tp_tiers", tp_tiers)
    save_data("dca_settings", dca_settings)
    save_data("perf_stats", perf_stats)


def load_positions():
    global positions, daily_pnl, signal_history, price_alerts, trade_history
    global strategy_performance, autopilot_settings, watchlist, risk_state
    global tp_tiers, dca_settings, perf_stats
    
    positions = load_data("positions", {})
    daily_pnl = load_data("daily_pnl", {"realized": 0.0, "trades": 0, "wins": 0, "starting_balance": 0})
    signal_history = load_data("signal_history", [])
    price_alerts = load_data("price_alerts", {})
    trade_history = load_data("trade_history", [])
    strategy_performance = load_data("strategy_performance", strategy_performance)
    
    # Load autopilot settings
    saved_autopilot = load_data("autopilot_settings", {})
    if saved_autopilot:
        autopilot_settings.update(saved_autopilot)
    
    # Load new state
    watchlist = load_data("watchlist", {})
    saved_risk = load_data("risk_state", {})
    if saved_risk:
        risk_state.update(saved_risk)
    saved_tp = load_data("tp_tiers", {})
    if saved_tp:
        tp_tiers.update(saved_tp)
    saved_dca = load_data("dca_settings", {})
    if saved_dca:
        dca_settings.update(saved_dca)
    saved_perf = load_data("perf_stats", {})
    if saved_perf:
        perf_stats.update(saved_perf)
    
    print(f"[REDIS] Loaded {len(positions)} positions, {len(trade_history)} trades")
    print(f"[REDIS] Autopilot: {'ENABLED' if autopilot_settings.get('enabled') else 'DISABLED'}")


# ============ CONFIG ============
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "quantsignals-secret")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Coinbase CDP API
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

# Trading Config
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "5"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "10"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "3"))
DCA_DROP_PCT = float(os.getenv("DCA_DROP_PCT", "5"))

# ============ ULTRA MODE CONFIG ============
# Risk Management
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "25"))  # Max 25% per position
MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", "60"))  # Max 60% at risk
MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "10"))  # Stop if down 10%
MIN_EXPECTED_VALUE = float(os.getenv("MIN_EXPECTED_VALUE", "1.5"))  # Min 1.5% EV

# Kelly Criterion
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # Use 25% Kelly for safety

# ============ MULTI-ASSET CONFIGURATION ============
# Alpaca API (for stocks)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # paper or live

# Asset Classes
ASSET_CLASSES = {
    "crypto": {
        "enabled": True,
        "provider": "coinbase",
        "pairs": [
            "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
            "DOGE-USD", "XRP-USD", "ADA-USD", "MATIC-USD", "DOT-USD",
            "ATOM-USD", "UNI-USD", "AAVE-USD", "LTC-USD", "NEAR-USD"
        ],
        "trading_hours": "24/7",
        "min_trade": 1,
    },
    "stocks": {
        "enabled": bool(ALPACA_API_KEY),
        "provider": "alpaca",
        "symbols": [
            # Mega Cap Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            # Financials
            "JPM", "BAC", "V", "MA",
            # ETFs
            "SPY", "QQQ", "IWM", "DIA",
            # High Volume
            "AMD", "INTC", "NFLX", "CRM", "PYPL"
        ],
        "trading_hours": "9:30-16:00 ET",
        "min_trade": 1,
    },
    "forex": {
        "enabled": bool(ALPACA_API_KEY),  # Alpaca supports forex
        "provider": "alpaca",
        "pairs": [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD",
            "EUR/GBP", "EUR/JPY", "GBP/JPY"
        ],
        "trading_hours": "24/5",  # Sun 5pm - Fri 5pm ET
        "min_trade": 100,
    }
}

# Legacy support
TRADING_PAIRS = ASSET_CLASSES["crypto"]["pairs"]
STOCK_SYMBOLS = ASSET_CLASSES["stocks"]["symbols"]
FOREX_PAIRS = ASSET_CLASSES["forex"]["pairs"]

# Strategy Types
STRATEGY_TYPES = [
    "momentum_breakout",
    "mean_reversion", 
    "volatility_expansion",
    "trend_following",
    "liquidity_sweep",
    "sentiment_anomaly",
    "cross_asset_correlation"
]

# Market Regimes
MARKET_REGIMES = [
    "bull_high_vol",
    "bull_low_vol", 
    "bear_high_vol",
    "bear_low_vol",
    "sideways"
]

app = FastAPI()
tg_app = Application.builder().token(TOKEN).build()

# State tracking
positions = {}
daily_pnl = {"realized": 0.0, "trades": 0, "wins": 0, "starting_balance": 0}
signal_history = []  # Feature 7: Leaderboard
price_alerts = {}  # Feature 8: Custom Alerts
trade_history = []  # Trade history for reports
watchlist = {}  # Watchlist with alerts

# Risk Management State
risk_state = {
    "max_drawdown_pct": 10,  # Auto-pause if down this %
    "daily_loss_limit": 50,  # Max $ loss per day
    "max_position_pct": 30,  # Max % of portfolio in one coin
    "paused_reason": None,   # Why trading paused
    "highest_balance": 0,    # For drawdown calc
    "current_drawdown": 0
}

# Take Profit Tiers
tp_tiers = {
    "enabled": True,
    "tiers": [
        {"pct_gain": 5, "sell_pct": 25},   # At +5%, sell 25%
        {"pct_gain": 10, "sell_pct": 25},  # At +10%, sell another 25%
        {"pct_gain": 20, "sell_pct": 25},  # At +20%, sell another 25%
        # Remaining 25% rides with trailing stop
    ]
}

# DCA Settings
dca_settings = {
    "enabled": False,
    "coins": ["BTC-USD", "ETH-USD"],
    "drop_pct": 5,       # Buy when down 5%
    "amount_usd": 25,    # DCA amount
    "max_per_day": 3,    # Max DCA buys per day
    "today_count": 0
}

# Performance Stats
perf_stats = {
    "best_trade": {"pair": None, "pnl_pct": 0, "pnl_usd": 0},
    "worst_trade": {"pair": None, "pnl_pct": 0, "pnl_usd": 0},
    "current_streak": 0,
    "best_streak": 0,
    "worst_streak": 0,
    "total_volume": 0
}

# Live trading mode
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Auto signal config
AUTO_SIGNAL_CHATS = os.getenv("AUTO_SIGNAL_CHATS", "").split(",")
AUTO_SIGNAL_CHATS = [c.strip() for c in AUTO_SIGNAL_CHATS if c.strip()]
SIGNAL_HOURS = [6, 12, 18]

# Autopilot mode
autopilot_settings = {
    "enabled": False,
    "max_daily_trades": 15,
    "min_confidence": 70,
    "trades_today": 0,
    "last_reset": None,
    "use_percentage": True,
    "trade_percentage": 20,
    "min_trade_usd": 5,
    "max_trade_usd": 500,
    "reinvest_profits": True,
    "total_profit": 0.0,
    # Ultra features
    "use_kelly": True,
    "min_expected_value": 1.5,
    "scan_interval": 60,  # Scan every 60 seconds
    "daily_target_pct": 3.0,
    "daily_achieved_pct": 0.0,
    "current_regime": "unknown",
    "regime_confidence": 0,
    "paused": False,
    "pause_reason": None,
}

# Strategy Performance Tracking (Self-Learning)
strategy_performance = {
    "momentum_breakout": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
    "mean_reversion": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
    "volatility_expansion": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
    "trend_following": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
    "liquidity_sweep": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
    "sentiment_anomaly": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
    "cross_asset_correlation": {"trades": 0, "wins": 0, "pnl": 0, "weight": 1.0, "avg_ev": 0},
}

# Portfolio Tracking
portfolio_stats = {
    "starting_balance": 0,
    "peak_balance": 0,
    "current_drawdown": 0,
    "max_drawdown": 0,
    "sharpe_ratio": 0,
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "avg_win": 0,
    "avg_loss": 0,
    "profit_factor": 0,
    "daily_returns": [],
}


# ============ COINBASE CDP CLIENT ============
class CoinbaseCDPClient:
    """Coinbase CDP API client with JWT auth."""
    
    BASE_URL = "https://api.coinbase.com"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.replace("\\n", "\n")
    
    def _build_jwt(self, method: str, path: str) -> str:
        import jwt as pyjwt
        import secrets as sec
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        uri = f"{method} api.coinbase.com{path}"
        payload = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
            "uri": uri,
            "nonce": sec.token_hex(16),
        }
        headers = {"kid": self.api_key, "nonce": sec.token_hex(16)}
        
        private_key = serialization.load_pem_private_key(
            self.api_secret.encode(), password=None, backend=default_backend()
        )
        return pyjwt.encode(payload, private_key, algorithm="ES256", headers=headers)
    
    def _headers(self, method: str, path: str) -> dict:
        return {"Authorization": f"Bearer {self._build_jwt(method, path)}", "Content-Type": "application/json"}
    
    async def get_accounts(self) -> dict:
        path = "/api/v3/brokerage/accounts"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.BASE_URL}{path}", headers=self._headers("GET", path), timeout=15)
                return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def place_market_order(self, product_id: str, side: str, usd_amount: float) -> dict:
        path = "/api/v3/brokerage/orders"
        body = {
            "client_order_id": f"qs_{int(time.time())}_{secrets.token_hex(4)}",
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {"market_market_ioc": {"quote_size": str(usd_amount)}}
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self.BASE_URL}{path}", headers=self._headers("POST", path), 
                                         content=json.dumps(body), timeout=15)
                return resp.json() if resp.status_code in [200, 201] else {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_auth(self) -> dict:
        path = "/api/v3/brokerage/accounts"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.BASE_URL}{path}", headers=self._headers("GET", path), timeout=15)
                return {"status": resp.status_code, "body": resp.text[:500]}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_usd_balance(self) -> float:
        """Get available USD balance."""
        try:
            accounts = await self.get_accounts()
            for account in accounts.get("accounts", []):
                if account.get("currency") == "USD":
                    return float(account.get("available_balance", {}).get("value", 0))
        except Exception as e:
            print(f"[CDP] Balance error: {e}")
        return 0.0
    
    async def get_all_holdings(self) -> dict:
        """Get all crypto holdings from Coinbase with current values."""
        holdings = {
            "usd_balance": 0,
            "crypto_value": 0,
            "total_value": 0,
            "assets": []
        }
        
        try:
            accounts = await self.get_accounts()
            
            for account in accounts.get("accounts", []):
                currency = account.get("currency", "")
                balance = float(account.get("available_balance", {}).get("value", 0))
                
                if balance <= 0.0000001:
                    continue
                
                if currency == "USD":
                    holdings["usd_balance"] = balance
                    holdings["total_value"] += balance
                else:
                    # Get current price
                    price = await get_public_price(f"{currency}-USD")
                    if price > 0:
                        value = balance * price
                        holdings["crypto_value"] += value
                        holdings["total_value"] += value
                        
                        holdings["assets"].append({
                            "currency": currency,
                            "balance": balance,
                            "price": price,
                            "value": value
                        })
            
            # Sort by value descending
            holdings["assets"] = sorted(holdings["assets"], key=lambda x: x["value"], reverse=True)
            
        except Exception as e:
            print(f"[CDP] Holdings error: {e}")
        
        return holdings


# Initialize CDP client
cdp_client = None
if COINBASE_API_KEY and COINBASE_API_SECRET:
    try:
        cdp_client = CoinbaseCDPClient(COINBASE_API_KEY, COINBASE_API_SECRET)
        print("✅ Coinbase CDP client initialized")
    except Exception as e:
        print(f"❌ CDP client init failed: {e}")


# ============ ALPACA CLIENT (STOCKS & FOREX) ============
class AlpacaClient:
    """Alpaca API client for stocks and forex trading."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.data_url = "https://data.alpaca.markets"
    
    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
    
    async def get_account(self) -> dict:
        """Get account info."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/v2/account", headers=self._headers(), timeout=15)
                return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_positions(self) -> list:
        """Get all open positions."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/v2/positions", headers=self._headers(), timeout=15)
                return resp.json()
        except Exception as e:
            return []
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get latest stock price."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.data_url}/v2/stocks/{symbol}/trades/latest",
                    headers=self._headers(),
                    timeout=10
                )
                data = resp.json()
                return float(data.get("trade", {}).get("p", 0))
        except:
            return 0
    
    async def get_stock_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 24) -> list:
        """Get stock OHLCV bars."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.data_url}/v2/stocks/{symbol}/bars",
                    headers=self._headers(),
                    params={"timeframe": timeframe, "limit": limit},
                    timeout=10
                )
                return resp.json().get("bars", [])
        except:
            return []
    
    async def get_forex_rate(self, pair: str) -> float:
        """Get forex rate (e.g., EUR/USD)."""
        # Convert EUR/USD to EURUSD format
        symbol = pair.replace("/", "")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.data_url}/v1beta1/forex/latest/rates",
                    headers=self._headers(),
                    params={"currency_pairs": symbol},
                    timeout=10
                )
                data = resp.json()
                rates = data.get("rates", {})
                if symbol in rates:
                    return float(rates[symbol].get("bp", 0))  # bid price
        except:
            pass
        return 0
    
    async def place_stock_order(self, symbol: str, side: str, qty: float = None, notional: float = None) -> dict:
        """Place stock market order."""
        body = {
            "symbol": symbol,
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day"
        }
        if qty:
            body["qty"] = str(qty)
        elif notional:
            body["notional"] = str(notional)  # Dollar amount
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/v2/orders",
                    headers=self._headers(),
                    json=body,
                    timeout=15
                )
                return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_buying_power(self) -> float:
        """Get available buying power."""
        try:
            account = await self.get_account()
            return float(account.get("buying_power", 0))
        except:
            return 0
    
    async def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        try:
            account = await self.get_account()
            return float(account.get("portfolio_value", 0))
        except:
            return 0
    
    async def is_market_open(self) -> bool:
        """Check if stock market is open."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/v2/clock", headers=self._headers(), timeout=10)
                return resp.json().get("is_open", False)
        except:
            return False
    
    async def get_all_holdings(self) -> dict:
        """Get all stock/forex holdings."""
        holdings = {
            "cash": 0,
            "portfolio_value": 0,
            "assets": []
        }
        
        try:
            account = await self.get_account()
            holdings["cash"] = float(account.get("cash", 0))
            holdings["portfolio_value"] = float(account.get("portfolio_value", 0))
            
            positions = await self.get_positions()
            for pos in positions:
                holdings["assets"].append({
                    "symbol": pos.get("symbol"),
                    "qty": float(pos.get("qty", 0)),
                    "market_value": float(pos.get("market_value", 0)),
                    "avg_entry": float(pos.get("avg_entry_price", 0)),
                    "current_price": float(pos.get("current_price", 0)),
                    "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                    "unrealized_plpc": float(pos.get("unrealized_plpc", 0)) * 100
                })
            
            holdings["assets"] = sorted(holdings["assets"], key=lambda x: x["market_value"], reverse=True)
        except Exception as e:
            print(f"[ALPACA] Holdings error: {e}")
        
        return holdings


# Initialize Alpaca client
alpaca_client = None
if ALPACA_API_KEY and ALPACA_API_SECRET:
    try:
        alpaca_client = AlpacaClient(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)
        print("✅ Alpaca client initialized")
    except Exception as e:
        print(f"❌ Alpaca client init failed: {e}")


# ============ UNIFIED PRICE GETTER ============
async def get_asset_price(symbol: str, asset_class: str = "auto") -> float:
    """Get price for any asset class."""
    
    # Auto-detect asset class
    if asset_class == "auto":
        if "-USD" in symbol or symbol in ["BTC", "ETH", "SOL"]:
            asset_class = "crypto"
        elif "/" in symbol:
            asset_class = "forex"
        else:
            asset_class = "stocks"
    
    if asset_class == "crypto":
        pair = symbol if "-USD" in symbol else f"{symbol}-USD"
        return await get_public_price(pair)
    
    elif asset_class == "stocks" and alpaca_client:
        return await alpaca_client.get_stock_price(symbol)
    
    elif asset_class == "forex" and alpaca_client:
        return await alpaca_client.get_forex_rate(symbol)
    
    return 0


# ============ PUBLIC PRICE APIs ============
async def get_public_price(product_id: str) -> float:
    """Get price from Coinbase public API."""
    url = f"https://api.coinbase.com/v2/prices/{product_id}/spot"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            return float(resp.json().get("data", {}).get("amount", 0))
    except:
        return 0


async def get_public_candles(product_id: str, granularity: int = 3600) -> dict:
    """Get candle data from Coinbase Exchange API."""
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity={granularity}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            candles = resp.json()
            if candles and isinstance(candles, list):
                prices = [c[4] for c in candles[:24]]
                return {
                    "prices": prices,
                    "high_24h": max(c[2] for c in candles[:24]),
                    "low_24h": min(c[1] for c in candles[:24]),
                    "volume_24h": sum(c[5] for c in candles[:24]),
                }
    except:
        pass
    return {}


# ============ FEATURE 3: FEAR & GREED INDEX ============
async def get_fear_greed_index() -> dict:
    """Get crypto Fear & Greed Index."""
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            data = resp.json().get("data", [{}])[0]
            return {
                "value": int(data.get("value", 50)),
                "classification": data.get("value_classification", "Neutral"),
                "timestamp": data.get("timestamp", "")
            }
    except:
        return {"value": 50, "classification": "Neutral", "timestamp": ""}


# ============ FEATURE 4: WHALE ALERTS ============
async def get_whale_alerts() -> list:
    """Get recent large crypto transactions (simulated - real API needs key)."""
    # In production, use whale-alert.io API
    # For now, return mock data structure
    try:
        # Check for large movements via public APIs
        alerts = []
        for pair in ["BTC-USD", "ETH-USD"]:
            candles = await get_public_candles(pair, 300)  # 5min candles
            if candles and candles.get("volume_24h", 0) > 0:
                avg_vol = candles["volume_24h"] / 24
                # Detect volume spikes (simulated)
                alerts.append({
                    "coin": pair.split("-")[0],
                    "type": "volume_spike",
                    "message": f"High volume detected"
                })
        return alerts[:3]
    except:
        return []


# ============ FEATURE 9: NEWS INTEGRATION ============
async def get_crypto_news() -> list:
    """Get latest crypto news headlines from multiple sources."""
    news = []
    
    # Try CoinGecko news (free, no auth)
    try:
        url = "https://api.coingecko.com/api/v3/news"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:5]:
                    news.append({
                        "title": item.get("title", "")[:100],
                        "source": item.get("author", "CoinGecko"),
                        "url": item.get("url", ""),
                        "currencies": []
                    })
                if news:
                    return news
    except Exception as e:
        print(f"[NEWS] CoinGecko error: {e}")
    
    # Fallback: CryptoCompare news (free)
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("Data", [])[:5]:
                    categories = item.get("categories", "").split("|")
                    news.append({
                        "title": item.get("title", "")[:100],
                        "source": item.get("source", "Unknown"),
                        "url": item.get("url", ""),
                        "currencies": [c.strip() for c in categories[:3] if c.strip()]
                    })
                if news:
                    return news
    except Exception as e:
        print(f"[NEWS] CryptoCompare error: {e}")
    
    # Final fallback: Return market-based "news"
    try:
        fg = await get_fear_greed_index()
        news.append({
            "title": f"Market sentiment: {fg.get('classification', 'Neutral')} (Fear & Greed: {fg.get('value', 50)})",
            "source": "Market Data",
            "url": "",
            "currencies": ["BTC", "ETH"]
        })
        
        # Add top mover as news
        for pair in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            candles = await get_public_candles(pair)
            if candles and candles.get("prices"):
                prices = candles["prices"]
                if len(prices) >= 2:
                    change = ((prices[0] - prices[1]) / prices[1]) * 100
                    coin = pair.split("-")[0]
                    direction = "up" if change > 0 else "down"
                    news.append({
                        "title": f"{coin} is {direction} {abs(change):.1f}% in the last hour",
                        "source": "Price Alert",
                        "url": "",
                        "currencies": [coin]
                    })
    except:
        pass
    
    return news


# ============ FEATURE 2: MULTI-TIMEFRAME ANALYSIS ============
async def get_multi_timeframe_data(pair: str) -> dict:
    """Get data for multiple timeframes."""
    timeframes = {
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }
    
    result = {}
    for tf_name, granularity in timeframes.items():
        candles = await get_public_candles(pair, granularity)
        if candles and candles.get("prices"):
            prices = candles["prices"]
            sma_8 = sum(prices[:8]) / 8 if len(prices) >= 8 else prices[0]
            sma_21 = sum(prices[:21]) / 21 if len(prices) >= 21 else prices[0]
            trend = "bullish" if sma_8 > sma_21 else "bearish"
            
            change = 0
            if len(prices) >= 2:
                change = ((prices[0] - prices[1]) / prices[1]) * 100
            
            result[tf_name] = {
                "trend": trend,
                "change": round(change, 2),
                "sma_8": round(sma_8, 2),
                "sma_21": round(sma_21, 2)
            }
    
    return result


# ============ ULTRA: MARKET REGIME DETECTION ============
async def detect_market_regime() -> dict:
    """Detect current market regime using BTC as market proxy."""
    try:
        # Get BTC data for regime detection
        candles = await get_public_candles("BTC-USD", 3600)  # 1h candles
        if not candles or not candles.get("prices"):
            return {"regime": "unknown", "confidence": 0}
        
        prices = candles["prices"]
        
        # Calculate metrics
        sma_20 = sum(prices[:20]) / 20 if len(prices) >= 20 else prices[0]
        sma_50 = sum(prices[:50]) / 50 if len(prices) >= 50 else prices[0]
        current = prices[0]
        
        # Volatility (standard deviation)
        if len(prices) >= 20:
            mean = sum(prices[:20]) / 20
            variance = sum((p - mean) ** 2 for p in prices[:20]) / 20
            volatility = (variance ** 0.5) / mean * 100
        else:
            volatility = 2
        
        # Trend direction
        trend_bullish = current > sma_20 > sma_50
        trend_bearish = current < sma_20 < sma_50
        
        # High/low volatility threshold
        high_vol = volatility > 3
        
        # Determine regime
        if trend_bullish and high_vol:
            regime = "bull_high_vol"
            regime_display = "🐂📈 Bull + High Volatility"
        elif trend_bullish and not high_vol:
            regime = "bull_low_vol"
            regime_display = "🐂📊 Bull + Low Volatility"
        elif trend_bearish and high_vol:
            regime = "bear_high_vol"
            regime_display = "🐻📉 Bear + High Volatility"
        elif trend_bearish and not high_vol:
            regime = "bear_low_vol"
            regime_display = "🐻📊 Bear + Low Volatility"
        else:
            regime = "sideways"
            regime_display = "↔️ Sideways/Ranging"
        
        # Confidence based on trend strength
        trend_strength = abs(current - sma_50) / sma_50 * 100
        confidence = min(95, max(50, 50 + trend_strength * 5))
        
        return {
            "regime": regime,
            "regime_display": regime_display,
            "confidence": round(confidence),
            "volatility": round(volatility, 2),
            "trend_strength": round(trend_strength, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2)
        }
    except Exception as e:
        print(f"[REGIME] Error: {e}")
        return {"regime": "unknown", "confidence": 0}


# ============ ULTRA: KELLY CRITERION POSITION SIZING ============
def calculate_kelly_position(win_rate: float, avg_win: float, avg_loss: float, 
                             balance: float, confidence: float) -> float:
    """Calculate optimal position size using Kelly Criterion."""
    if win_rate <= 0 or avg_loss <= 0:
        return balance * 0.05  # Default 5%
    
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = avg_win / avg_loss if avg_loss > 0 else 1
    p = win_rate / 100
    q = 1 - p
    
    kelly = (b * p - q) / b if b > 0 else 0
    
    # Apply Kelly fraction for safety (typically 25% of full Kelly)
    kelly_adjusted = kelly * KELLY_FRACTION
    
    # Adjust by confidence
    confidence_factor = confidence / 100
    kelly_adjusted *= confidence_factor
    
    # Cap at max position size
    kelly_adjusted = max(0.01, min(kelly_adjusted, MAX_POSITION_PCT / 100))
    
    position_size = balance * kelly_adjusted
    
    return round(position_size, 2)


# ============ ULTRA: CALCULATE EXPECTED VALUE ============
def calculate_expected_value(confidence: float, take_profit: float, stop_loss: float) -> float:
    """Calculate expected value of a trade."""
    win_rate = confidence / 100
    loss_rate = 1 - win_rate
    
    ev = (win_rate * take_profit) - (loss_rate * stop_loss)
    return round(ev, 2)


# ============ ULTRA: UPDATE STRATEGY WEIGHTS (Self-Learning) ============
def update_strategy_weights():
    """Update strategy weights based on performance."""
    global strategy_performance
    
    total_trades = sum(s["trades"] for s in strategy_performance.values())
    if total_trades < 10:
        return  # Need more data
    
    for strategy, stats in strategy_performance.items():
        if stats["trades"] >= 3:
            win_rate = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            pnl_factor = 1 + (stats["pnl"] / 100) if stats["pnl"] != 0 else 1
            
            # Weight = win_rate * pnl_factor, normalized
            new_weight = win_rate * pnl_factor
            
            # Smooth update (80% old, 20% new)
            stats["weight"] = stats["weight"] * 0.8 + new_weight * 0.2
            
            # Disable underperforming strategies (weight < 0.3)
            if stats["weight"] < 0.3 and stats["trades"] >= 10:
                stats["weight"] = 0.1  # Heavily reduce but don't completely disable
            
            # Cap weights
            stats["weight"] = max(0.1, min(2.0, stats["weight"]))
    
    save_data("strategy_performance", strategy_performance)


# ============ ULTRA: CHECK RISK LIMITS ============
async def check_risk_limits() -> Tuple[bool, str]:
    """Check if we can trade based on risk limits."""
    global autopilot_settings, portfolio_stats
    
    # Check daily drawdown
    if cdp_client:
        current_balance = await cdp_client.get_usd_balance()
        if portfolio_stats.get("starting_balance", 0) > 0:
            daily_pnl_pct = ((current_balance - portfolio_stats["starting_balance"]) / 
                           portfolio_stats["starting_balance"]) * 100
            if daily_pnl_pct <= -MAX_DAILY_DRAWDOWN:
                return False, f"Daily drawdown limit hit ({daily_pnl_pct:.1f}%)"
    
    # Check max trades per day
    if autopilot_settings["trades_today"] >= autopilot_settings["max_daily_trades"]:
        return False, "Max daily trades reached"
    
    # Check max positions
    if len(positions) >= MAX_POSITIONS:
        return False, "Max positions reached"
    
    # Check portfolio risk
    total_at_risk = sum(p.get("amount_usd", 0) for p in positions.values())
    if cdp_client:
        balance = await cdp_client.get_usd_balance()
        if balance > 0:
            risk_pct = (total_at_risk / balance) * 100
            if risk_pct >= MAX_PORTFOLIO_RISK:
                return False, f"Portfolio risk limit ({risk_pct:.1f}%)"
    
    return True, "OK"


# ============ AI SIGNAL GENERATOR (Enhanced Ultra) ============
async def generate_trading_signals(include_news: bool = True) -> dict:
    """Generate AI trading signals with enhanced data."""
    
    if not OPENAI_API_KEY:
        return {"error": "OpenAI not configured"}
    
    # Get market regime
    regime = await detect_market_regime()
    autopilot_settings["current_regime"] = regime.get("regime", "unknown")
    autopilot_settings["regime_confidence"] = regime.get("confidence", 0)
    
    # Get strategy weights for prompt
    active_strategies = {k: v for k, v in strategy_performance.items() if v["weight"] >= 0.3}
    
    # Gather market data
    market_data = {}
    for pair in TRADING_PAIRS:
        try:
            price = await get_public_price(pair)
            candle_data = await get_public_candles(pair)
            
            if price > 0:
                prices = candle_data.get("prices", [price])
                sma_8 = sum(prices[:8]) / 8 if len(prices) >= 8 else price
                sma_21 = sum(prices[:21]) / 21 if len(prices) >= 21 else price
                change_1h = ((prices[0] - prices[1]) / prices[1]) * 100 if len(prices) >= 2 else 0
                
                # Calculate volatility
                if len(prices) >= 20:
                    mean = sum(prices[:20]) / 20
                    variance = sum((p - mean) ** 2 for p in prices[:20]) / 20
                    volatility = (variance ** 0.5) / mean * 100
                else:
                    volatility = 0
                
                # RSI calculation (simplified)
                gains = losses = 0
                for i in range(1, min(15, len(prices))):
                    diff = prices[i-1] - prices[i]
                    if diff > 0:
                        gains += diff
                    else:
                        losses -= diff
                avg_gain = gains / 14 if gains else 0.001
                avg_loss = losses / 14 if losses else 0.001
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                market_data[pair] = {
                    "price": price,
                    "high_24h": candle_data.get("high_24h", price),
                    "low_24h": candle_data.get("low_24h", price),
                    "volume_24h": candle_data.get("volume_24h", 0),
                    "sma_8": round(sma_8, 2),
                    "sma_21": round(sma_21, 2),
                    "change_1h": round(change_1h, 2),
                    "volatility": round(volatility, 2),
                    "rsi": round(rsi, 1),
                    "trend": "bullish" if sma_8 > sma_21 else "bearish"
                }
        except Exception as e:
            print(f"[ERROR] {pair}: {e}")
    
    if not market_data:
        return {"error": "No market data available"}
    
    # Get Fear & Greed
    fear_greed = await get_fear_greed_index()
    
    # Get news if enabled
    news = await get_crypto_news() if include_news else []
    
    # Build AI prompt
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    prompt = f"""You are QuantSignals - an elite autonomous quantitative trading AI designed to maximize long-term portfolio growth.

CURRENT TIME: {now.strftime("%Y-%m-%d %H:%M %Z")}
TRADING STYLE: Multi-strategy quantitative trading

═══════════════════════════════════════════════════
MARKET REGIME: {regime.get('regime_display', 'Unknown')}
Regime Confidence: {regime.get('confidence', 0)}%
Volatility: {regime.get('volatility', 0)}%
═══════════════════════════════════════════════════

FEAR & GREED INDEX: {fear_greed['value']} ({fear_greed['classification']})
- 0-25: Extreme Fear → BUY opportunity
- 75-100: Extreme Greed → SELL signal

AVAILABLE STRATEGIES (use based on regime):
1. MOMENTUM_BREAKOUT - Best in: bull_high_vol, bull_low_vol
2. MEAN_REVERSION - Best in: sideways, bear_low_vol
3. VOLATILITY_EXPANSION - Best in: high volatility regimes
4. TREND_FOLLOWING - Best in: bull_low_vol, bear_low_vol
5. LIQUIDITY_SWEEP - Best in: high volatility after consolidation
6. SENTIMENT_ANOMALY - When fear/greed is extreme
7. CROSS_ASSET_CORRELATION - When BTC leads altcoin moves

LIVE MARKET DATA:
"""
    
    for pair, data in market_data.items():
        prompt += f"""
{pair}: ${data['price']:,.2f}
- 24h Range: ${data['low_24h']:,.2f} - ${data['high_24h']:,.2f}
- 1h Change: {data['change_1h']:+.2f}%
- Volatility: {data.get('volatility', 0):.1f}%
- RSI(14): {data.get('rsi', 50):.0f}
- SMA8/SMA21: ${data['sma_8']:,.2f} / ${data['sma_21']:,.2f}
- Trend: {data['trend'].upper()}
"""
    
    if news:
        prompt += "\n\nRECENT NEWS:\n"
        for n in news[:3]:
            coins = ", ".join(n.get("currencies", [])[:3])
            prompt += f"- {n['title']} [{coins}]\n"
    
    prompt += """

═══════════════════════════════════════════════════
QUANTITATIVE ANALYSIS RULES:
═══════════════════════════════════════════════════

1. REGIME-BASED STRATEGY SELECTION:
   - Bull + High Vol → Momentum breakouts, trend following
   - Bull + Low Vol → Trend following, mean reversion at support
   - Bear + High Vol → Short momentum, volatility plays
   - Bear + Low Vol → Mean reversion, wait for reversal
   - Sideways → Mean reversion, range trading

2. ENTRY CRITERIA (ALL must be met):
   - Confidence > 70%
   - Expected Value > 1.5%
   - Risk/Reward > 2:1
   - Aligned with current regime

3. POSITION SIZING:
   - Calculate based on stop distance and risk tolerance
   - Scale position with confidence level

4. MULTI-TARGET EXITS:
   - TP1: Conservative (scale out 40%)
   - TP2: Standard (scale out 40%)  
   - TP3: Extended (let 20% run)

OUTPUT FORMAT (JSON only, no markdown):
{
    "regime_analysis": "Current regime assessment",
    "signals": [
        {
            "pair": "BTC-USD",
            "direction": "BUY",
            "confidence": 82,
            "expected_value": 3.2,
            "risk_level": "MEDIUM",
            "strategy": "momentum_breakout",
            "entry_price": 97000,
            "stop_loss": 94090,
            "take_profit_1": 99910,
            "take_profit_2": 102820,
            "take_profit_3": 106700,
            "position_size_pct": 15,
            "reasoning": "SMA8 crossed above SMA21, volume confirming breakout, RSI not overbought"
        }
    ],
    "market_sentiment": "bullish",
    "regime": "bull_high_vol",
    "summary": "Brief market summary"
}

PRIORITY: Capital preservation → Consistency → Compounding
Maximum 3 high-quality signals. If no good setups, return empty signals array.
"""
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are QuantSignals, an elite quantitative trading AI. Output valid JSON only, no markdown code blocks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip("`").strip()
        
        signals = json.loads(result_text)
        signals["market_data"] = market_data
        signals["fear_greed"] = fear_greed
        signals["regime_data"] = regime
        signals["generated_at"] = now.isoformat()
        
        # Calculate EV for each signal and track
        for sig in signals.get("signals", []):
            # Calculate EV if not provided
            if "expected_value" not in sig:
                entry = sig.get("entry_price", 0)
                sl = sig.get("stop_loss", 0)
                tp = sig.get("take_profit_1", sig.get("take_profit", 0))
                if entry and sl and tp:
                    risk = abs(entry - sl) / entry * 100
                    reward = abs(tp - entry) / entry * 100
                    confidence = sig.get("confidence", 70)
                    sig["expected_value"] = calculate_expected_value(confidence, reward, risk)
            
            # Track signal for learning
            signal_history.append({
                "pair": sig.get("pair"),
                "direction": sig.get("direction", sig.get("action")),
                "strategy": sig.get("strategy", "unknown"),
                "entry_price": sig.get("entry_price"),
                "confidence": sig.get("confidence"),
                "expected_value": sig.get("expected_value", 0),
                "timestamp": now.isoformat(),
                "result": None
            })
        
        save_positions()
        return signals
        
    except Exception as e:
        return {"error": str(e), "market_data": market_data, "regime_data": regime}


# ============ FEATURE 6: BACKTESTING ============
async def run_backtest(pair: str, days: int = 30) -> dict:
    """Simple backtest using historical data."""
    try:
        # Get historical candles
        candles = await get_public_candles(pair, 3600)  # 1h candles
        if not candles or not candles.get("prices"):
            return {"error": "No historical data"}
        
        prices = candles["prices"][:min(days*24, len(candles["prices"]))]
        
        # Simple SMA crossover strategy backtest
        trades = []
        position = None
        wins = 0
        losses = 0
        total_pnl = 0
        
        for i in range(21, len(prices)):
            sma_8 = sum(prices[i-8:i]) / 8
            sma_21 = sum(prices[i-21:i]) / 21
            price = prices[i]
            
            # Buy signal: SMA8 crosses above SMA21
            if sma_8 > sma_21 and position is None:
                position = {"entry": price, "index": i}
            
            # Sell signal: SMA8 crosses below SMA21 or stop/take profit
            elif position:
                pnl_pct = ((price - position["entry"]) / position["entry"]) * 100
                
                if sma_8 < sma_21 or pnl_pct >= 10 or pnl_pct <= -5:
                    trades.append({"entry": position["entry"], "exit": price, "pnl": pnl_pct})
                    total_pnl += pnl_pct
                    if pnl_pct > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
        
        win_rate = (wins / len(trades) * 100) if trades else 0
        
        return {
            "pair": pair,
            "period": f"{days} days",
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0
        }
    except Exception as e:
        return {"error": str(e)}


# ============ TELEGRAM COMMANDS ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = """🚀 <b>QUANTSIGNALS ULTRA</b>
<i>Multi-Asset Trading Intelligence</i>

═══════════════════════════════════

<b>🪙 CRYPTO</b>
/buy [coin] [amt] - Buy crypto
/sell [coin] [%] - Sell crypto
/signals - AI crypto signals
/portfolio - Crypto holdings

<b>📈 STOCKS</b>
/stocks - Stock market overview
/stockbuy [sym] [amt] - Buy stocks
/stocksell [sym] - Sell stocks

<b>💱 FOREX</b>
/forex - Forex rates

<b>🌐 MULTI-ASSET</b>
/all - Combined portfolio
/performance - Full report

<b>🤖 AUTOPILOT</b>
/autopilot - Auto trading
/dcaauto - DCA autopilot
/tptiers - Take profit tiers

<b>🛡️ RISK</b>
/risk - Risk dashboard
/watchlist - Your watchlist

<b>📊 ANALYTICS</b>
/chart - P&L chart
/streak - Win streak
/besttrades - Best trades
/summary - Daily summary

<b>⚙️ TOOLS</b>
/menu - Button menu
/market - Live prices
/fear - Fear & Greed
/settings - Settings

═══════════════════════════════════
<i>Crypto • Stocks • Forex</i>"""
    
    keyboard = [
        [InlineKeyboardButton("🪙 Crypto", callback_data="menu_portfolio"), InlineKeyboardButton("📈 Stocks", callback_data="menu_stocks")],
        [InlineKeyboardButton("💱 Forex", callback_data="menu_forex"), InlineKeyboardButton("🌐 All Assets", callback_data="all_portfolio")],
        [InlineKeyboardButton("🤖 Autopilot", callback_data="menu_autopilot"), InlineKeyboardButton("📊 Menu", callback_data="menu_back")]
    ]
    
    await update.message.reply_text(welcome, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interactive menu with buttons."""
    keyboard = [
        [
            InlineKeyboardButton("📊 Signals", callback_data="menu_signals"),
            InlineKeyboardButton("💼 Portfolio", callback_data="menu_portfolio")
        ],
        [
            InlineKeyboardButton("💰 Buy", callback_data="menu_buy"),
            InlineKeyboardButton("💸 Sell", callback_data="menu_sell")
        ],
        [
            InlineKeyboardButton("📈 Market", callback_data="menu_market"),
            InlineKeyboardButton("😱 Fear/Greed", callback_data="menu_fear")
        ],
        [
            InlineKeyboardButton("🔍 Regime", callback_data="menu_regime"),
            InlineKeyboardButton("📰 News", callback_data="menu_news")
        ],
        [
            InlineKeyboardButton("🤖 Autopilot", callback_data="menu_autopilot"),
            InlineKeyboardButton("📊 Performance", callback_data="menu_performance")
        ],
        [
            InlineKeyboardButton("📜 History", callback_data="menu_history"),
            InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings")
        ]
    ]
    
    await update.message.reply_text(
        "🚀 <b>QUANTSIGNALS ULTRA</b>\n\n"
        "Select an option:",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# ============ ULTRA: REGIME COMMAND ============
async def regime_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current market regime analysis."""
    msg = await update.message.reply_text("🔍 Analyzing market regime...")
    
    regime = await detect_market_regime()
    fg = await get_fear_greed_index()
    
    # Strategy recommendations based on regime
    regime_strategies = {
        "bull_high_vol": ["⚡ Momentum Breakout", "📈 Trend Following"],
        "bull_low_vol": ["📈 Trend Following", "🔄 Mean Reversion (support)"],
        "bear_high_vol": ["💥 Volatility Plays", "🔄 Mean Reversion"],
        "bear_low_vol": ["🔄 Mean Reversion", "⏳ Wait for reversal"],
        "sideways": ["🔄 Mean Reversion", "📊 Range Trading"],
    }
    
    rec_strategies = regime_strategies.get(regime.get("regime", ""), ["⏳ Wait for clarity"])
    
    text = f"""🔍 <b>MARKET REGIME ANALYSIS</b>
═══════════════════════════════════

<b>Current Regime:</b>
{regime.get('regime_display', '❓ Unknown')}

<b>Metrics:</b>
📊 Confidence: {regime.get('confidence', 0)}%
📈 Volatility: {regime.get('volatility', 0):.1f}%
💪 Trend Strength: {regime.get('trend_strength', 0):.1f}%
📉 SMA20: ${regime.get('sma_20', 0):,.2f}
📉 SMA50: ${regime.get('sma_50', 0):,.2f}

<b>Fear & Greed:</b> {fg['value']} ({fg['classification']})

═══════════════════════════════════
<b>Recommended Strategies:</b>
"""
    for strat in rec_strategies:
        text += f"• {strat}\n"
    
    text += """
═══════════════════════════════════
<i>Regime updated every scan cycle</i>"""
    
    await msg.edit_text(text, parse_mode="HTML")


# ============ ULTRA: STRATEGIES COMMAND ============
async def strategies_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show strategy performance (self-learning stats)."""
    
    text = """📊 <b>STRATEGY PERFORMANCE</b>
<i>Self-Learning Weights</i>
═══════════════════════════════════

"""
    
    # Sort by weight
    sorted_strats = sorted(strategy_performance.items(), key=lambda x: x[1]["weight"], reverse=True)
    
    for strat, stats in sorted_strats:
        win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        weight_bar = "█" * int(stats["weight"] * 5) + "░" * (5 - int(stats["weight"] * 5))
        
        status = "✅" if stats["weight"] >= 0.7 else "⚠️" if stats["weight"] >= 0.3 else "❌"
        
        strat_display = strat.replace("_", " ").title()
        text += f"""{status} <b>{strat_display}</b>
   Weight: [{weight_bar}] {stats['weight']:.2f}
   Trades: {stats['trades']} | Wins: {stats['wins']} ({win_rate:.0f}%)
   P&L: ${stats['pnl']:+.2f}

"""
    
    text += """═══════════════════════════════════
<i>Weights auto-adjust based on performance</i>"""
    
    await update.message.reply_text(text, parse_mode="HTML")


# ============ ULTRA: STATS COMMAND ============
async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show portfolio statistics."""
    
    # Get current balance
    usd_balance = 0
    if cdp_client:
        usd_balance = await cdp_client.get_usd_balance()
    
    total_trades = portfolio_stats.get("total_trades", 0) + daily_pnl.get("trades", 0)
    winning_trades = portfolio_stats.get("winning_trades", 0) + daily_pnl.get("wins", 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = autopilot_settings.get("total_profit", 0) + daily_pnl.get("realized", 0)
    
    text = f"""📈 <b>PORTFOLIO STATISTICS</b>
═══════════════════════════════════

<b>💰 Account:</b>
Balance: ${usd_balance:,.2f}
Open Positions: {len(positions)}

<b>📊 Performance:</b>
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Total P&L: ${total_pnl:+,.2f}

<b>📅 Today:</b>
Trades: {daily_pnl.get('trades', 0)}
P&L: ${daily_pnl.get('realized', 0):+,.2f}
Target: {autopilot_settings.get('daily_target_pct', 2)}%

<b>🎯 Risk Metrics:</b>
Max Position: {MAX_POSITION_PCT}%
Max Portfolio Risk: {MAX_PORTFOLIO_RISK}%
Max Daily Drawdown: {MAX_DAILY_DRAWDOWN}%

<b>🤖 Autopilot:</b>
Status: {'🟢 Active' if autopilot_settings['enabled'] else '🔴 Off'}
Regime: {autopilot_settings.get('current_regime', 'unknown')}

═══════════════════════════════════"""
    
    await update.message.reply_text(text, parse_mode="HTML")


# Feature 3: Fear & Greed
async def fear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show Fear & Greed Index."""
    msg = await update.message.reply_text("📊 Loading Fear & Greed Index...")
    
    fg = await get_fear_greed_index()
    
    # Visual meter
    value = fg["value"]
    if value <= 25:
        emoji = "😱"
        color = "🔴"
    elif value <= 45:
        emoji = "😰"
        color = "🟠"
    elif value <= 55:
        emoji = "😐"
        color = "🟡"
    elif value <= 75:
        emoji = "😊"
        color = "🟢"
    else:
        emoji = "🤑"
        color = "🔵"
    
    bar_filled = int(value / 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    
    text = f"""📊 <b>FEAR & GREED INDEX</b>
━━━━━━━━━━━━━━━

{emoji} <b>{fg['classification'].upper()}</b>

{color} [{bar}] {value}/100

<b>What it means:</b>
• 0-25: Extreme Fear → Buy opportunity
• 25-45: Fear → Consider buying
• 45-55: Neutral → Hold
• 55-75: Greed → Consider selling
• 75-100: Extreme Greed → Sell signal

<i>Updated hourly</i>"""
    
    await msg.edit_text(text, parse_mode="HTML")


# Feature 9: News
async def news_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show crypto news."""
    msg = await update.message.reply_text("📰 Loading news...")
    
    news = await get_crypto_news()
    
    if not news:
        await msg.edit_text("❌ Could not fetch news")
        return
    
    text = """📰 <b>CRYPTO NEWS</b>
━━━━━━━━━━━━━━━

"""
    
    for i, n in enumerate(news[:5], 1):
        coins = ", ".join(n.get("currencies", [])[:3]) or "General"
        text += f"{i}. <b>{n['title']}</b>\n"
        text += f"   📌 {coins} | {n['source']}\n\n"
    
    text += "<i>Use /signals for AI analysis including news</i>"
    
    await msg.edit_text(text, parse_mode="HTML")


# Feature 4: Whale Alerts
async def whale_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show whale alerts."""
    msg = await update.message.reply_text("🐋 Checking whale activity...")
    
    alerts = await get_whale_alerts()
    
    text = """🐋 <b>WHALE ALERTS</b>
━━━━━━━━━━━━━━━

"""
    
    if alerts:
        for alert in alerts:
            text += f"🔔 <b>{alert['coin']}</b>: {alert['message']}\n"
    else:
        text += "No major whale activity detected\n"
    
    text += """
━━━━━━━━━━━━━━━
<i>Large transactions can signal price moves</i>"""
    
    await msg.edit_text(text, parse_mode="HTML")


# Feature 2: Multi-timeframe
async def timeframe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Multi-timeframe analysis."""
    args = context.args
    pair = f"{args[0].upper()}-USD" if args else "BTC-USD"
    
    if pair not in TRADING_PAIRS:
        await update.message.reply_text(f"❌ Unknown pair. Use: {', '.join([p.split('-')[0] for p in TRADING_PAIRS])}")
        return
    
    msg = await update.message.reply_text(f"📊 Analyzing {pair} across timeframes...")
    
    tf_data = await get_multi_timeframe_data(pair)
    price = await get_public_price(pair)
    
    text = f"""📊 <b>{pair} MULTI-TIMEFRAME</b>
━━━━━━━━━━━━━━━

💰 Current: <b>${price:,.2f}</b>

"""
    
    for tf, data in tf_data.items():
        trend_emoji = "🟢" if data["trend"] == "bullish" else "🔴"
        text += f"""<b>{tf.upper()}</b>
{trend_emoji} Trend: {data['trend'].upper()}
📈 Change: {data['change']:+.2f}%
📊 SMA8: ${data['sma_8']:,.2f} / SMA21: ${data['sma_21']:,.2f}

"""
    
    # Confluence check
    trends = [d["trend"] for d in tf_data.values()]
    if all(t == "bullish" for t in trends):
        text += "✅ <b>STRONG BUY</b> - All timeframes bullish"
    elif all(t == "bearish" for t in trends):
        text += "🛑 <b>STRONG SELL</b> - All timeframes bearish"
    else:
        text += "⚠️ <b>MIXED</b> - Conflicting signals"
    
    await msg.edit_text(text, parse_mode="HTML")


# Feature 6: Backtest
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run backtest on a pair."""
    args = context.args
    pair = f"{args[0].upper()}-USD" if args else "BTC-USD"
    days = int(args[1]) if len(args) > 1 else 30
    
    msg = await update.message.reply_text(f"🔬 Running backtest on {pair}...")
    
    result = await run_backtest(pair, days)
    
    if "error" in result:
        await msg.edit_text(f"❌ Error: {result['error']}")
        return
    
    emoji = "🟢" if result["total_pnl"] > 0 else "🔴"
    
    text = f"""🔬 <b>BACKTEST RESULTS</b>
━━━━━━━━━━━━━━━

📊 Pair: <b>{result['pair']}</b>
📅 Period: {result['period']}

<b>Performance:</b>
📈 Total Trades: {result['total_trades']}
✅ Wins: {result['wins']}
❌ Losses: {result['losses']}
🎯 Win Rate: {result['win_rate']}%

{emoji} Total P&L: <b>{result['total_pnl']:+.2f}%</b>
📊 Avg per Trade: {result['avg_pnl']:+.2f}%

<i>Strategy: SMA 8/21 crossover with 5% SL, 10% TP</i>"""
    
    await msg.edit_text(text, parse_mode="HTML")


# Feature 7: Leaderboard
async def leaderboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show signal performance leaderboard."""
    
    if not signal_history:
        await update.message.reply_text("📊 No signal history yet. Signals will be tracked automatically.")
        return
    
    # Calculate stats per pair
    pair_stats = {}
    for sig in signal_history[-100:]:  # Last 100 signals
        pair = sig.get("pair", "Unknown")
        if pair not in pair_stats:
            pair_stats[pair] = {"total": 0, "wins": 0}
        pair_stats[pair]["total"] += 1
        if sig.get("result") == "win":
            pair_stats[pair]["wins"] += 1
    
    text = """🏆 <b>SIGNAL LEADERBOARD</b>
━━━━━━━━━━━━━━━

<b>Top Performing Pairs:</b>

"""
    
    # Sort by win rate
    sorted_pairs = sorted(pair_stats.items(), 
                         key=lambda x: x[1]["wins"]/x[1]["total"] if x[1]["total"] > 0 else 0, 
                         reverse=True)
    
    for i, (pair, stats) in enumerate(sorted_pairs[:5], 1):
        win_rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        text += f"{medal} {pair}: {win_rate:.0f}% ({stats['wins']}/{stats['total']})\n"
    
    text += f"""
━━━━━━━━━━━━━━━
📊 Total Signals: {len(signal_history)}
<i>Last 100 signals analyzed</i>"""
    
    await update.message.reply_text(text, parse_mode="HTML")


# Feature 8: Price Alerts
async def alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set price alert."""
    args = context.args
    
    if len(args) < 2:
        # Show current alerts
        if not price_alerts:
            await update.message.reply_text(
                "🔔 <b>PRICE ALERTS</b>\n\n"
                "No alerts set.\n\n"
                "Usage: /alert BTC 100000\n"
                "Sets alert when BTC reaches $100,000",
                parse_mode="HTML"
            )
            return
        
        text = "🔔 <b>YOUR ALERTS</b>\n━━━━━━━━━━━━━━━\n\n"
        for pair, targets in price_alerts.items():
            for target in targets:
                text += f"• {pair}: ${target:,.2f}\n"
        text += "\nUse /alert clear to remove all"
        await update.message.reply_text(text, parse_mode="HTML")
        return
    
    if args[0].lower() == "clear":
        price_alerts.clear()
        save_positions()
        await update.message.reply_text("✅ All alerts cleared")
        return
    
    coin = args[0].upper()
    pair = f"{coin}-USD"
    try:
        target_price = float(args[1].replace(",", ""))
    except:
        await update.message.reply_text("❌ Invalid price. Use: /alert BTC 100000")
        return
    
    if pair not in price_alerts:
        price_alerts[pair] = []
    price_alerts[pair].append(target_price)
    save_positions()
    
    current = await get_public_price(pair)
    direction = "📈 above" if target_price > current else "📉 below"
    
    await update.message.reply_text(
        f"✅ Alert set!\n\n"
        f"🔔 {pair} at ${target_price:,.2f}\n"
        f"📍 Current: ${current:,.2f}\n"
        f"Triggers when price goes {direction} target",
        parse_mode="HTML"
    )


# Feature 5: DCA Opportunities
async def dca_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show DCA opportunities (coins down significantly)."""
    msg = await update.message.reply_text("📉 Finding DCA opportunities...")
    
    opportunities = []
    
    for pair in TRADING_PAIRS:
        try:
            candles = await get_public_candles(pair)
            if candles and candles.get("prices"):
                price = await get_public_price(pair)
                high = candles["high_24h"]
                drop_pct = ((price - high) / high) * 100
                
                if drop_pct <= -DCA_DROP_PCT:
                    opportunities.append({
                        "pair": pair,
                        "price": price,
                        "high": high,
                        "drop": drop_pct
                    })
        except:
            pass
    
    text = """📉 <b>DCA OPPORTUNITIES</b>
━━━━━━━━━━━━━━━

Coins down more than {:.0f}% from 24h high:

""".format(DCA_DROP_PCT)
    
    if opportunities:
        opportunities.sort(key=lambda x: x["drop"])
        for opp in opportunities[:5]:
            text += f"""🔻 <b>{opp['pair'].split('-')[0]}</b>
   Price: ${opp['price']:,.2f}
   24h High: ${opp['high']:,.2f}
   Drop: {opp['drop']:.1f}%

"""
    else:
        text += "No significant dips found right now.\n"
    
    text += "<i>DCA = Dollar Cost Average into dips</i>"
    
    await msg.edit_text(text, parse_mode="HTML")


async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate AI trading signals."""
    msg = await update.message.reply_text("🔄 Analyzing markets with AI...")
    
    signals = await generate_trading_signals()
    
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    text = f"""📊 <b>QUANTSIGNALS</b>
📅 {now.strftime("%B %d, %Y %I:%M %p %Z")}
━━━━━━━━━━━━━━━

"""
    
    # Fear & Greed
    fg = signals.get("fear_greed", {})
    text += f"😱 Fear/Greed: <b>{fg.get('value', 'N/A')}</b> ({fg.get('classification', 'N/A')})\n\n"
    
    if "error" in signals and "market_data" not in signals:
        text += f"❌ Error: {signals['error']}"
    elif signals.get("signals"):
        sentiment = signals.get('market_sentiment', 'neutral').upper()
        text += f"<b>Sentiment:</b> {sentiment}\n\n"
        
        for signal in signals.get("signals", []):
            action = signal.get("action", "HOLD")
            emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            
            text += f"""{emoji} <b>{signal.get('pair')}</b>
Action: <b>{action}</b>
Confidence: {signal.get('confidence')}%
Entry: ${signal.get('entry_price', 0):,.2f}
SL: ${signal.get('stop_loss', 0):,.2f} | TP: ${signal.get('take_profit', 0):,.2f}
📝 {signal.get('reasoning')}

"""
        
        text += f"━━━━━━━━━━━━━━━\n{signals.get('summary', '')}"
    else:
        text += "⚪ No high-confidence signals\n\n"
        for pair, data in list(signals.get("market_data", {}).items())[:5]:
            emoji = "🟢" if data.get("change_1h", 0) > 0 else "🔴"
            text += f"{emoji} {pair.split('-')[0]}: ${data['price']:,.2f}\n"
    
    text += "\n\n<i>⚠️ Not financial advice</i>"
    
    # Trade buttons
    keyboard = []
    for signal in signals.get("signals", []):
        if signal.get("action") == "BUY":
            keyboard.append([InlineKeyboardButton(
                f"🟢 Buy {signal['pair'].split('-')[0]} (${TRADE_AMOUNT_USD})",
                callback_data=f"trade_buy_{signal['pair']}"
            )])
    
    await msg.edit_text(text, parse_mode="HTML", 
                       reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None)


async def market_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick market overview."""
    msg = await update.message.reply_text("📊 Loading...")
    
    async def get_pair_data(pair):
        price = await get_public_price(pair)
        return {"pair": pair, "price": price}
    
    results = await asyncio.gather(*[get_pair_data(pair) for pair in TRADING_PAIRS])
    
    text = "📊 <b>LIVE MARKET</b>\n━━━━━━━━━━━━━━━\n\n"
    
    for r in results:
        if r["price"] > 0:
            text += f"🪙 <b>{r['pair'].split('-')[0]}</b>: ${r['price']:,.2f}\n"
    
    text += "\n<i>/signals for AI analysis</i>"
    
    await msg.edit_text(text, parse_mode="HTML")


# ============ STOCKS COMMANDS ============
async def stocks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show stock market overview."""
    if not alpaca_client:
        await update.message.reply_text(
            "📈 <b>STOCKS</b>\n\n"
            "❌ Alpaca not configured.\n\n"
            "Add to Railway:\n"
            "<code>ALPACA_API_KEY=your_key</code>\n"
            "<code>ALPACA_API_SECRET=your_secret</code>\n\n"
            "Get free keys at: alpaca.markets",
            parse_mode="HTML"
        )
        return
    
    msg = await update.message.reply_text("📈 Loading stocks...")
    
    # Check market status
    market_open = await alpaca_client.is_market_open()
    status = "🟢 OPEN" if market_open else "🔴 CLOSED"
    
    # Get prices for top stocks
    text = f"""📈 <b>STOCK MARKET</b>
━━━━━━━━━━━━━━━━━━━━━
Market: {status}

<b>Tech Giants:</b>
"""
    
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    for symbol in tech_stocks:
        price = await alpaca_client.get_stock_price(symbol)
        if price > 0:
            text += f"📊 <b>{symbol}</b>: ${price:,.2f}\n"
    
    text += "\n<b>ETFs:</b>\n"
    etfs = ["SPY", "QQQ", "IWM"]
    for symbol in etfs:
        price = await alpaca_client.get_stock_price(symbol)
        if price > 0:
            text += f"📊 <b>{symbol}</b>: ${price:,.2f}\n"
    
    # Show buying power if available
    buying_power = await alpaca_client.get_buying_power()
    if buying_power > 0:
        text += f"\n💵 Buying Power: ${buying_power:,.2f}"
    
    keyboard = [
        [InlineKeyboardButton("📊 Stock Signals", callback_data="stock_signals")],
        [InlineKeyboardButton("💼 Stock Portfolio", callback_data="stock_portfolio")],
        [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
    ]
    
    await msg.edit_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


async def stockbuy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Buy stocks: /stockbuy AAPL 100"""
    if not alpaca_client:
        await update.message.reply_text("❌ Alpaca not configured")
        return
    
    args = context.args
    if len(args) < 1:
        await update.message.reply_text(
            "📈 <b>BUY STOCKS</b>\n\n"
            "Usage:\n"
            "<code>/stockbuy AAPL</code> - Buy with default $\n"
            "<code>/stockbuy AAPL 100</code> - Buy $100 of AAPL\n"
            "<code>/stockbuy TSLA 500</code> - Buy $500 of TSLA",
            parse_mode="HTML"
        )
        return
    
    symbol = args[0].upper()
    amount = float(args[1]) if len(args) > 1 else TRADE_AMOUNT_USD
    
    # Check if valid stock
    if symbol not in STOCK_SYMBOLS:
        await update.message.reply_text(f"❌ Unknown stock. Available: {', '.join(STOCK_SYMBOLS[:10])}...")
        return
    
    # Check market hours
    market_open = await alpaca_client.is_market_open()
    if not market_open:
        await update.message.reply_text("❌ Market is closed. Try during market hours (9:30 AM - 4:00 PM ET)")
        return
    
    price = await alpaca_client.get_stock_price(symbol)
    shares = amount / price if price > 0 else 0
    
    keyboard = [
        [InlineKeyboardButton(f"✅ Buy ${amount} of {symbol}", callback_data=f"confirm_stockbuy_{symbol}_{amount}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_order")]
    ]
    
    await update.message.reply_text(
        f"📈 <b>CONFIRM STOCK BUY</b>\n\n"
        f"📊 Symbol: <b>{symbol}</b>\n"
        f"💵 Amount: <b>${amount:.2f}</b>\n"
        f"💲 Price: ${price:,.2f}\n"
        f"📈 Shares: ~{shares:.4f}\n\n"
        f"Mode: {'🔴 LIVE' if LIVE_TRADING else '🟡 PAPER'}",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def stocksell_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sell stocks: /stocksell AAPL"""
    if not alpaca_client:
        await update.message.reply_text("❌ Alpaca not configured")
        return
    
    args = context.args
    
    # Show holdings if no args
    if not args:
        holdings = await alpaca_client.get_all_holdings()
        
        if not holdings["assets"]:
            await update.message.reply_text("📭 No stock positions to sell")
            return
        
        text = "📉 <b>SELL STOCKS</b>\n━━━━━━━━━━━━━━━\n\n"
        keyboard = []
        
        for asset in holdings["assets"][:8]:
            symbol = asset["symbol"]
            value = asset["market_value"]
            pl = asset["unrealized_pl"]
            pl_pct = asset["unrealized_plpc"]
            emoji = "🟢" if pl >= 0 else "🔴"
            
            text += f"{emoji} <b>{symbol}</b>: ${value:,.2f} ({pl_pct:+.1f}%)\n"
            keyboard.append([InlineKeyboardButton(f"Sell {symbol}", callback_data=f"stocksell_menu_{symbol}")])
        
        keyboard.append([InlineKeyboardButton("🔙 Menu", callback_data="menu_back")])
        
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    
    symbol = args[0].upper()
    sell_pct = float(args[1]) if len(args) > 1 else 100
    
    # Get position
    holdings = await alpaca_client.get_all_holdings()
    position = next((a for a in holdings["assets"] if a["symbol"] == symbol), None)
    
    if not position:
        await update.message.reply_text(f"❌ No position in {symbol}")
        return
    
    sell_qty = position["qty"] * (sell_pct / 100)
    sell_value = position["market_value"] * (sell_pct / 100)
    
    keyboard = [
        [InlineKeyboardButton(f"✅ Sell {sell_pct}% ({sell_qty:.2f} shares)", callback_data=f"confirm_stocksell_{symbol}_{sell_qty}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_order")]
    ]
    
    await update.message.reply_text(
        f"📉 <b>CONFIRM STOCK SELL</b>\n\n"
        f"📊 Symbol: <b>{symbol}</b>\n"
        f"📈 Shares: {sell_qty:.4f}\n"
        f"💵 Value: ~${sell_value:,.2f}\n"
        f"💲 Price: ${position['current_price']:,.2f}\n"
        f"📊 P&L: {position['unrealized_plpc']:+.1f}%",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# ============ FOREX COMMANDS ============
async def forex_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show forex rates."""
    if not alpaca_client:
        await update.message.reply_text(
            "💱 <b>FOREX</b>\n\n"
            "❌ Alpaca not configured.\n\n"
            "Add to Railway:\n"
            "<code>ALPACA_API_KEY=your_key</code>\n"
            "<code>ALPACA_API_SECRET=your_secret</code>",
            parse_mode="HTML"
        )
        return
    
    msg = await update.message.reply_text("💱 Loading forex rates...")
    
    text = """💱 <b>FOREX RATES</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Major Pairs:</b>
"""
    
    major_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]
    for pair in major_pairs:
        rate = await alpaca_client.get_forex_rate(pair)
        if rate > 0:
            text += f"💱 <b>{pair}</b>: {rate:.5f}\n"
    
    text += "\n<b>Commodity Pairs:</b>\n"
    commodity_pairs = ["AUD/USD", "USD/CAD", "NZD/USD"]
    for pair in commodity_pairs:
        rate = await alpaca_client.get_forex_rate(pair)
        if rate > 0:
            text += f"💱 <b>{pair}</b>: {rate:.5f}\n"
    
    text += "\n<b>Cross Pairs:</b>\n"
    cross_pairs = ["EUR/GBP", "EUR/JPY", "GBP/JPY"]
    for pair in cross_pairs:
        rate = await alpaca_client.get_forex_rate(pair)
        if rate > 0:
            text += f"💱 <b>{pair}</b>: {rate:.5f}\n"
    
    text += "\n<i>Forex trades 24/5 (Sun 5pm - Fri 5pm ET)</i>"
    
    keyboard = [
        [InlineKeyboardButton("📊 Forex Signals", callback_data="forex_signals")],
        [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
    ]
    
    await msg.edit_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ MULTI-ASSET PORTFOLIO ============
async def allportfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show combined portfolio across all asset classes."""
    msg = await update.message.reply_text("📊 Loading all assets...")
    
    # Get crypto holdings
    crypto_value = 0
    crypto_assets = []
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
        crypto_value = holdings.get("crypto_value", 0)
        crypto_assets = holdings.get("assets", [])
    
    # Get stock holdings
    stock_value = 0
    stock_assets = []
    stock_cash = 0
    if alpaca_client:
        holdings = await alpaca_client.get_all_holdings()
        stock_value = holdings.get("portfolio_value", 0) - holdings.get("cash", 0)
        stock_cash = holdings.get("cash", 0)
        stock_assets = holdings.get("assets", [])
    
    total_value = crypto_value + stock_value + stock_cash
    
    text = f"""
╔═══════════════════════════════════════╗
║    🌐 <b>MULTI-ASSET PORTFOLIO</b>          ║
╠═══════════════════════════════════════╣
║                                       ║
║  💰 <b>Total:</b> <code>${total_value:,.2f}</code>               ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>BY ASSET CLASS</b>                      ║
╠═══════════════════════════════════════╣
"""
    
    if crypto_value > 0:
        crypto_pct = (crypto_value / total_value * 100) if total_value > 0 else 0
        text += f"║  🪙 Crypto:  <code>${crypto_value:,.2f}</code> ({crypto_pct:.1f}%)     ║\n"
    
    if stock_value > 0:
        stock_pct = (stock_value / total_value * 100) if total_value > 0 else 0
        text += f"║  📈 Stocks:  <code>${stock_value:,.2f}</code> ({stock_pct:.1f}%)     ║\n"
    
    if stock_cash > 0:
        cash_pct = (stock_cash / total_value * 100) if total_value > 0 else 0
        text += f"║  💵 Cash:    <code>${stock_cash:,.2f}</code> ({cash_pct:.1f}%)     ║\n"
    
    text += "╚═══════════════════════════════════════╝\n"
    
    # Top holdings across all
    text += "\n<b>🏆 TOP HOLDINGS</b>\n"
    
    all_assets = []
    for a in crypto_assets[:5]:
        all_assets.append({"symbol": a["currency"], "value": a["value"], "type": "🪙"})
    for a in stock_assets[:5]:
        all_assets.append({"symbol": a["symbol"], "value": a["market_value"], "type": "📈"})
    
    all_assets.sort(key=lambda x: x["value"], reverse=True)
    
    for asset in all_assets[:8]:
        pct = (asset["value"] / total_value * 100) if total_value > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        text += f"{asset['type']} <b>{asset['symbol']}</b>: ${asset['value']:,.2f} ({pct:.1f}%)\n"
    
    keyboard = [
        [InlineKeyboardButton("🪙 Crypto", callback_data="menu_portfolio"), InlineKeyboardButton("📈 Stocks", callback_data="stock_portfolio")],
        [InlineKeyboardButton("🔄 Refresh", callback_data="all_portfolio"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
    ]
    
    await msg.edit_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show complete portfolio with all Coinbase holdings."""
    msg = await update.message.reply_text("📊 Loading portfolio...")
    
    # Get all holdings from Coinbase
    holdings = {"usd_balance": 0, "crypto_value": 0, "total_value": 0, "assets": []}
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
    
    total_value = holdings["total_value"]
    usd_balance = holdings["usd_balance"]
    crypto_value = holdings["crypto_value"]
    
    # Calculate allocation percentages
    usd_pct = (usd_balance / total_value * 100) if total_value > 0 else 0
    crypto_pct = (crypto_value / total_value * 100) if total_value > 0 else 0
    
    # Build beautiful UI
    text = f"""
╔═══════════════════════════════════════╗
║      💼 <b>PORTFOLIO OVERVIEW</b>            ║
╠═══════════════════════════════════════╣
║                                       ║
║  💰 <b>Total Value:</b>  <code>${total_value:,.2f}</code>        ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>ALLOCATION</b>                          ║
╠═══════════════════════════════════════╣
║                                       ║
║  💵 USD Cash:     <code>${usd_balance:,.2f}</code>  ({usd_pct:.1f}%)   ║
║  🪙 Crypto:       <code>${crypto_value:,.2f}</code>  ({crypto_pct:.1f}%)   ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>HOLDINGS</b>                            ║
╚═══════════════════════════════════════╝
"""
    
    if holdings["assets"]:
        for asset in holdings["assets"][:10]:  # Top 10 holdings
            currency = asset["currency"]
            balance = asset["balance"]
            price = asset["price"]
            value = asset["value"]
            pct = (value / total_value * 100) if total_value > 0 else 0
            
            # Create visual bar
            bar_filled = int(pct / 5)  # Each █ = 5%
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            
            # Check if we have an active position for P&L
            pair = f"{currency}-USD"
            pnl_str = ""
            if pair in positions:
                entry = positions[pair]["entry_price"]
                pnl_pct = ((price - entry) / entry) * 100
                emoji = "🟢" if pnl_pct >= 0 else "🔴"
                pnl_str = f" {emoji}{pnl_pct:+.1f}%"
            
            text += f"""
┌─ <b>{currency}</b>{pnl_str}
│  Balance: <code>{balance:.6f}</code>
│  Price:   <code>${price:,.2f}</code>
│  Value:   <code>${value:,.2f}</code> ({pct:.1f}%)
│  [{bar}]
└───────────────────────────
"""
    else:
        text += "\n<i>No crypto holdings found</i>\n"
    
    # Show bot-tracked positions with entry/P&L
    if positions:
        text += """
╔═══════════════════════════════════════╗
║  🤖 <b>ACTIVE TRADES</b>                    ║
╠═══════════════════════════════════════╣
"""
        total_pnl = 0
        for pair, pos in positions.items():
            current = await get_public_price(pair)
            entry = pos["entry_price"]
            amount = pos.get("amount_usd", 0)
            pnl_pct = ((current - entry) / entry) * 100
            pnl_usd = (pnl_pct / 100) * amount
            total_pnl += pnl_usd
            
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            auto_tag = "🤖" if pos.get("autopilot") else "👤"
            strategy = pos.get("strategy", "manual")[:12]
            
            peak = pos.get("highest_price", entry)
            from_peak = ((current - peak) / peak) * 100 if peak > 0 else 0
            
            text += f"""║
║ {emoji} <b>{pair}</b> {auto_tag}
║   Entry: ${entry:,.2f} → Now: ${current:,.2f}
║   P&L: <b>{pnl_pct:+.2f}%</b> (${pnl_usd:+.2f})
║   Peak: ${peak:,.2f} ({from_peak:+.1f}%)
║   Size: ${amount:.2f} | {strategy}
"""
        
        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        text += f"""╠═══════════════════════════════════════╣
║ {total_emoji} <b>Unrealized P&L:</b> ${total_pnl:+.2f}           ║
╚═══════════════════════════════════════╝
"""
    
    # Show daily stats
    win_rate = (daily_pnl["wins"] / daily_pnl["trades"] * 100) if daily_pnl["trades"] > 0 else 0
    text += f"""
📅 <b>Today:</b> ${daily_pnl['realized']:+.2f} | {daily_pnl['trades']} trades | {win_rate:.0f}% wins
"""
    
    # Add action buttons
    keyboard = []
    if positions:
        keyboard.append([InlineKeyboardButton("🔄 Refresh", callback_data="refresh_portfolio")])
        for pair in list(positions.keys())[:3]:  # Max 3 close buttons
            keyboard.append([InlineKeyboardButton(f"❌ Close {pair}", callback_data=f"close_{pair}")])
    
    keyboard.append([InlineKeyboardButton("📊 Signals", callback_data="get_signals")])
    
    await msg.edit_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None)


async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Daily P&L."""
    win_rate = (daily_pnl["wins"] / daily_pnl["trades"] * 100) if daily_pnl["trades"] > 0 else 0
    emoji = "🟢" if daily_pnl["realized"] >= 0 else "🔴"
    
    text = f"""📈 <b>TODAY'S P&L</b>
━━━━━━━━━━━━━━━

{emoji} Realized: <b>${daily_pnl['realized']:+.2f}</b>
📊 Trades: {daily_pnl['trades']}
✅ Wins: {daily_pnl['wins']}
🎯 Win Rate: {win_rate:.1f}%"""
    
    await update.message.reply_text(text, parse_mode="HTML")


# ============ QUICK BUY COMMAND ============
async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick buy: /buy BTC 50 or /buy BTC"""
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "💰 <b>QUICK BUY</b>\n\n"
            "Usage:\n"
            "<code>/buy BTC</code> - Buy with default amount\n"
            "<code>/buy BTC 50</code> - Buy $50 of BTC\n"
            "<code>/buy ETH 100</code> - Buy $100 of ETH\n\n"
            f"Default amount: ${TRADE_AMOUNT_USD}",
            parse_mode="HTML"
        )
        return
    
    coin = args[0].upper()
    pair = f"{coin}-USD"
    
    # Validate pair
    if pair not in TRADING_PAIRS:
        coins = ", ".join([p.split("-")[0] for p in TRADING_PAIRS])
        await update.message.reply_text(f"❌ Unknown coin. Available: {coins}")
        return
    
    # Get amount
    amount = float(args[1]) if len(args) > 1 else TRADE_AMOUNT_USD
    
    if amount < 1:
        await update.message.reply_text("❌ Minimum buy is $1")
        return
    
    # Check balance
    usd_balance = 0
    if cdp_client:
        usd_balance = await cdp_client.get_usd_balance()
    
    if LIVE_TRADING and amount > usd_balance:
        await update.message.reply_text(
            f"❌ Insufficient balance\n\n"
            f"Requested: ${amount:.2f}\n"
            f"Available: ${usd_balance:.2f}"
        )
        return
    
    price = await get_public_price(pair)
    
    # Confirm
    keyboard = [
        [InlineKeyboardButton(f"✅ Confirm ${amount} Buy", callback_data=f"confirm_buy_{pair}_{amount}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_order")]
    ]
    
    await update.message.reply_text(
        f"💰 <b>CONFIRM BUY</b>\n\n"
        f"📊 Pair: <b>{pair}</b>\n"
        f"💵 Amount: <b>${amount:.2f}</b>\n"
        f"💲 Price: ${price:,.2f}\n"
        f"🪙 You get: ~{amount/price:.6f} {coin}\n\n"
        f"Mode: {'🔴 LIVE' if LIVE_TRADING else '🟡 PAPER'}",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# ============ QUICK SELL COMMAND ============
async def sell_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick sell: /sell BTC or /sell BTC 50%"""
    args = context.args
    
    if not args:
        # Show sellable holdings
        holdings = {"assets": []}
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
        
        if not holdings["assets"]:
            await update.message.reply_text("📭 No crypto holdings to sell.")
            return
        
        text = "💸 <b>QUICK SELL</b>\n━━━━━━━━━━━━━━━\n\n"
        text += "Your holdings:\n\n"
        
        keyboard = []
        for asset in holdings["assets"][:8]:
            currency = asset["currency"]
            value = asset["value"]
            text += f"🪙 <b>{currency}</b>: ${value:,.2f}\n"
            keyboard.append([InlineKeyboardButton(f"Sell {currency}", callback_data=f"sell_menu_{currency}")])
        
        text += "\nUsage:\n"
        text += "<code>/sell BTC</code> - Sell all BTC\n"
        text += "<code>/sell BTC 50</code> - Sell 50% of BTC\n"
        
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    
    coin = args[0].upper()
    pair = f"{coin}-USD"
    sell_pct = float(args[1]) if len(args) > 1 else 100
    sell_pct = max(1, min(100, sell_pct))
    
    # Get holdings
    holdings = {"assets": []}
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
    
    # Find the asset
    asset = next((a for a in holdings["assets"] if a["currency"] == coin), None)
    
    if not asset:
        await update.message.reply_text(f"❌ You don't hold any {coin}")
        return
    
    sell_value = asset["value"] * (sell_pct / 100)
    sell_amount = asset["balance"] * (sell_pct / 100)
    
    keyboard = [
        [InlineKeyboardButton(f"✅ Sell {sell_pct}% (${sell_value:.2f})", callback_data=f"confirm_sell_{pair}_{sell_pct}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_order")]
    ]
    
    # Check P&L if tracked position
    pnl_str = ""
    if pair in positions:
        entry = positions[pair]["entry_price"]
        pnl_pct = ((asset["price"] - entry) / entry) * 100
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        pnl_str = f"\n{emoji} P&L: {pnl_pct:+.2f}%"
    
    await update.message.reply_text(
        f"💸 <b>CONFIRM SELL</b>\n\n"
        f"📊 Pair: <b>{pair}</b>\n"
        f"📉 Selling: <b>{sell_pct}%</b>\n"
        f"🪙 Amount: {sell_amount:.6f} {coin}\n"
        f"💵 Value: ~${sell_value:.2f}\n"
        f"💲 Price: ${asset['price']:,.2f}"
        f"{pnl_str}\n\n"
        f"Mode: {'🔴 LIVE' if LIVE_TRADING else '🟡 PAPER'}",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# ============ LIMIT ORDER COMMAND ============
async def limit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set limit order alert: /limit BTC 95000"""
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            "📋 <b>LIMIT ORDER ALERTS</b>\n\n"
            "Set price alerts that auto-buy when hit:\n\n"
            "<code>/limit BTC 95000</code> - Alert at $95k\n"
            "<code>/limit ETH 3500 100</code> - Alert + buy $100\n\n"
            "<i>Note: These are alerts, not real limit orders.\n"
            "Bot will notify you when price hits target.</i>",
            parse_mode="HTML"
        )
        return
    
    coin = args[0].upper()
    pair = f"{coin}-USD"
    target_price = float(args[1])
    amount = float(args[2]) if len(args) > 2 else TRADE_AMOUNT_USD
    
    if pair not in TRADING_PAIRS:
        await update.message.reply_text("❌ Unknown coin")
        return
    
    current = await get_public_price(pair)
    direction = "below" if target_price < current else "above"
    pct_away = ((target_price - current) / current) * 100
    
    # Store limit alert
    limit_key = f"limit_{pair}_{target_price}"
    price_alerts[limit_key] = {
        "pair": pair,
        "target": target_price,
        "amount": amount,
        "direction": direction,
        "created": datetime.now().isoformat(),
        "type": "limit"
    }
    save_positions()
    
    await update.message.reply_text(
        f"📋 <b>LIMIT ALERT SET</b>\n\n"
        f"📊 Pair: <b>{pair}</b>\n"
        f"🎯 Target: ${target_price:,.2f} ({direction})\n"
        f"💵 Amount: ${amount:.2f}\n"
        f"💲 Current: ${current:,.2f} ({pct_away:+.1f}%)\n\n"
        f"✅ You'll be notified when price hits target.\n\n"
        f"<i>Use /alerts to view all alerts</i>",
        parse_mode="HTML"
    )


# ============ TRADE HISTORY COMMAND ============
async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show trade history."""
    
    if not trade_history:
        await update.message.reply_text(
            "📜 <b>TRADE HISTORY</b>\n\n"
            "No trades recorded yet.\n\n"
            "<i>Trades are tracked when positions are closed.</i>",
            parse_mode="HTML"
        )
        return
    
    # Get recent trades
    recent = trade_history[-20:][::-1]  # Last 20, newest first
    
    text = "📜 <b>TRADE HISTORY</b>\n━━━━━━━━━━━━━━━\n\n"
    
    total_pnl = 0
    wins = 0
    
    for trade in recent:
        pnl = trade.get("pnl_usd", 0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        
        emoji = "🟢" if pnl >= 0 else "🔴"
        date = trade.get("closed_at", trade.get("timestamp", ""))[:10]
        
        text += f"{emoji} <b>{trade.get('pair', 'N/A')}</b> | {date}\n"
        text += f"   ${trade.get('entry', 0):,.2f} → ${trade.get('exit', 0):,.2f}\n"
        text += f"   P&L: ${pnl:+.2f} ({trade.get('pnl_pct', 0):+.1f}%)\n\n"
    
    win_rate = (wins / len(recent) * 100) if recent else 0
    
    text += f"━━━━━━━━━━━━━━━\n"
    text += f"📊 Last {len(recent)} trades\n"
    text += f"{'🟢' if total_pnl >= 0 else '🔴'} Total P&L: ${total_pnl:+.2f}\n"
    text += f"🎯 Win Rate: {win_rate:.1f}%\n\n"
    
    keyboard = [[InlineKeyboardButton("📤 Export CSV", callback_data="export_history")]]
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ PERFORMANCE REPORT COMMAND ============
async def performance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Weekly/monthly performance report."""
    
    # Calculate stats
    total_trades = len(trade_history)
    
    if total_trades == 0:
        await update.message.reply_text(
            "📈 <b>PERFORMANCE REPORT</b>\n\n"
            "No trading data yet.\n\n"
            "<i>Start trading to see performance stats.</i>",
            parse_mode="HTML"
        )
        return
    
    # Calculate metrics
    total_pnl = sum(t.get("pnl_usd", 0) for t in trade_history)
    wins = sum(1 for t in trade_history if t.get("pnl_usd", 0) > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    winning_trades = [t.get("pnl_usd", 0) for t in trade_history if t.get("pnl_usd", 0) > 0]
    losing_trades = [t.get("pnl_usd", 0) for t in trade_history if t.get("pnl_usd", 0) < 0]
    
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
    
    profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
    
    # Get balance
    usd_balance = 0
    total_value = 0
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
        usd_balance = holdings.get("usd_balance", 0)
        total_value = holdings.get("total_value", 0)
    
    # This week's stats
    week_ago = datetime.now() - timedelta(days=7)
    week_trades = [t for t in trade_history if t.get("closed_at", "") > week_ago.isoformat()]
    week_pnl = sum(t.get("pnl_usd", 0) for t in week_trades)
    
    # This month's stats
    month_ago = datetime.now() - timedelta(days=30)
    month_trades = [t for t in trade_history if t.get("closed_at", "") > month_ago.isoformat()]
    month_pnl = sum(t.get("pnl_usd", 0) for t in month_trades)
    
    text = f"""
╔═══════════════════════════════════════╗
║      📈 <b>PERFORMANCE REPORT</b>           ║
╠═══════════════════════════════════════╣
║                                       ║
║  💰 Portfolio: <code>${total_value:,.2f}</code>           ║
║  💵 Cash: <code>${usd_balance:,.2f}</code>                ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>ALL TIME</b>                           ║
╠═══════════════════════════════════════╣
║                                       ║
║  📊 Total Trades: {total_trades}                   ║
║  ✅ Wins: {wins} | ❌ Losses: {losses}             ║
║  🎯 Win Rate: {win_rate:.1f}%                     ║
║  {'🟢' if total_pnl >= 0 else '🔴'} Total P&L: <b>${total_pnl:+,.2f}</b>            ║
║                                       ║
║  📈 Avg Win: ${avg_win:+.2f}                  ║
║  📉 Avg Loss: ${avg_loss:.2f}                 ║
║  ⚖️ Profit Factor: {profit_factor:.2f}               ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>THIS WEEK</b> ({len(week_trades)} trades)             ║
╠═══════════════════════════════════════╣
║  {'🟢' if week_pnl >= 0 else '🔴'} P&L: <b>${week_pnl:+,.2f}</b>                    ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>THIS MONTH</b> ({len(month_trades)} trades)           ║
╠═══════════════════════════════════════╣
║  {'🟢' if month_pnl >= 0 else '🔴'} P&L: <b>${month_pnl:+,.2f}</b>                   ║
║                                       ║
╠═══════════════════════════════════════╣
║  <b>STRATEGY BREAKDOWN</b>                 ║
╚═══════════════════════════════════════╝
"""
    
    # Add strategy performance
    for strat, stats in sorted(strategy_performance.items(), key=lambda x: x[1]["pnl"], reverse=True):
        if stats["trades"] > 0:
            strat_wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
            emoji = "🟢" if stats["pnl"] >= 0 else "🔴"
            text += f"\n{emoji} <b>{strat.replace('_', ' ').title()}</b>\n"
            text += f"   {stats['trades']} trades | {strat_wr:.0f}% WR | ${stats['pnl']:+.2f}\n"
    
    await update.message.reply_text(text, parse_mode="HTML")


# ============ ALERTS COMMAND (Enhanced) ============
async def alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all active alerts including limit orders."""
    
    if not price_alerts:
        await update.message.reply_text(
            "🔔 <b>ALERTS</b>\n\n"
            "No active alerts.\n\n"
            "Set alerts:\n"
            "<code>/alert BTC 100000</code> - Price alert\n"
            "<code>/limit BTC 95000</code> - Limit order alert",
            parse_mode="HTML"
        )
        return
    
    text = "🔔 <b>ACTIVE ALERTS</b>\n━━━━━━━━━━━━━━━\n\n"
    
    for key, alert in price_alerts.items():
        alert_type = alert.get("type", "price")
        pair = alert.get("pair", "")
        target = alert.get("target", alert.get("price", 0))
        
        current = await get_public_price(pair)
        pct_away = ((target - current) / current) * 100 if current else 0
        
        if alert_type == "limit":
            text += f"📋 <b>{pair}</b> LIMIT\n"
            text += f"   Target: ${target:,.2f} ({pct_away:+.1f}%)\n"
            text += f"   Amount: ${alert.get('amount', 0):.2f}\n\n"
        else:
            direction = alert.get("direction", "above" if target > current else "below")
            text += f"🔔 <b>{pair}</b>\n"
            text += f"   Alert {direction} ${target:,.2f} ({pct_away:+.1f}%)\n\n"
    
    keyboard = [[InlineKeyboardButton("🗑️ Clear All Alerts", callback_data="clear_all_alerts")]]
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ RISK MANAGEMENT ============
async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risk management dashboard."""
    args = context.args
    
    # Get current portfolio value
    total_value = 0
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
        total_value = holdings.get("total_value", 0)
    
    # Calculate current drawdown
    if total_value > risk_state["highest_balance"]:
        risk_state["highest_balance"] = total_value
    
    drawdown = 0
    if risk_state["highest_balance"] > 0:
        drawdown = ((risk_state["highest_balance"] - total_value) / risk_state["highest_balance"]) * 100
    risk_state["current_drawdown"] = drawdown
    
    # Calculate position concentration
    position_pcts = []
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
        for asset in holdings.get("assets", []):
            if total_value > 0:
                pct = (asset["value"] / total_value) * 100
                position_pcts.append({"coin": asset["currency"], "pct": pct})
    
    position_pcts.sort(key=lambda x: x["pct"], reverse=True)
    
    # Check if paused
    paused = risk_state.get("paused_reason")
    
    text = f"""🛡️ <b>RISK MANAGEMENT</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Status:</b> {'🔴 PAUSED - ' + paused if paused else '🟢 ACTIVE'}

━━━━━━━━━━━━━━━━━━━━━
<b>💰 PORTFOLIO</b>
━━━━━━━━━━━━━━━━━━━━━
Current Value: <code>${total_value:,.2f}</code>
Peak Value: <code>${risk_state['highest_balance']:,.2f}</code>
Drawdown: <code>{drawdown:.1f}%</code> (max {risk_state['max_drawdown_pct']}%)

━━━━━━━━━━━━━━━━━━━━━
<b>📊 DAILY LIMITS</b>
━━━━━━━━━━━━━━━━━━━━━
Today's P&L: <code>${daily_pnl['realized']:+,.2f}</code>
Loss Limit: <code>${risk_state['daily_loss_limit']}</code>
{'🔴 LIMIT HIT!' if daily_pnl['realized'] <= -risk_state['daily_loss_limit'] else '🟢 Within limits'}

━━━━━━━━━━━━━━━━━━━━━
<b>📈 CONCENTRATION</b>
━━━━━━━━━━━━━━━━━━━━━
Max Position: {risk_state['max_position_pct']}%

"""
    for pos in position_pcts[:5]:
        bar = "█" * int(pos["pct"] / 5) + "░" * (20 - int(pos["pct"] / 5))
        warn = "⚠️" if pos["pct"] > risk_state["max_position_pct"] else ""
        text += f"{pos['coin']}: {pos['pct']:.1f}% {warn}\n[{bar}]\n"
    
    keyboard = [
        [InlineKeyboardButton("━━━ DRAWDOWN LIMIT ━━━", callback_data="none")],
        [
            InlineKeyboardButton("5%", callback_data="risk_dd_5"),
            InlineKeyboardButton("10%", callback_data="risk_dd_10"),
            InlineKeyboardButton("15%", callback_data="risk_dd_15"),
            InlineKeyboardButton("20%", callback_data="risk_dd_20")
        ],
        [InlineKeyboardButton("━━━ DAILY LOSS LIMIT ━━━", callback_data="none")],
        [
            InlineKeyboardButton("$25", callback_data="risk_dll_25"),
            InlineKeyboardButton("$50", callback_data="risk_dll_50"),
            InlineKeyboardButton("$100", callback_data="risk_dll_100"),
            InlineKeyboardButton("$200", callback_data="risk_dll_200")
        ],
        [InlineKeyboardButton("━━━ MAX POSITION ━━━", callback_data="none")],
        [
            InlineKeyboardButton("20%", callback_data="risk_mp_20"),
            InlineKeyboardButton("30%", callback_data="risk_mp_30"),
            InlineKeyboardButton("40%", callback_data="risk_mp_40"),
            InlineKeyboardButton("50%", callback_data="risk_mp_50")
        ],
        [InlineKeyboardButton("🔄 Reset Peak", callback_data="risk_reset_peak")],
        [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
    ]
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ WATCHLIST ============
async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manage watchlist."""
    args = context.args
    
    if args and len(args) >= 1:
        action = args[0].lower()
        
        if action == "add" and len(args) >= 2:
            coin = args[1].upper()
            pair = f"{coin}-USD"
            target = float(args[2]) if len(args) > 2 else None
            
            watchlist[pair] = {
                "added": datetime.now().isoformat(),
                "target_price": target,
                "notes": " ".join(args[3:]) if len(args) > 3 else None
            }
            save_positions()
            await update.message.reply_text(f"✅ Added {coin} to watchlist")
            return
        
        elif action == "remove" and len(args) >= 2:
            coin = args[1].upper()
            pair = f"{coin}-USD"
            if pair in watchlist:
                del watchlist[pair]
                save_positions()
                await update.message.reply_text(f"✅ Removed {coin} from watchlist")
            else:
                await update.message.reply_text(f"❌ {coin} not in watchlist")
            return
    
    # Show watchlist
    text = """👀 <b>WATCHLIST</b>
━━━━━━━━━━━━━━━━━━━━━

"""
    
    if not watchlist:
        text += "<i>Empty watchlist</i>\n\n"
        text += "Add coins:\n<code>/watchlist add BTC</code>\n<code>/watchlist add ETH 4000</code>"
    else:
        for pair, data in watchlist.items():
            price = await get_public_price(pair)
            coin = pair.split("-")[0]
            target = data.get("target_price")
            
            if target:
                pct_to_target = ((target - price) / price) * 100
                target_str = f"\n   🎯 Target: ${target:,.2f} ({pct_to_target:+.1f}%)"
            else:
                target_str = ""
            
            text += f"🪙 <b>{coin}</b>: ${price:,.2f}{target_str}\n\n"
    
    keyboard = []
    for pair in list(watchlist.keys())[:4]:
        coin = pair.split("-")[0]
        keyboard.append([
            InlineKeyboardButton(f"Buy {coin}", callback_data=f"quickbuy_{coin}"),
            InlineKeyboardButton(f"Remove {coin}", callback_data=f"wl_remove_{coin}")
        ])
    
    keyboard.append([InlineKeyboardButton("🔙 Menu", callback_data="menu_back")])
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ TAKE PROFIT TIERS ============
async def tptiers_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Configure take profit tiers."""
    
    text = f"""🎯 <b>TAKE PROFIT TIERS</b>
━━━━━━━━━━━━━━━━━━━━━

Status: {'🟢 ENABLED' if tp_tiers['enabled'] else '🔴 DISABLED'}

<b>How it works:</b>
Instead of one big take profit, sell in stages:

"""
    
    remaining = 100
    for i, tier in enumerate(tp_tiers["tiers"], 1):
        text += f"📊 Tier {i}: At +{tier['pct_gain']}% → Sell {tier['sell_pct']}%\n"
        remaining -= tier["sell_pct"]
    
    text += f"\n🔄 Remaining {remaining}% rides with trailing stop\n"
    
    text += """
━━━━━━━━━━━━━━━━━━━━━
<b>Example:</b>
Buy $100 of BTC at $95,000

• +5% ($99,750): Sell $25 ✅
• +10% ($104,500): Sell $25 ✅
• +20% ($114,000): Sell $25 ✅
• Trailing stop protects final $25
"""
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'🔴 Disable' if tp_tiers['enabled'] else '🟢 Enable'} Tiers",
            callback_data="tp_toggle"
        )],
        [InlineKeyboardButton("Conservative", callback_data="tp_preset_conservative")],
        [InlineKeyboardButton("Balanced", callback_data="tp_preset_balanced")],
        [InlineKeyboardButton("Aggressive", callback_data="tp_preset_aggressive")],
        [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
    ]
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ DCA AUTOPILOT ============
async def dcaauto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """DCA autopilot settings."""
    
    text = f"""📉 <b>DCA AUTOPILOT</b>
━━━━━━━━━━━━━━━━━━━━━

Status: {'🟢 ENABLED' if dca_settings['enabled'] else '🔴 DISABLED'}

<b>Settings:</b>
• Drop trigger: {dca_settings['drop_pct']}%
• Amount per buy: ${dca_settings['amount_usd']}
• Max buys/day: {dca_settings['max_per_day']}
• Today's buys: {dca_settings['today_count']}

<b>Watched Coins:</b>
{', '.join([p.split('-')[0] for p in dca_settings['coins']])}

<b>How it works:</b>
When a coin drops {dca_settings['drop_pct']}% from 24h high,
auto-buy ${dca_settings['amount_usd']} (up to {dca_settings['max_per_day']}x/day)
"""
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'🔴 Disable' if dca_settings['enabled'] else '🟢 Enable'} DCA",
            callback_data="dca_toggle"
        )],
        [InlineKeyboardButton("━━━ DROP TRIGGER ━━━", callback_data="none")],
        [
            InlineKeyboardButton("3%", callback_data="dca_drop_3"),
            InlineKeyboardButton("5%", callback_data="dca_drop_5"),
            InlineKeyboardButton("7%", callback_data="dca_drop_7"),
            InlineKeyboardButton("10%", callback_data="dca_drop_10")
        ],
        [InlineKeyboardButton("━━━ BUY AMOUNT ━━━", callback_data="none")],
        [
            InlineKeyboardButton("$10", callback_data="dca_amt_10"),
            InlineKeyboardButton("$25", callback_data="dca_amt_25"),
            InlineKeyboardButton("$50", callback_data="dca_amt_50"),
            InlineKeyboardButton("$100", callback_data="dca_amt_100")
        ],
        [InlineKeyboardButton("━━━ MAX/DAY ━━━", callback_data="none")],
        [
            InlineKeyboardButton("1", callback_data="dca_max_1"),
            InlineKeyboardButton("3", callback_data="dca_max_3"),
            InlineKeyboardButton("5", callback_data="dca_max_5"),
            InlineKeyboardButton("10", callback_data="dca_max_10")
        ],
        [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
    ]
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# ============ DAILY SUMMARY ============
async def summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate daily summary."""
    await send_daily_summary(update.effective_chat.id)


async def send_daily_summary(chat_id: int):
    """Send daily summary report."""
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    # Get portfolio value
    total_value = 0
    usd_balance = 0
    crypto_value = 0
    if cdp_client:
        holdings = await cdp_client.get_all_holdings()
        total_value = holdings.get("total_value", 0)
        usd_balance = holdings.get("usd_balance", 0)
        crypto_value = holdings.get("crypto_value", 0)
    
    # Today's trades
    today_trades = [t for t in trade_history if t.get("closed_at", "").startswith(now.date().isoformat())]
    today_pnl = sum(t.get("pnl_usd", 0) for t in today_trades)
    today_wins = sum(1 for t in today_trades if t.get("pnl_usd", 0) > 0)
    today_wr = (today_wins / len(today_trades) * 100) if today_trades else 0
    
    # Best/worst today
    best_today = max(today_trades, key=lambda x: x.get("pnl_usd", 0)) if today_trades else None
    worst_today = min(today_trades, key=lambda x: x.get("pnl_usd", 0)) if today_trades else None
    
    # Open positions P&L
    open_pnl = 0
    for pair, pos in positions.items():
        price = await get_public_price(pair)
        entry = pos["entry_price"]
        pnl = ((price - entry) / entry) * pos.get("amount_usd", 0) / 100
        open_pnl += pnl
    
    text = f"""📊 <b>DAILY SUMMARY</b>
{now.strftime("%A, %B %d, %Y")}
━━━━━━━━━━━━━━━━━━━━━

<b>💰 PORTFOLIO</b>
Total: <code>${total_value:,.2f}</code>
Cash: ${usd_balance:,.2f} | Crypto: ${crypto_value:,.2f}

━━━━━━━━━━━━━━━━━━━━━
<b>📈 TODAY'S PERFORMANCE</b>
━━━━━━━━━━━━━━━━━━━━━
Realized P&L: <b>${today_pnl:+,.2f}</b>
Unrealized P&L: ${open_pnl:+,.2f}
Trades: {len(today_trades)} | Win Rate: {today_wr:.0f}%
"""
    
    if best_today:
        text += f"\n🏆 Best: {best_today['pair']} ${best_today['pnl_usd']:+.2f}"
    if worst_today and worst_today.get("pnl_usd", 0) < 0:
        text += f"\n💔 Worst: {worst_today['pair']} ${worst_today['pnl_usd']:.2f}"
    
    text += f"""

━━━━━━━━━━━━━━━━━━━━━
<b>🤖 AUTOPILOT</b>
━━━━━━━━━━━━━━━━━━━━━
Status: {'🟢 ON' if autopilot_settings['enabled'] else '🔴 OFF'}
Auto Trades: {autopilot_settings['trades_today']}

━━━━━━━━━━━━━━━━━━━━━
<b>📊 OPEN POSITIONS</b>
━━━━━━━━━━━━━━━━━━━━━
"""
    
    if positions:
        for pair, pos in list(positions.items())[:5]:
            price = await get_public_price(pair)
            entry = pos["entry_price"]
            pnl_pct = ((price - entry) / entry) * 100
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            text += f"{emoji} {pair.split('-')[0]}: {pnl_pct:+.1f}%\n"
    else:
        text += "No open positions\n"
    
    text += "\n<i>Good night! 🌙</i>"
    
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")


# ============ PERFORMANCE CHARTS ============
async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate performance chart."""
    msg = await update.message.reply_text("📊 Generating chart...")
    
    if not trade_history:
        await msg.edit_text("📊 No trade data for chart yet.")
        return
    
    # Calculate cumulative P&L
    cumulative = []
    running_total = 0
    for trade in trade_history[-50:]:  # Last 50 trades
        running_total += trade.get("pnl_usd", 0)
        cumulative.append(running_total)
    
    # Generate text-based chart
    if cumulative:
        max_val = max(cumulative) if max(cumulative) > 0 else 1
        min_val = min(cumulative) if min(cumulative) < 0 else 0
        range_val = max_val - min_val if max_val != min_val else 1
        
        chart_height = 10
        chart_width = min(len(cumulative), 30)
        
        # Sample data if too many points
        step = max(1, len(cumulative) // chart_width)
        sampled = cumulative[::step][:chart_width]
        
        text = f"""📈 <b>P&L CHART</b>
━━━━━━━━━━━━━━━━━━━━━

Last {len(trade_history)} trades:

"""
        
        # Create ASCII chart
        for row in range(chart_height, -1, -1):
            threshold = min_val + (range_val * row / chart_height)
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "█"
                elif row == chart_height // 2 and min_val <= 0 <= max_val:
                    line += "─"
                else:
                    line += " "
            
            if row == chart_height:
                text += f"<code>${max_val:+.0f} |{line}</code>\n"
            elif row == 0:
                text += f"<code>${min_val:+.0f} |{line}</code>\n"
            elif row == chart_height // 2:
                text += f"<code> $0  |{line}</code>\n"
            else:
                text += f"<code>     |{line}</code>\n"
        
        text += f"""
━━━━━━━━━━━━━━━━━━━━━
Final P&L: <b>${running_total:+,.2f}</b>
Trades: {len(trade_history)}
"""
    else:
        text = "📊 Not enough data for chart"
    
    await msg.edit_text(text, parse_mode="HTML")


# ============ STREAK TRACKING ============
async def streak_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show win/loss streaks."""
    
    # Calculate current streak
    current_streak = 0
    streak_type = None
    
    for trade in reversed(trade_history):
        pnl = trade.get("pnl_usd", 0)
        if streak_type is None:
            streak_type = "win" if pnl > 0 else "loss"
            current_streak = 1
        elif (streak_type == "win" and pnl > 0) or (streak_type == "loss" and pnl <= 0):
            current_streak += 1
        else:
            break
    
    # Calculate best/worst streaks
    best_streak = 0
    worst_streak = 0
    temp_win = 0
    temp_loss = 0
    
    for trade in trade_history:
        if trade.get("pnl_usd", 0) > 0:
            temp_win += 1
            temp_loss = 0
            best_streak = max(best_streak, temp_win)
        else:
            temp_loss += 1
            temp_win = 0
            worst_streak = max(worst_streak, temp_loss)
    
    perf_stats["current_streak"] = current_streak if streak_type == "win" else -current_streak
    perf_stats["best_streak"] = best_streak
    perf_stats["worst_streak"] = worst_streak
    
    streak_emoji = "🔥" * min(current_streak, 5) if streak_type == "win" else "❄️" * min(current_streak, 5)
    
    text = f"""🎰 <b>STREAK TRACKER</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Current Streak:</b>
{streak_emoji}
{current_streak} {'WINS' if streak_type == 'win' else 'LOSSES'} in a row!

━━━━━━━━━━━━━━━━━━━━━
<b>Records:</b>
🏆 Best Win Streak: {best_streak}
💔 Worst Loss Streak: {worst_streak}

━━━━━━━━━━━━━━━━━━━━━
<b>Recent Results:</b>
"""
    
    for trade in trade_history[-10:][::-1]:
        emoji = "✅" if trade.get("pnl_usd", 0) > 0 else "❌"
        text += emoji
    
    text += "\n\n<i>Keep the streak going! 🚀</i>"
    
    await update.message.reply_text(text, parse_mode="HTML")


# ============ BEST/WORST TRADES ============  
async def besttrades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show best and worst trades."""
    
    if not trade_history:
        await update.message.reply_text("📊 No trades yet.")
        return
    
    # Sort by P&L
    sorted_trades = sorted(trade_history, key=lambda x: x.get("pnl_usd", 0), reverse=True)
    
    best_5 = sorted_trades[:5]
    worst_5 = sorted_trades[-5:][::-1]
    
    # Update perf stats
    if best_5:
        perf_stats["best_trade"] = best_5[0]
    if worst_5:
        perf_stats["worst_trade"] = worst_5[-1]
    
    text = """🏆 <b>BEST & WORST TRADES</b>
━━━━━━━━━━━━━━━━━━━━━

<b>🥇 TOP 5 WINNERS</b>
"""
    
    for i, trade in enumerate(best_5, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        text += f"{medal} {trade.get('pair', 'N/A')}: <b>${trade.get('pnl_usd', 0):+.2f}</b> ({trade.get('pnl_pct', 0):+.1f}%)\n"
    
    text += """
━━━━━━━━━━━━━━━━━━━━━
<b>💔 TOP 5 LOSERS</b>
"""
    
    for i, trade in enumerate(worst_5, 1):
        text += f"  {trade.get('pair', 'N/A')}: <b>${trade.get('pnl_usd', 0):.2f}</b> ({trade.get('pnl_pct', 0):.1f}%)\n"
    
    # Stats
    total_wins = sum(1 for t in trade_history if t.get("pnl_usd", 0) > 0)
    total_losses = len(trade_history) - total_wins
    avg_win = sum(t.get("pnl_usd", 0) for t in trade_history if t.get("pnl_usd", 0) > 0) / total_wins if total_wins else 0
    avg_loss = sum(t.get("pnl_usd", 0) for t in trade_history if t.get("pnl_usd", 0) <= 0) / total_losses if total_losses else 0
    
    text += f"""
━━━━━━━━━━━━━━━━━━━━━
<b>📊 STATISTICS</b>
Total Trades: {len(trade_history)}
Wins: {total_wins} | Losses: {total_losses}
Avg Win: ${avg_win:+.2f}
Avg Loss: ${avg_loss:.2f}
"""
    
    await update.message.reply_text(text, parse_mode="HTML")


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Settings."""
    text = f"""⚙️ <b>SETTINGS</b>
━━━━━━━━━━━━━━━

💵 Trade: ${TRADE_AMOUNT_USD}
🛑 Stop Loss: {STOP_LOSS_PCT}%
🎯 Take Profit: {TAKE_PROFIT_PCT}%
📈 Trailing Stop: {TRAILING_STOP_PCT}%
📉 DCA Trigger: {DCA_DROP_PCT}%

<b>Status:</b>
{"✅" if LIVE_TRADING else "🟡"} Trading: {"LIVE" if LIVE_TRADING else "Paper"}
{"🤖" if autopilot_settings["enabled"] else "👤"} Mode: {"AUTOPILOT" if autopilot_settings["enabled"] else "Manual"}
{"✅" if cdp_client else "❌"} Coinbase: {"Connected" if cdp_client else "Not configured"}
{"✅" if redis_client else "⚠️"} Redis: {"Connected" if redis_client else "Memory only"}"""
    
    await update.message.reply_text(text, parse_mode="HTML")


# ============ AUTOPILOT MODE ============
async def autopilot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle autopilot mode."""
    args = context.args
    
    if not args:
        # Get current USD balance
        usd_balance = 0
        if cdp_client:
            usd_balance = await cdp_client.get_usd_balance()
        
        status = "🟢 ACTIVE" if autopilot_settings["enabled"] else "🔴 OFF"
        
        text = f"""🤖 <b>AUTOPILOT MODE</b>
━━━━━━━━━━━━━━━

Status: {status}
💰 USD Balance: <b>${usd_balance:,.2f}</b>

<b>Trade Settings:</b>
• Trade size: {autopilot_settings['trade_percentage']}% of balance
• Min trade: ${autopilot_settings['min_trade_usd']}
• Max trade: ${autopilot_settings['max_trade_usd']}
• Reinvest profits: {'✅ Yes' if autopilot_settings['reinvest_profits'] else '❌ No'}

<b>Limits:</b>
• Max daily trades: {autopilot_settings['max_daily_trades']}
• Min confidence: {autopilot_settings['min_confidence']}%
• Trades today: {autopilot_settings['trades_today']}

<b>Performance:</b>
• Total profit: ${autopilot_settings['total_profit']:+,.2f}

<b>How it works:</b>
1. Uses {autopilot_settings['trade_percentage']}% of your USD balance per trade
2. Auto-buys signals with {autopilot_settings['min_confidence']}%+ confidence
3. Profits are reinvested for compounding
4. Stop loss & trailing stop protect positions

<b>Commands:</b>
/autopilot on - Enable
/autopilot off - Disable
/autopilot max 10 - Max daily trades
/autopilot conf 80 - Min confidence %
/autopilot pct 25 - Trade % of balance
/autopilot minusd 10 - Min trade $
/autopilot maxusd 200 - Max trade $

<i>⚠️ Trades real money when LIVE_TRADING=true</i>"""
        
        await update.message.reply_text(text, parse_mode="HTML")
        return
    
    cmd = args[0].lower()
    
    if cmd == "on":
        # Check balance first
        usd_balance = 0
        if cdp_client:
            usd_balance = await cdp_client.get_usd_balance()
        
        if LIVE_TRADING and usd_balance < autopilot_settings['min_trade_usd']:
            await update.message.reply_text(
                f"⚠️ <b>Low Balance</b>\n\n"
                f"USD Balance: ${usd_balance:,.2f}\n"
                f"Min trade size: ${autopilot_settings['min_trade_usd']}\n\n"
                f"Add funds to Coinbase or lower min with:\n"
                f"/autopilot minusd 5",
                parse_mode="HTML"
            )
            return
        
        autopilot_settings["enabled"] = True
        autopilot_settings["trades_today"] = 0
        save_data("autopilot", autopilot_settings)
        
        trade_size = min(
            autopilot_settings['max_trade_usd'],
            max(autopilot_settings['min_trade_usd'], usd_balance * autopilot_settings['trade_percentage'] / 100)
        )
        
        await update.message.reply_text(
            f"🤖 <b>AUTOPILOT ENABLED</b>\n\n"
            f"💰 USD Balance: ${usd_balance:,.2f}\n"
            f"📊 Trade size: ~${trade_size:,.2f} ({autopilot_settings['trade_percentage']}%)\n"
            f"🎯 Min confidence: {autopilot_settings['min_confidence']}%\n"
            f"📈 Max trades/day: {autopilot_settings['max_daily_trades']}\n"
            f"♻️ Reinvest profits: {'Yes' if autopilot_settings['reinvest_profits'] else 'No'}\n\n"
            f"Mode: {'💰 LIVE' if LIVE_TRADING else '📝 PAPER'}\n\n"
            f"<i>Bot will auto-trade high confidence signals!</i>",
            parse_mode="HTML"
        )
    
    elif cmd == "off":
        autopilot_settings["enabled"] = False
        save_data("autopilot", autopilot_settings)
        
        await update.message.reply_text(
            f"👤 <b>AUTOPILOT DISABLED</b>\n\n"
            f"Total profit this session: ${autopilot_settings['total_profit']:+,.2f}\n\n"
            f"Use /signals for manual trading.",
            parse_mode="HTML"
        )
    
    elif cmd == "max" and len(args) > 1:
        try:
            max_trades = int(args[1])
            autopilot_settings["max_daily_trades"] = max(1, min(50, max_trades))
            save_data("autopilot", autopilot_settings)
            await update.message.reply_text(f"✅ Max daily trades: {autopilot_settings['max_daily_trades']}")
        except:
            await update.message.reply_text("❌ Invalid. Use: /autopilot max 10")
    
    elif cmd == "conf" and len(args) > 1:
        try:
            conf = int(args[1])
            autopilot_settings["min_confidence"] = max(50, min(95, conf))
            save_data("autopilot", autopilot_settings)
            await update.message.reply_text(f"✅ Min confidence: {autopilot_settings['min_confidence']}%")
        except:
            await update.message.reply_text("❌ Invalid. Use: /autopilot conf 80")
    
    elif cmd == "pct" and len(args) > 1:
        try:
            pct = int(args[1])
            autopilot_settings["trade_percentage"] = max(5, min(50, pct))
            save_data("autopilot", autopilot_settings)
            await update.message.reply_text(f"✅ Trade percentage: {autopilot_settings['trade_percentage']}% of balance")
        except:
            await update.message.reply_text("❌ Invalid. Use: /autopilot pct 25")
    
    elif cmd == "minusd" and len(args) > 1:
        try:
            min_usd = float(args[1])
            autopilot_settings["min_trade_usd"] = max(1, min(100, min_usd))
            save_data("autopilot", autopilot_settings)
            await update.message.reply_text(f"✅ Min trade: ${autopilot_settings['min_trade_usd']}")
        except:
            await update.message.reply_text("❌ Invalid. Use: /autopilot minusd 10")
    
    elif cmd == "maxusd" and len(args) > 1:
        try:
            max_usd = float(args[1])
            autopilot_settings["max_trade_usd"] = max(10, min(1000, max_usd))
            save_data("autopilot", autopilot_settings)
            await update.message.reply_text(f"✅ Max trade: ${autopilot_settings['max_trade_usd']}")
        except:
            await update.message.reply_text("❌ Invalid. Use: /autopilot maxusd 200")
    
    else:
        await update.message.reply_text(
            "❌ Unknown command.\n\n"
            "Use: /autopilot on/off/max/conf/pct/minusd/maxusd"
        )


async def pause_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pause all trading."""
    autopilot_settings["enabled"] = False
    save_data("autopilot", autopilot_settings)
    
    await update.message.reply_text(
        "⏸️ <b>TRADING PAUSED</b>\n\n"
        "• Autopilot disabled\n"
        "• Stop loss still active for open positions\n\n"
        "Use /autopilot on to resume",
        parse_mode="HTML"
    )


# ============ CALLBACK HANDLER ============
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    # Menu button handlers
    if data == "menu_signals":
        await query.edit_message_text("🔄 Generating signals...")
        signals = await generate_trading_signals(include_news=False)
        text = "📊 <b>AI SIGNALS</b>\n━━━━━━━━━━━━━━━\n\n"
        if signals.get("signals"):
            for sig in signals["signals"][:3]:
                direction = sig.get("direction", sig.get("action", "HOLD"))
                emoji = "🟢" if direction == "BUY" else "🔴"
                text += f"{emoji} <b>{sig.get('pair')}</b>: {direction} ({sig.get('confidence')}%)\n"
                text += f"   Entry: ${sig.get('entry_price', 0):,.2f}\n"
                text += f"   EV: {sig.get('expected_value', 0):.1f}%\n\n"
        else:
            text += "⚪ No high-confidence signals\n"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="menu_signals"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_portfolio":
        await query.edit_message_text("📊 Loading portfolio...")
        holdings = {"usd_balance": 0, "crypto_value": 0, "total_value": 0, "assets": []}
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
        text = f"💼 <b>PORTFOLIO</b>\n━━━━━━━━━━━━━━━\n\n"
        text += f"💰 Total: <b>${holdings['total_value']:,.2f}</b>\n"
        text += f"💵 USD: ${holdings['usd_balance']:,.2f}\n"
        text += f"🪙 Crypto: ${holdings['crypto_value']:,.2f}\n\n"
        for asset in holdings["assets"][:5]:
            text += f"• <b>{asset['currency']}</b>: ${asset['value']:,.2f}\n"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="menu_portfolio"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_buy":
        text = "💰 <b>QUICK BUY</b>\n━━━━━━━━━━━━━━━\n\nSelect coin to buy:"
        keyboard = [
            [InlineKeyboardButton("BTC", callback_data="quickbuy_BTC"), InlineKeyboardButton("ETH", callback_data="quickbuy_ETH"), InlineKeyboardButton("SOL", callback_data="quickbuy_SOL")],
            [InlineKeyboardButton("XRP", callback_data="quickbuy_XRP"), InlineKeyboardButton("DOGE", callback_data="quickbuy_DOGE"), InlineKeyboardButton("ADA", callback_data="quickbuy_ADA")],
            [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
        ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_sell":
        holdings = {"assets": []}
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
        text = "💸 <b>QUICK SELL</b>\n━━━━━━━━━━━━━━━\n\n"
        keyboard = []
        for asset in holdings["assets"][:6]:
            text += f"• {asset['currency']}: ${asset['value']:,.2f}\n"
            keyboard.append([InlineKeyboardButton(f"Sell {asset['currency']}", callback_data=f"sell_menu_{asset['currency']}")])
        keyboard.append([InlineKeyboardButton("🔙 Menu", callback_data="menu_back")])
        if not holdings["assets"]:
            text += "No holdings to sell"
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_market":
        text = "📈 <b>LIVE MARKET</b>\n━━━━━━━━━━━━━━━\n\n"
        for pair in TRADING_PAIRS[:8]:
            price = await get_public_price(pair)
            text += f"🪙 <b>{pair.split('-')[0]}</b>: ${price:,.2f}\n"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="menu_market"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_stocks":
        if not alpaca_client:
            text = "📈 <b>STOCKS</b>\n\n❌ Alpaca not configured.\n\nAdd to Railway:\n<code>ALPACA_API_KEY</code>\n<code>ALPACA_API_SECRET</code>"
            keyboard = [[InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        else:
            market_open = await alpaca_client.is_market_open()
            status = "🟢 OPEN" if market_open else "🔴 CLOSED"
            
            text = f"📈 <b>STOCK MARKET</b>\n━━━━━━━━━━━━━━━\nStatus: {status}\n\n<b>Tech:</b>\n"
            
            for sym in ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]:
                price = await alpaca_client.get_stock_price(sym)
                if price > 0:
                    text += f"📊 <b>{sym}</b>: ${price:,.2f}\n"
            
            text += "\n<b>ETFs:</b>\n"
            for sym in ["SPY", "QQQ"]:
                price = await alpaca_client.get_stock_price(sym)
                if price > 0:
                    text += f"📊 <b>{sym}</b>: ${price:,.2f}\n"
            
            buying_power = await alpaca_client.get_buying_power()
            text += f"\n💵 Buying Power: ${buying_power:,.2f}"
            
            keyboard = [
                [InlineKeyboardButton("💰 Buy Stock", callback_data="stock_buy_menu"), InlineKeyboardButton("💸 Sell Stock", callback_data="stock_sell_menu")],
                [InlineKeyboardButton("💼 Stock Portfolio", callback_data="stock_portfolio")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="menu_stocks"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
            ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_forex":
        if not alpaca_client:
            text = "💱 <b>FOREX</b>\n\n❌ Alpaca not configured.\n\nAdd to Railway:\n<code>ALPACA_API_KEY</code>\n<code>ALPACA_API_SECRET</code>"
            keyboard = [[InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        else:
            text = "💱 <b>FOREX RATES</b>\n━━━━━━━━━━━━━━━\n\n<b>Major Pairs:</b>\n"
            
            for pair in ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]:
                rate = await alpaca_client.get_forex_rate(pair)
                if rate > 0:
                    text += f"💱 <b>{pair}</b>: {rate:.5f}\n"
            
            text += "\n<b>Commodity:</b>\n"
            for pair in ["AUD/USD", "USD/CAD", "NZD/USD"]:
                rate = await alpaca_client.get_forex_rate(pair)
                if rate > 0:
                    text += f"💱 <b>{pair}</b>: {rate:.5f}\n"
            
            text += "\n<i>24/5 (Sun 5pm - Fri 5pm ET)</i>"
            
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="menu_forex"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
            ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "stock_portfolio":
        if not alpaca_client:
            await query.edit_message_text("❌ Alpaca not configured")
            return
        
        holdings = await alpaca_client.get_all_holdings()
        
        text = f"💼 <b>STOCK PORTFOLIO</b>\n━━━━━━━━━━━━━━━\n\n"
        text += f"💰 Portfolio: <b>${holdings['portfolio_value']:,.2f}</b>\n"
        text += f"💵 Cash: ${holdings['cash']:,.2f}\n\n"
        
        if holdings["assets"]:
            text += "<b>Positions:</b>\n"
            for asset in holdings["assets"][:8]:
                emoji = "🟢" if asset["unrealized_pl"] >= 0 else "🔴"
                text += f"{emoji} <b>{asset['symbol']}</b>: ${asset['market_value']:,.2f} ({asset['unrealized_plpc']:+.1f}%)\n"
        else:
            text += "<i>No positions</i>"
        
        keyboard = [
            [InlineKeyboardButton("💰 Buy", callback_data="stock_buy_menu"), InlineKeyboardButton("💸 Sell", callback_data="stock_sell_menu")],
            [InlineKeyboardButton("🔄 Refresh", callback_data="stock_portfolio"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
        ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "stock_buy_menu":
        text = "💰 <b>BUY STOCKS</b>\n━━━━━━━━━━━━━━━\n\nSelect stock:"
        keyboard = [
            [InlineKeyboardButton("AAPL", callback_data="stockbuy_AAPL"), InlineKeyboardButton("MSFT", callback_data="stockbuy_MSFT"), InlineKeyboardButton("NVDA", callback_data="stockbuy_NVDA")],
            [InlineKeyboardButton("GOOGL", callback_data="stockbuy_GOOGL"), InlineKeyboardButton("AMZN", callback_data="stockbuy_AMZN"), InlineKeyboardButton("TSLA", callback_data="stockbuy_TSLA")],
            [InlineKeyboardButton("SPY", callback_data="stockbuy_SPY"), InlineKeyboardButton("QQQ", callback_data="stockbuy_QQQ"), InlineKeyboardButton("META", callback_data="stockbuy_META")],
            [InlineKeyboardButton("🔙 Back", callback_data="menu_stocks")]
        ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data.startswith("stockbuy_"):
        symbol = data.replace("stockbuy_", "")
        price = await alpaca_client.get_stock_price(symbol)
        
        keyboard = [
            [InlineKeyboardButton("$25", callback_data=f"confirm_stockbuy_{symbol}_25"), InlineKeyboardButton("$50", callback_data=f"confirm_stockbuy_{symbol}_50")],
            [InlineKeyboardButton("$100", callback_data=f"confirm_stockbuy_{symbol}_100"), InlineKeyboardButton("$250", callback_data=f"confirm_stockbuy_{symbol}_250")],
            [InlineKeyboardButton("🔙 Back", callback_data="stock_buy_menu")]
        ]
        
        await query.edit_message_text(
            f"💰 <b>BUY {symbol}</b>\n\n"
            f"Price: ${price:,.2f}\n\n"
            f"Select amount:",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("confirm_stockbuy_"):
        parts = data.replace("confirm_stockbuy_", "").split("_")
        symbol = parts[0]
        amount = float(parts[1])
        
        market_open = await alpaca_client.is_market_open()
        if not market_open:
            await query.edit_message_text("❌ Market is closed. Try during market hours (9:30 AM - 4:00 PM ET)")
            return
        
        result = await alpaca_client.place_stock_order(symbol, "buy", notional=amount)
        
        if "id" in result:
            price = await alpaca_client.get_stock_price(symbol)
            await query.edit_message_text(
                f"✅ <b>STOCK ORDER PLACED</b>\n\n"
                f"📊 {symbol}\n"
                f"💵 ${amount:.2f}\n"
                f"💲 ~${price:,.2f}/share\n\n"
                f"Order ID: <code>{result['id'][:8]}...</code>",
                parse_mode="HTML"
            )
        else:
            error = result.get("message", result.get("error", "Unknown error"))
            await query.edit_message_text(f"❌ Order failed: {error}")
    
    elif data == "stock_sell_menu":
        if not alpaca_client:
            await query.edit_message_text("❌ Alpaca not configured")
            return
        
        holdings = await alpaca_client.get_all_holdings()
        
        if not holdings["assets"]:
            await query.edit_message_text("📭 No stock positions to sell")
            return
        
        text = "💸 <b>SELL STOCKS</b>\n━━━━━━━━━━━━━━━\n\n"
        keyboard = []
        
        for asset in holdings["assets"][:6]:
            symbol = asset["symbol"]
            value = asset["market_value"]
            pl_pct = asset["unrealized_plpc"]
            emoji = "🟢" if pl_pct >= 0 else "🔴"
            text += f"{emoji} <b>{symbol}</b>: ${value:,.2f} ({pl_pct:+.1f}%)\n"
            keyboard.append([InlineKeyboardButton(f"Sell {symbol}", callback_data=f"stocksell_menu_{symbol}")])
        
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="menu_stocks")])
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data.startswith("stocksell_menu_"):
        symbol = data.replace("stocksell_menu_", "")
        
        holdings = await alpaca_client.get_all_holdings()
        position = next((a for a in holdings["assets"] if a["symbol"] == symbol), None)
        
        if not position:
            await query.edit_message_text(f"❌ No position in {symbol}")
            return
        
        keyboard = [
            [InlineKeyboardButton("25%", callback_data=f"confirm_stocksell_{symbol}_25"), InlineKeyboardButton("50%", callback_data=f"confirm_stocksell_{symbol}_50")],
            [InlineKeyboardButton("75%", callback_data=f"confirm_stocksell_{symbol}_75"), InlineKeyboardButton("100%", callback_data=f"confirm_stocksell_{symbol}_100")],
            [InlineKeyboardButton("🔙 Back", callback_data="stock_sell_menu")]
        ]
        
        await query.edit_message_text(
            f"💸 <b>SELL {symbol}</b>\n\n"
            f"Shares: {position['qty']:.4f}\n"
            f"Value: ${position['market_value']:,.2f}\n"
            f"P&L: {position['unrealized_plpc']:+.1f}%\n\n"
            f"How much to sell?",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("confirm_stocksell_"):
        parts = data.replace("confirm_stocksell_", "").split("_")
        symbol = parts[0]
        sell_pct = float(parts[1])
        
        market_open = await alpaca_client.is_market_open()
        if not market_open:
            await query.edit_message_text("❌ Market is closed")
            return
        
        holdings = await alpaca_client.get_all_holdings()
        position = next((a for a in holdings["assets"] if a["symbol"] == symbol), None)
        
        if not position:
            await query.edit_message_text(f"❌ No position in {symbol}")
            return
        
        sell_qty = position["qty"] * (sell_pct / 100)
        
        result = await alpaca_client.place_stock_order(symbol, "sell", qty=sell_qty)
        
        if "id" in result:
            await query.edit_message_text(
                f"✅ <b>SELL ORDER PLACED</b>\n\n"
                f"📊 {symbol}\n"
                f"📈 {sell_qty:.4f} shares ({sell_pct}%)\n"
                f"💵 ~${position['market_value'] * sell_pct / 100:,.2f}\n\n"
                f"Order ID: <code>{result['id'][:8]}...</code>",
                parse_mode="HTML"
            )
        else:
            error = result.get("message", result.get("error", "Unknown error"))
            await query.edit_message_text(f"❌ Order failed: {error}")
    
    elif data == "all_portfolio":
        await query.edit_message_text("📊 Loading all assets...")
        
        # Crypto
        crypto_value = 0
        crypto_assets = []
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
            crypto_value = holdings.get("crypto_value", 0)
            crypto_assets = holdings.get("assets", [])
        
        # Stocks
        stock_value = 0
        stock_assets = []
        stock_cash = 0
        if alpaca_client:
            holdings = await alpaca_client.get_all_holdings()
            stock_value = holdings.get("portfolio_value", 0) - holdings.get("cash", 0)
            stock_cash = holdings.get("cash", 0)
            stock_assets = holdings.get("assets", [])
        
        total = crypto_value + stock_value + stock_cash
        
        text = f"🌐 <b>MULTI-ASSET PORTFOLIO</b>\n━━━━━━━━━━━━━━━━━━━━━\n\n"
        text += f"💰 <b>Total: ${total:,.2f}</b>\n\n"
        
        if crypto_value > 0:
            pct = (crypto_value / total * 100) if total > 0 else 0
            text += f"🪙 Crypto: ${crypto_value:,.2f} ({pct:.1f}%)\n"
        if stock_value > 0:
            pct = (stock_value / total * 100) if total > 0 else 0
            text += f"📈 Stocks: ${stock_value:,.2f} ({pct:.1f}%)\n"
        if stock_cash > 0:
            pct = (stock_cash / total * 100) if total > 0 else 0
            text += f"💵 Cash: ${stock_cash:,.2f} ({pct:.1f}%)\n"
        
        text += "\n<b>Top Holdings:</b>\n"
        
        all_assets = []
        for a in crypto_assets[:3]:
            all_assets.append({"sym": a["currency"], "val": a["value"], "type": "🪙"})
        for a in stock_assets[:3]:
            all_assets.append({"sym": a["symbol"], "val": a["market_value"], "type": "📈"})
        
        all_assets.sort(key=lambda x: x["val"], reverse=True)
        for a in all_assets[:6]:
            text += f"{a['type']} <b>{a['sym']}</b>: ${a['val']:,.2f}\n"
        
        keyboard = [
            [InlineKeyboardButton("🪙 Crypto", callback_data="menu_portfolio"), InlineKeyboardButton("📈 Stocks", callback_data="stock_portfolio")],
            [InlineKeyboardButton("🔄 Refresh", callback_data="all_portfolio"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
        ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_fear":
        fg = await get_fear_greed_index()
        value = fg["value"]
        emoji = "😱" if value <= 25 else "😰" if value <= 45 else "😐" if value <= 55 else "😊" if value <= 75 else "🤑"
        text = f"😱 <b>FEAR & GREED INDEX</b>\n━━━━━━━━━━━━━━━\n\n"
        text += f"{emoji} Value: <b>{value}</b>\n"
        text += f"📊 Status: {fg['classification']}\n\n"
        text += "0-25: Extreme Fear (Buy)\n26-45: Fear\n46-55: Neutral\n56-75: Greed\n76-100: Extreme Greed (Sell)"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="menu_fear"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_regime":
        regime = await detect_market_regime()
        text = f"🔍 <b>MARKET REGIME</b>\n━━━━━━━━━━━━━━━\n\n"
        text += f"📊 {regime.get('regime_display', 'Unknown')}\n\n"
        text += f"Confidence: {regime.get('confidence', 0)}%\n"
        text += f"Volatility: {regime.get('volatility', 0):.1f}%\n"
        text += f"Trend: {regime.get('trend_strength', 0):.1f}%"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="menu_regime"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_news":
        news = await get_crypto_news()
        text = "📰 <b>CRYPTO NEWS</b>\n━━━━━━━━━━━━━━━\n\n"
        for n in news[:4]:
            text += f"• {n['title'][:60]}...\n\n"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="menu_news"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_risk":
        # Get portfolio value
        total_value = 0
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
            total_value = holdings.get("total_value", 0)
        
        drawdown = risk_state.get("current_drawdown", 0)
        paused = risk_state.get("paused_reason")
        
        text = f"""🛡️ <b>RISK DASHBOARD</b>
━━━━━━━━━━━━━━━━━━━━━

Status: {'🔴 PAUSED - ' + paused if paused else '🟢 ACTIVE'}

💰 Portfolio: ${total_value:,.2f}
📉 Drawdown: {drawdown:.1f}%
💔 Today's P&L: ${daily_pnl['realized']:+,.2f}

━━━━━━━━━━━━━━━━━━━━━
<b>Limits:</b>
• Max Drawdown: {risk_state['max_drawdown_pct']}%
• Daily Loss: ${risk_state['daily_loss_limit']}
• Max Position: {risk_state['max_position_pct']}%
"""
        keyboard = [
            [InlineKeyboardButton("⚙️ Configure", callback_data="menu_risk_config")],
            [InlineKeyboardButton("🔄 Refresh", callback_data="menu_risk"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
        ]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_risk_config":
        keyboard = [
            [InlineKeyboardButton("━━━ DRAWDOWN ━━━", callback_data="none")],
            [
                InlineKeyboardButton("5%", callback_data="risk_dd_5"),
                InlineKeyboardButton("10%", callback_data="risk_dd_10"),
                InlineKeyboardButton("15%", callback_data="risk_dd_15"),
                InlineKeyboardButton("20%", callback_data="risk_dd_20")
            ],
            [InlineKeyboardButton("━━━ DAILY LOSS ━━━", callback_data="none")],
            [
                InlineKeyboardButton("$25", callback_data="risk_dll_25"),
                InlineKeyboardButton("$50", callback_data="risk_dll_50"),
                InlineKeyboardButton("$100", callback_data="risk_dll_100"),
                InlineKeyboardButton("$200", callback_data="risk_dll_200")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="menu_risk")]
        ]
        await query.edit_message_text("🛡️ <b>RISK CONFIG</b>\n\nSelect limits:", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_autopilot":
        status = "🟢 ACTIVE" if autopilot_settings["enabled"] else "🔴 OFF"
        usd_balance = 0
        if cdp_client:
            usd_balance = await cdp_client.get_usd_balance()
        
        # Calculate next trade size
        trade_size = usd_balance * autopilot_settings["trade_percentage"] / 100
        trade_size = max(autopilot_settings["min_trade_usd"], min(autopilot_settings["max_trade_usd"], trade_size))
        
        text = f"""🤖 <b>AUTOPILOT CONTROL CENTER</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Status:</b> {status}
<b>Mode:</b> {'🔴 LIVE TRADING' if LIVE_TRADING else '🟡 PAPER MODE'}

━━━━━━━━━━━━━━━━━━━━━
<b>💰 ACCOUNT</b>
━━━━━━━━━━━━━━━━━━━━━
Balance: <code>${usd_balance:,.2f}</code>
Next Trade: <code>~${trade_size:,.2f}</code>

━━━━━━━━━━━━━━━━━━━━━
<b>📊 SETTINGS</b>
━━━━━━━━━━━━━━━━━━━━━
Trade Size: {autopilot_settings['trade_percentage']}% of balance
Min Trade: ${autopilot_settings['min_trade_usd']}
Max Trade: ${autopilot_settings['max_trade_usd']}
Min Confidence: {autopilot_settings['min_confidence']}%
Min EV: {autopilot_settings.get('min_expected_value', 1.5)}%

━━━━━━━━━━━━━━━━━━━━━
<b>📈 TODAY'S ACTIVITY</b>
━━━━━━━━━━━━━━━━━━━━━
Trades: {autopilot_settings['trades_today']}/{autopilot_settings['max_daily_trades']}
Profit: ${autopilot_settings['total_profit']:+,.2f}
Regime: {autopilot_settings.get('current_regime', 'unknown')}

━━━━━━━━━━━━━━━━━━━━━
<b>⚙️ RISK CONTROLS</b>
━━━━━━━━━━━━━━━━━━━━━
Kelly Sizing: {'✅' if autopilot_settings.get('use_kelly') else '❌'}
Reinvest: {'✅' if autopilot_settings.get('reinvest_profits') else '❌'}
Daily Target: {autopilot_settings.get('daily_target_pct', 3)}%
"""
        
        # Build keyboard based on current state
        toggle_text = "🔴 STOP AUTOPILOT" if autopilot_settings["enabled"] else "🟢 START AUTOPILOT"
        toggle_action = "autopilot_off" if autopilot_settings["enabled"] else "autopilot_on"
        
        keyboard = [
            [InlineKeyboardButton(toggle_text, callback_data=toggle_action)],
            [InlineKeyboardButton("━━━ TRADE SIZE ━━━", callback_data="none")],
            [
                InlineKeyboardButton("10%", callback_data="ap_pct_10"),
                InlineKeyboardButton("15%", callback_data="ap_pct_15"),
                InlineKeyboardButton("20%", callback_data="ap_pct_20"),
                InlineKeyboardButton("25%", callback_data="ap_pct_25")
            ],
            [InlineKeyboardButton("━━━ MIN CONFIDENCE ━━━", callback_data="none")],
            [
                InlineKeyboardButton("65%", callback_data="ap_conf_65"),
                InlineKeyboardButton("70%", callback_data="ap_conf_70"),
                InlineKeyboardButton("75%", callback_data="ap_conf_75"),
                InlineKeyboardButton("80%", callback_data="ap_conf_80")
            ],
            [InlineKeyboardButton("━━━ MAX TRADES/DAY ━━━", callback_data="none")],
            [
                InlineKeyboardButton("5", callback_data="ap_max_5"),
                InlineKeyboardButton("10", callback_data="ap_max_10"),
                InlineKeyboardButton("15", callback_data="ap_max_15"),
                InlineKeyboardButton("20", callback_data="ap_max_20")
            ],
            [InlineKeyboardButton("━━━ MAX TRADE $ ━━━", callback_data="none")],
            [
                InlineKeyboardButton("$50", callback_data="ap_maxusd_50"),
                InlineKeyboardButton("$100", callback_data="ap_maxusd_100"),
                InlineKeyboardButton("$250", callback_data="ap_maxusd_250"),
                InlineKeyboardButton("$500", callback_data="ap_maxusd_500")
            ],
            [InlineKeyboardButton("━━━ TOGGLES ━━━", callback_data="none")],
            [
                InlineKeyboardButton(f"Kelly: {'✅' if autopilot_settings.get('use_kelly') else '❌'}", callback_data="ap_toggle_kelly"),
                InlineKeyboardButton(f"Reinvest: {'✅' if autopilot_settings.get('reinvest_profits') else '❌'}", callback_data="ap_toggle_reinvest")
            ],
            [InlineKeyboardButton("🔄 Refresh", callback_data="menu_autopilot"), InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
        ]
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    # Autopilot setting handlers
    elif data.startswith("ap_pct_"):
        pct = int(data.replace("ap_pct_", ""))
        autopilot_settings["trade_percentage"] = pct
        save_data("autopilot_settings", autopilot_settings)
        await query.answer(f"✅ Trade size set to {pct}%")
        # Refresh the menu
        await button_callback(update, context)
    
    elif data.startswith("ap_conf_"):
        conf = int(data.replace("ap_conf_", ""))
        autopilot_settings["min_confidence"] = conf
        save_data("autopilot_settings", autopilot_settings)
        await query.answer(f"✅ Min confidence set to {conf}%")
        # Trigger refresh by setting data and calling again
        query.data = "menu_autopilot"
        await button_callback(update, context)
    
    elif data.startswith("ap_max_"):
        max_trades = int(data.replace("ap_max_", ""))
        autopilot_settings["max_daily_trades"] = max_trades
        save_data("autopilot_settings", autopilot_settings)
        await query.answer(f"✅ Max trades set to {max_trades}/day")
        query.data = "menu_autopilot"
        await button_callback(update, context)
    
    elif data.startswith("ap_maxusd_"):
        max_usd = int(data.replace("ap_maxusd_", ""))
        autopilot_settings["max_trade_usd"] = max_usd
        save_data("autopilot_settings", autopilot_settings)
        await query.answer(f"✅ Max trade set to ${max_usd}")
        query.data = "menu_autopilot"
        await button_callback(update, context)
    
    elif data == "ap_toggle_kelly":
        autopilot_settings["use_kelly"] = not autopilot_settings.get("use_kelly", True)
        save_data("autopilot_settings", autopilot_settings)
        status = "ON" if autopilot_settings["use_kelly"] else "OFF"
        await query.answer(f"✅ Kelly sizing {status}")
        query.data = "menu_autopilot"
        await button_callback(update, context)
    
    elif data == "ap_toggle_reinvest":
        autopilot_settings["reinvest_profits"] = not autopilot_settings.get("reinvest_profits", True)
        save_data("autopilot_settings", autopilot_settings)
        status = "ON" if autopilot_settings["reinvest_profits"] else "OFF"
        await query.answer(f"✅ Reinvest profits {status}")
        query.data = "menu_autopilot"
        await button_callback(update, context)
    
    elif data == "none":
        await query.answer()  # Do nothing for separator buttons
    
    elif data == "menu_performance":
        total_trades = len(trade_history)
        total_pnl = sum(t.get("pnl_usd", 0) for t in trade_history)
        wins = sum(1 for t in trade_history if t.get("pnl_usd", 0) > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        text = f"📈 <b>PERFORMANCE</b>\n━━━━━━━━━━━━━━━\n\n"
        text += f"Total Trades: {total_trades}\n"
        text += f"Win Rate: {win_rate:.1f}%\n"
        text += f"Total P&L: ${total_pnl:+,.2f}"
        keyboard = [[InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_history":
        text = "📜 <b>RECENT TRADES</b>\n━━━━━━━━━━━━━━━\n\n"
        for trade in trade_history[-5:][::-1]:
            emoji = "🟢" if trade.get("pnl_usd", 0) >= 0 else "🔴"
            text += f"{emoji} {trade.get('pair')}: ${trade.get('pnl_usd', 0):+.2f}\n"
        if not trade_history:
            text += "No trades yet"
        keyboard = [[InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_settings":
        text = f"⚙️ <b>SETTINGS</b>\n━━━━━━━━━━━━━━━\n\n"
        text += f"Trade: ${TRADE_AMOUNT_USD}\n"
        text += f"Stop Loss: {STOP_LOSS_PCT}%\n"
        text += f"Take Profit: {TAKE_PROFIT_PCT}%\n"
        text += f"Mode: {'🔴 LIVE' if LIVE_TRADING else '🟡 PAPER'}"
        keyboard = [[InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "menu_back":
        keyboard = [
            [InlineKeyboardButton("🪙 Crypto", callback_data="menu_portfolio"), InlineKeyboardButton("📈 Stocks", callback_data="menu_stocks")],
            [InlineKeyboardButton("💱 Forex", callback_data="menu_forex"), InlineKeyboardButton("🌐 All Assets", callback_data="all_portfolio")],
            [InlineKeyboardButton("📊 Signals", callback_data="menu_signals"), InlineKeyboardButton("📈 Market", callback_data="menu_market")],
            [InlineKeyboardButton("🤖 Autopilot", callback_data="menu_autopilot"), InlineKeyboardButton("🛡️ Risk", callback_data="menu_risk")],
            [InlineKeyboardButton("😱 Fear/Greed", callback_data="menu_fear"), InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings")]
        ]
        await query.edit_message_text("🚀 <b>QUANTSIGNALS ULTRA</b>\n<i>Crypto • Stocks • Forex</i>\n\nSelect an option:", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data.startswith("quickbuy_"):
        coin = data.replace("quickbuy_", "")
        pair = f"{coin}-USD"
        price = await get_public_price(pair)
        keyboard = [
            [InlineKeyboardButton(f"Buy $10", callback_data=f"confirm_buy_{pair}_10")],
            [InlineKeyboardButton(f"Buy $25", callback_data=f"confirm_buy_{pair}_25")],
            [InlineKeyboardButton(f"Buy $50", callback_data=f"confirm_buy_{pair}_50")],
            [InlineKeyboardButton(f"Buy $100", callback_data=f"confirm_buy_{pair}_100")],
            [InlineKeyboardButton("🔙 Back", callback_data="menu_buy")]
        ]
        await query.edit_message_text(
            f"💰 <b>BUY {coin}</b>\n\n"
            f"Price: ${price:,.2f}\n\n"
            f"Select amount:",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "autopilot_on":
        autopilot_settings["enabled"] = True
        autopilot_settings["trades_today"] = 0
        save_data("autopilot_settings", autopilot_settings)
        save_positions()  # Also save to ensure persistence
        
        usd_balance = 0
        if cdp_client:
            usd_balance = await cdp_client.get_usd_balance()
        
        await query.answer("🟢 Autopilot ENABLED!")
        
        await query.edit_message_text(
            f"🟢 <b>AUTOPILOT ACTIVATED!</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Balance: ${usd_balance:,.2f}\n"
            f"📊 Trade Size: {autopilot_settings['trade_percentage']}%\n"
            f"🎯 Min Confidence: {autopilot_settings['min_confidence']}%\n"
            f"📈 Max Trades: {autopilot_settings['max_daily_trades']}/day\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Mode: {'🔴 LIVE TRADING' if LIVE_TRADING else '🟡 PAPER MODE'}\n\n"
            f"<i>Bot is now scanning for opportunities...</i>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Autopilot Settings", callback_data="menu_autopilot")],
                [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
            ])
        )
    
    elif data == "autopilot_off":
        autopilot_settings["enabled"] = False
        save_data("autopilot_settings", autopilot_settings)
        save_positions()
        
        await query.answer("🔴 Autopilot DISABLED!")
        
        await query.edit_message_text(
            f"🔴 <b>AUTOPILOT STOPPED</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Trades Today: {autopilot_settings['trades_today']}\n"
            f"💰 Session Profit: ${autopilot_settings['total_profit']:+,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Switched to manual mode.\n"
            f"Open positions still have stop-loss active.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🟢 Turn Back ON", callback_data="autopilot_on")],
                [InlineKeyboardButton("🔙 Menu", callback_data="menu_back")]
            ])
        )
    
    elif data == "refresh_portfolio":
        # Refresh portfolio - call portfolio logic
        await query.edit_message_text("📊 Refreshing portfolio...")
        
        holdings = {"usd_balance": 0, "crypto_value": 0, "total_value": 0, "assets": []}
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
        
        total_value = holdings["total_value"]
        usd_balance = holdings["usd_balance"]
        crypto_value = holdings["crypto_value"]
        
        usd_pct = (usd_balance / total_value * 100) if total_value > 0 else 0
        crypto_pct = (crypto_value / total_value * 100) if total_value > 0 else 0
        
        text = f"""
╔═══════════════════════════════════════╗
║      💼 <b>PORTFOLIO OVERVIEW</b>            ║
╠═══════════════════════════════════════╣
║  💰 <b>Total:</b>  <code>${total_value:,.2f}</code>               ║
╠═══════════════════════════════════════╣
║  💵 USD:    <code>${usd_balance:,.2f}</code>  ({usd_pct:.1f}%)        ║
║  🪙 Crypto: <code>${crypto_value:,.2f}</code>  ({crypto_pct:.1f}%)        ║
╚═══════════════════════════════════════╝
"""
        
        if holdings["assets"]:
            for asset in holdings["assets"][:8]:
                currency = asset["currency"]
                value = asset["value"]
                pct = (value / total_value * 100) if total_value > 0 else 0
                bar_filled = int(pct / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                
                pair = f"{currency}-USD"
                pnl_str = ""
                if pair in positions:
                    entry = positions[pair]["entry_price"]
                    pnl_pct = ((asset["price"] - entry) / entry) * 100
                    emoji = "🟢" if pnl_pct >= 0 else "🔴"
                    pnl_str = f" {emoji}{pnl_pct:+.1f}%"
                
                text += f"\n<b>{currency}</b>{pnl_str}: ${value:,.2f} ({pct:.1f}%)\n[{bar}]\n"
        
        text += f"\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_portfolio")],
            [InlineKeyboardButton("📊 Signals", callback_data="get_signals")]
        ]
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "get_signals":
        await query.edit_message_text("🔄 Generating signals...")
        
        signals = await generate_trading_signals(include_news=False)
        
        text = "📊 <b>QUICK SIGNALS</b>\n━━━━━━━━━━━━━━━\n\n"
        
        if signals.get("signals"):
            for sig in signals["signals"][:3]:
                direction = sig.get("direction", sig.get("action", "HOLD"))
                emoji = "🟢" if direction == "BUY" else "🔴"
                text += f"{emoji} <b>{sig.get('pair')}</b>: {direction} ({sig.get('confidence')}%)\n"
                text += f"   Entry: ${sig.get('entry_price', 0):,.2f}\n\n"
        else:
            text += "⚪ No signals right now\n"
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="get_signals")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data.startswith("trade_buy_"):
        pair = data.replace("trade_buy_", "")
        price = await get_public_price(pair)
        
        if LIVE_TRADING and cdp_client:
            await query.edit_message_text(f"🔄 Executing LIVE BUY {pair}...")
            result = await cdp_client.place_market_order(pair, "BUY", TRADE_AMOUNT_USD)
            
            if result.get("success_response") or result.get("order_id"):
                positions[pair] = {
                    "entry_price": price,
                    "highest_price": price,  # For trailing stop
                    "amount_usd": TRADE_AMOUNT_USD,
                    "timestamp": datetime.now().isoformat(),
                    "live": True
                }
                save_positions()
                
                await query.edit_message_text(
                    f"✅ <b>LIVE ORDER EXECUTED</b>\n\n"
                    f"📊 {pair}: ${price:,.2f}\n"
                    f"💵 Amount: ${TRADE_AMOUNT_USD}\n"
                    f"🛑 SL: ${price*(1-STOP_LOSS_PCT/100):,.2f}\n"
                    f"🎯 TP: ${price*(1+TAKE_PROFIT_PCT/100):,.2f}\n\n"
                    f"📈 Trailing stop active at {TRAILING_STOP_PCT}%",
                    parse_mode="HTML"
                )
            else:
                error = result.get("error_response", {}).get("message", result.get("error", "Unknown"))
                await query.edit_message_text(f"❌ Order failed: {error}")
        else:
            positions[pair] = {
                "entry_price": price,
                "highest_price": price,
                "amount_usd": TRADE_AMOUNT_USD,
                "timestamp": datetime.now().isoformat(),
                "live": False
            }
            save_positions()
            await query.edit_message_text(
                f"✅ <b>PAPER POSITION</b>\n\n{pair}: ${price:,.2f}\n\n"
                f"Set LIVE_TRADING=true for real trades",
                parse_mode="HTML"
            )
    
    elif data.startswith("close_"):
        pair = data.replace("close_", "")
        if pair in positions:
            entry = positions[pair]["entry_price"]
            current = await get_public_price(pair)
            pnl_pct = ((current - entry) / entry) * 100
            pnl_usd = (pnl_pct / 100) * positions[pair]["amount_usd"]
            
            daily_pnl["realized"] += pnl_usd
            daily_pnl["trades"] += 1
            if pnl_usd > 0:
                daily_pnl["wins"] += 1
            
            # Track strategy performance
            strategy = positions[pair].get("strategy", "manual")
            if strategy in strategy_performance:
                strategy_performance[strategy]["trades"] += 1
                strategy_performance[strategy]["pnl"] += pnl_usd
                if pnl_usd > 0:
                    strategy_performance[strategy]["wins"] += 1
                update_strategy_weights()
            
            # Add to trade history
            trade_history.append({
                "pair": pair,
                "entry": entry,
                "exit": current,
                "pnl_pct": round(pnl_pct, 2),
                "pnl_usd": round(pnl_usd, 2),
                "amount_usd": positions[pair].get("amount_usd", 0),
                "strategy": strategy,
                "opened_at": positions[pair].get("timestamp"),
                "closed_at": datetime.now().isoformat(),
                "live": positions[pair].get("live", False)
            })
            
            del positions[pair]
            save_positions()
            
            emoji = "🟢" if pnl_usd >= 0 else "🔴"
            await query.edit_message_text(
                f"✅ <b>CLOSED</b>\n\n{pair}\n"
                f"Entry: ${entry:,.2f} → Exit: ${current:,.2f}\n"
                f"{emoji} P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})",
                parse_mode="HTML"
            )
    
    # New handlers for buy/sell/limit
    elif data.startswith("confirm_buy_"):
        parts = data.replace("confirm_buy_", "").split("_")
        pair = parts[0] + "-" + parts[1]  # BTC-USD
        amount = float(parts[2]) if len(parts) > 2 else TRADE_AMOUNT_USD
        
        price = await get_public_price(pair)
        
        if LIVE_TRADING and cdp_client:
            await query.edit_message_text(f"🔄 Executing BUY {pair}...")
            result = await cdp_client.place_market_order(pair, "BUY", amount)
            
            if result.get("success_response") or result.get("order_id"):
                positions[pair] = {
                    "entry_price": price,
                    "highest_price": price,
                    "amount_usd": amount,
                    "timestamp": datetime.now().isoformat(),
                    "live": True,
                    "strategy": "manual"
                }
                save_positions()
                
                await query.edit_message_text(
                    f"✅ <b>BUY EXECUTED</b>\n\n"
                    f"📊 {pair}\n"
                    f"💵 ${amount:.2f} @ ${price:,.2f}\n"
                    f"🪙 Got ~{amount/price:.6f} {pair.split('-')[0]}\n\n"
                    f"🛑 SL: ${price*(1-STOP_LOSS_PCT/100):,.2f}\n"
                    f"🎯 TP: ${price*(1+TAKE_PROFIT_PCT/100):,.2f}",
                    parse_mode="HTML"
                )
            else:
                error = result.get("error", "Unknown error")
                await query.edit_message_text(f"❌ Buy failed: {error}")
        else:
            positions[pair] = {
                "entry_price": price,
                "highest_price": price,
                "amount_usd": amount,
                "timestamp": datetime.now().isoformat(),
                "live": False,
                "strategy": "manual"
            }
            save_positions()
            await query.edit_message_text(
                f"✅ <b>PAPER BUY</b>\n\n"
                f"📊 {pair} @ ${price:,.2f}\n"
                f"💵 Amount: ${amount:.2f}",
                parse_mode="HTML"
            )
    
    elif data.startswith("confirm_sell_"):
        parts = data.replace("confirm_sell_", "").split("_")
        pair = parts[0] + "-" + parts[1]
        sell_pct = float(parts[2]) if len(parts) > 2 else 100
        
        if not cdp_client:
            await query.edit_message_text("❌ Coinbase not connected")
            return
        
        # Get holdings to find amount
        holdings = await cdp_client.get_all_holdings()
        coin = pair.split("-")[0]
        asset = next((a for a in holdings["assets"] if a["currency"] == coin), None)
        
        if not asset:
            await query.edit_message_text(f"❌ No {coin} to sell")
            return
        
        sell_value = asset["value"] * (sell_pct / 100)
        
        if LIVE_TRADING:
            await query.edit_message_text(f"🔄 Executing SELL {pair}...")
            result = await cdp_client.place_market_order(pair, "SELL", sell_value)
            
            if result.get("success_response") or result.get("order_id"):
                # Track P&L if we had a position
                pnl_str = ""
                if pair in positions:
                    entry = positions[pair]["entry_price"]
                    pnl_pct = ((asset["price"] - entry) / entry) * 100
                    pnl_usd = sell_value * (pnl_pct / 100)
                    
                    trade_history.append({
                        "pair": pair,
                        "entry": entry,
                        "exit": asset["price"],
                        "pnl_pct": round(pnl_pct, 2),
                        "pnl_usd": round(pnl_usd, 2),
                        "closed_at": datetime.now().isoformat(),
                        "live": True
                    })
                    
                    daily_pnl["realized"] += pnl_usd
                    daily_pnl["trades"] += 1
                    if pnl_usd > 0:
                        daily_pnl["wins"] += 1
                    
                    if sell_pct >= 100:
                        del positions[pair]
                    
                    emoji = "🟢" if pnl_usd >= 0 else "🔴"
                    pnl_str = f"\n{emoji} P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})"
                
                save_positions()
                
                await query.edit_message_text(
                    f"✅ <b>SELL EXECUTED</b>\n\n"
                    f"📊 {pair}\n"
                    f"💵 Sold {sell_pct}% (~${sell_value:.2f})"
                    f"{pnl_str}",
                    parse_mode="HTML"
                )
            else:
                await query.edit_message_text(f"❌ Sell failed")
        else:
            await query.edit_message_text("⚠️ Enable LIVE_TRADING for real sells")
    
    elif data.startswith("sell_menu_"):
        coin = data.replace("sell_menu_", "")
        pair = f"{coin}-USD"
        
        holdings = await cdp_client.get_all_holdings()
        asset = next((a for a in holdings["assets"] if a["currency"] == coin), None)
        
        if not asset:
            await query.edit_message_text(f"❌ No {coin} found")
            return
        
        keyboard = [
            [InlineKeyboardButton("Sell 25%", callback_data=f"confirm_sell_{pair}_25")],
            [InlineKeyboardButton("Sell 50%", callback_data=f"confirm_sell_{pair}_50")],
            [InlineKeyboardButton("Sell 100%", callback_data=f"confirm_sell_{pair}_100")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_order")]
        ]
        
        await query.edit_message_text(
            f"💸 <b>SELL {coin}</b>\n\n"
            f"Balance: {asset['balance']:.6f} {coin}\n"
            f"Value: ${asset['value']:,.2f}\n"
            f"Price: ${asset['price']:,.2f}\n\n"
            f"How much to sell?",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "cancel_order":
        await query.edit_message_text("❌ Order cancelled")
    
    elif data == "clear_all_alerts":
        price_alerts.clear()
        save_positions()
        await query.edit_message_text("✅ All alerts cleared")
    
    elif data == "export_history":
        if not trade_history:
            await query.edit_message_text("📭 No trade history to export")
            return
        
        # Create CSV
        csv_lines = ["Date,Pair,Entry,Exit,P&L %,P&L $,Strategy"]
        for trade in trade_history:
            date = trade.get("closed_at", "")[:10]
            csv_lines.append(
                f"{date},{trade.get('pair')},{trade.get('entry')},{trade.get('exit')},"
                f"{trade.get('pnl_pct')},{trade.get('pnl_usd')},{trade.get('strategy')}"
            )
        
        csv_text = "\n".join(csv_lines)
        
        await query.edit_message_text(
            f"📤 <b>TRADE HISTORY EXPORT</b>\n\n"
            f"<code>{csv_text[:2000]}</code>\n\n"
            f"<i>Copy the above CSV data</i>",
            parse_mode="HTML"
        )
    
    # ============ RISK MANAGEMENT CALLBACKS ============
    elif data.startswith("risk_dd_"):
        pct = int(data.replace("risk_dd_", ""))
        risk_state["max_drawdown_pct"] = pct
        save_positions()
        await query.answer(f"✅ Max drawdown set to {pct}%")
    
    elif data.startswith("risk_dll_"):
        limit = int(data.replace("risk_dll_", ""))
        risk_state["daily_loss_limit"] = limit
        save_positions()
        await query.answer(f"✅ Daily loss limit set to ${limit}")
    
    elif data.startswith("risk_mp_"):
        pct = int(data.replace("risk_mp_", ""))
        risk_state["max_position_pct"] = pct
        save_positions()
        await query.answer(f"✅ Max position set to {pct}%")
    
    elif data == "risk_reset_peak":
        if cdp_client:
            holdings = await cdp_client.get_all_holdings()
            risk_state["highest_balance"] = holdings.get("total_value", 0)
        save_positions()
        await query.answer("✅ Peak balance reset")
    
    # ============ WATCHLIST CALLBACKS ============
    elif data.startswith("wl_remove_"):
        coin = data.replace("wl_remove_", "")
        pair = f"{coin}-USD"
        if pair in watchlist:
            del watchlist[pair]
            save_positions()
        await query.answer(f"✅ Removed {coin} from watchlist")
        await query.edit_message_text(f"✅ Removed {coin} from watchlist", parse_mode="HTML")
    
    # ============ TP TIERS CALLBACKS ============
    elif data == "tp_toggle":
        tp_tiers["enabled"] = not tp_tiers["enabled"]
        save_positions()
        status = "ENABLED" if tp_tiers["enabled"] else "DISABLED"
        await query.answer(f"✅ TP Tiers {status}")
    
    elif data == "tp_preset_conservative":
        tp_tiers["tiers"] = [
            {"pct_gain": 3, "sell_pct": 33},
            {"pct_gain": 6, "sell_pct": 33},
            {"pct_gain": 10, "sell_pct": 34},
        ]
        save_positions()
        await query.answer("✅ Conservative preset applied")
    
    elif data == "tp_preset_balanced":
        tp_tiers["tiers"] = [
            {"pct_gain": 5, "sell_pct": 25},
            {"pct_gain": 10, "sell_pct": 25},
            {"pct_gain": 20, "sell_pct": 25},
        ]
        save_positions()
        await query.answer("✅ Balanced preset applied")
    
    elif data == "tp_preset_aggressive":
        tp_tiers["tiers"] = [
            {"pct_gain": 10, "sell_pct": 20},
            {"pct_gain": 25, "sell_pct": 30},
            {"pct_gain": 50, "sell_pct": 25},
        ]
        save_positions()
        await query.answer("✅ Aggressive preset applied")
    
    # ============ DCA CALLBACKS ============
    elif data == "dca_toggle":
        dca_settings["enabled"] = not dca_settings["enabled"]
        save_positions()
        status = "ENABLED" if dca_settings["enabled"] else "DISABLED"
        await query.answer(f"✅ DCA Autopilot {status}")
    
    elif data.startswith("dca_drop_"):
        pct = int(data.replace("dca_drop_", ""))
        dca_settings["drop_pct"] = pct
        save_positions()
        await query.answer(f"✅ DCA trigger set to {pct}% drop")
    
    elif data.startswith("dca_amt_"):
        amt = int(data.replace("dca_amt_", ""))
        dca_settings["amount_usd"] = amt
        save_positions()
        await query.answer(f"✅ DCA amount set to ${amt}")
    
    elif data.startswith("dca_max_"):
        max_buys = int(data.replace("dca_max_", ""))
        dca_settings["max_per_day"] = max_buys
        save_positions()
        await query.answer(f"✅ Max DCA buys set to {max_buys}/day")


# ============ REGISTER HANDLERS ============
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(CommandHandler("menu", menu_cmd))
tg_app.add_handler(CommandHandler("signals", signals_cmd))
tg_app.add_handler(CommandHandler("market", market_cmd))
tg_app.add_handler(CommandHandler("portfolio", portfolio_cmd))
tg_app.add_handler(CommandHandler("pnl", pnl_cmd))
tg_app.add_handler(CommandHandler("settings", settings_cmd))
tg_app.add_handler(CommandHandler("fear", fear_cmd))
tg_app.add_handler(CommandHandler("news", news_cmd))
tg_app.add_handler(CommandHandler("whale", whale_cmd))
tg_app.add_handler(CommandHandler("timeframe", timeframe_cmd))
tg_app.add_handler(CommandHandler("tf", timeframe_cmd))
tg_app.add_handler(CommandHandler("backtest", backtest_cmd))
tg_app.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
tg_app.add_handler(CommandHandler("alert", alert_cmd))
tg_app.add_handler(CommandHandler("alerts", alerts_cmd))
tg_app.add_handler(CommandHandler("dca", dca_cmd))
tg_app.add_handler(CommandHandler("autopilot", autopilot_cmd))
tg_app.add_handler(CommandHandler("pause", pause_cmd))
tg_app.add_handler(CommandHandler("regime", regime_cmd))
tg_app.add_handler(CommandHandler("strategies", strategies_cmd))
tg_app.add_handler(CommandHandler("stats", stats_cmd))
# Trading commands
tg_app.add_handler(CommandHandler("buy", buy_cmd))
tg_app.add_handler(CommandHandler("sell", sell_cmd))
tg_app.add_handler(CommandHandler("limit", limit_cmd))
tg_app.add_handler(CommandHandler("history", history_cmd))
tg_app.add_handler(CommandHandler("performance", performance_cmd))
# New feature commands
tg_app.add_handler(CommandHandler("risk", risk_cmd))
tg_app.add_handler(CommandHandler("watchlist", watchlist_cmd))
tg_app.add_handler(CommandHandler("watch", watchlist_cmd))
tg_app.add_handler(CommandHandler("tptiers", tptiers_cmd))
tg_app.add_handler(CommandHandler("dcaauto", dcaauto_cmd))
tg_app.add_handler(CommandHandler("summary", summary_cmd))
tg_app.add_handler(CommandHandler("chart", chart_cmd))
tg_app.add_handler(CommandHandler("streak", streak_cmd))
tg_app.add_handler(CommandHandler("besttrades", besttrades_cmd))
# Stocks & Forex commands
tg_app.add_handler(CommandHandler("stocks", stocks_cmd))
tg_app.add_handler(CommandHandler("stockbuy", stockbuy_cmd))
tg_app.add_handler(CommandHandler("stocksell", stocksell_cmd))
tg_app.add_handler(CommandHandler("forex", forex_cmd))
tg_app.add_handler(CommandHandler("allportfolio", allportfolio_cmd))
tg_app.add_handler(CommandHandler("all", allportfolio_cmd))
tg_app.add_handler(CommandHandler("backtest", backtest_cmd))
tg_app.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
tg_app.add_handler(CommandHandler("alert", alert_cmd))
tg_app.add_handler(CommandHandler("alerts", alerts_cmd))
tg_app.add_handler(CommandHandler("dca", dca_cmd))
tg_app.add_handler(CommandHandler("autopilot", autopilot_cmd))
tg_app.add_handler(CommandHandler("pause", pause_cmd))
tg_app.add_handler(CommandHandler("regime", regime_cmd))
tg_app.add_handler(CommandHandler("strategies", strategies_cmd))
tg_app.add_handler(CommandHandler("stats", stats_cmd))
tg_app.add_handler(CallbackQueryHandler(button_callback))


# ============ BACKGROUND TASKS ============
async def stop_loss_monitor():
    """Monitor positions for stop loss, take profit, and trailing stop."""
    while True:
        try:
            await asyncio.sleep(60)
            
            for pair, pos in list(positions.items()):
                current = await get_public_price(pair)
                entry = pos["entry_price"]
                highest = pos.get("highest_price", entry)
                
                # Update highest price for trailing stop
                if current > highest:
                    positions[pair]["highest_price"] = current
                    highest = current
                    save_positions()
                
                pnl_pct = ((current - entry) / entry) * 100
                trailing_drop = ((current - highest) / highest) * 100
                
                should_close = False
                close_reason = ""
                
                # Check stop loss
                if pnl_pct <= -STOP_LOSS_PCT:
                    should_close = True
                    close_reason = f"🛑 STOP LOSS (-{STOP_LOSS_PCT}%)"
                
                # Check take profit
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    should_close = True
                    close_reason = f"🎯 TAKE PROFIT (+{TAKE_PROFIT_PCT}%)"
                
                # Feature 1: Trailing stop
                elif pnl_pct > 0 and trailing_drop <= -TRAILING_STOP_PCT:
                    should_close = True
                    close_reason = f"📈 TRAILING STOP ({TRAILING_STOP_PCT}% from peak)"
                
                if should_close:
                    pnl_usd = (pnl_pct / 100) * pos["amount_usd"]
                    daily_pnl["realized"] += pnl_usd
                    daily_pnl["trades"] += 1
                    if pnl_usd > 0:
                        daily_pnl["wins"] += 1
                    
                    # Track autopilot profits for reinvestment
                    if pos.get("autopilot"):
                        autopilot_settings["total_profit"] += pnl_usd
                        save_data("autopilot", autopilot_settings)
                    
                    del positions[pair]
                    save_positions()
                    
                    emoji = "🟢" if pnl_usd > 0 else "🔴"
                    auto_tag = " 🤖" if pos.get("autopilot") else ""
                    for chat_id in AUTO_SIGNAL_CHATS:
                        try:
                            await tg_app.bot.send_message(
                                chat_id=chat_id,
                                text=f"{close_reason}{auto_tag}\n\n"
                                     f"📊 {pair}\n"
                                     f"Entry: ${entry:,.2f}\n"
                                     f"Exit: ${current:,.2f}\n"
                                     f"Peak: ${highest:,.2f}\n"
                                     f"{emoji} P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n\n"
                                     f"💰 Total autopilot profit: ${autopilot_settings['total_profit']:+,.2f}",
                                parse_mode="HTML"
                            )
                        except:
                            pass
        except Exception as e:
            print(f"[MONITOR ERROR] {e}")


async def price_alert_checker():
    """Check price alerts."""
    while True:
        try:
            await asyncio.sleep(60)
            
            for pair, targets in list(price_alerts.items()):
                current = await get_public_price(pair)
                
                for target in targets[:]:
                    if (target > current * 0.99 and target < current * 1.01):
                        # Price hit target (within 1%)
                        for chat_id in AUTO_SIGNAL_CHATS:
                            try:
                                await tg_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"🔔 <b>PRICE ALERT</b>\n\n"
                                         f"{pair} reached ${target:,.2f}!\n"
                                         f"Current: ${current:,.2f}",
                                    parse_mode="HTML"
                                )
                            except:
                                pass
                        targets.remove(target)
                        save_positions()
        except Exception as e:
            print(f"[ALERT ERROR] {e}")


async def auto_signal_scheduler():
    """Send scheduled signals."""
    while True:
        try:
            tz = pytz.timezone("America/Chicago")
            now = datetime.now(tz)
            
            # Reset daily trade count at midnight
            if autopilot_settings["last_reset"] != now.date().isoformat():
                autopilot_settings["trades_today"] = 0
                autopilot_settings["last_reset"] = now.date().isoformat()
                save_data("autopilot", autopilot_settings)
            
            if now.hour in SIGNAL_HOURS and now.minute < 5:
                signals = await generate_trading_signals()
                
                text = f"📊 <b>AUTO SIGNAL</b> | {now.strftime('%I:%M %p')}\n━━━━━━━━━━━━━━━\n\n"
                
                fg = signals.get("fear_greed", {})
                text += f"😱 Fear/Greed: {fg.get('value', 'N/A')}\n\n"
                
                if signals.get("signals"):
                    for sig in signals["signals"]:
                        emoji = "🟢" if sig["action"] == "BUY" else "🔴"
                        text += f"{emoji} {sig['pair']}: {sig['action']} ({sig['confidence']}%)\n"
                else:
                    text += "⚪ No signals\n"
                
                for chat_id in AUTO_SIGNAL_CHATS:
                    try:
                        await tg_app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
                    except:
                        pass
                
                await asyncio.sleep(600)
            else:
                await asyncio.sleep(60)
        except Exception as e:
            print(f"[SCHEDULER ERROR] {e}")


async def autopilot_scanner():
    """Autopilot mode - auto-execute trades with Ultra intelligence."""
    while True:
        try:
            scan_interval = autopilot_settings.get("scan_interval", 60)
            await asyncio.sleep(scan_interval)
            
            if not autopilot_settings["enabled"]:
                continue
            
            if autopilot_settings.get("paused"):
                continue
            
            # Check risk limits
            can_trade, reason = await check_risk_limits()
            if not can_trade:
                print(f"[AUTOPILOT] Blocked: {reason}")
                continue
            
            # Get current USD balance
            usd_balance = TRADE_AMOUNT_USD
            if cdp_client and LIVE_TRADING:
                usd_balance = await cdp_client.get_usd_balance()
            
            if usd_balance < autopilot_settings["min_trade_usd"]:
                continue
            
            # Generate signals
            signals = await generate_trading_signals(include_news=False)
            
            if not signals.get("signals"):
                continue
            
            for signal in signals["signals"]:
                # Get direction (support both old and new format)
                direction = signal.get("direction", signal.get("action", "")).upper()
                if direction != "BUY":
                    continue
                
                confidence = signal.get("confidence", 0)
                expected_value = signal.get("expected_value", 0)
                strategy = signal.get("strategy", "unknown")
                
                # Check confidence threshold
                if confidence < autopilot_settings["min_confidence"]:
                    continue
                
                # Check expected value threshold
                min_ev = autopilot_settings.get("min_expected_value", MIN_EXPECTED_VALUE)
                if expected_value < min_ev:
                    continue
                
                # Check strategy weight
                strat_weight = strategy_performance.get(strategy, {}).get("weight", 1.0)
                if strat_weight < 0.3:
                    print(f"[AUTOPILOT] Skipping {strategy} - low weight ({strat_weight})")
                    continue
                
                pair = signal.get("pair")
                if pair in positions:
                    continue
                
                # Calculate position size
                if autopilot_settings.get("use_kelly") and strategy in strategy_performance:
                    stats = strategy_performance[strategy]
                    win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 3 else 55
                    avg_win = TAKE_PROFIT_PCT
                    avg_loss = STOP_LOSS_PCT
                    trade_amount = calculate_kelly_position(win_rate, avg_win, avg_loss, usd_balance, confidence)
                else:
                    trade_amount = usd_balance * autopilot_settings["trade_percentage"] / 100
                
                # Apply limits
                trade_amount = max(autopilot_settings["min_trade_usd"],
                                 min(autopilot_settings["max_trade_usd"], trade_amount))
                
                # Verify balance
                if cdp_client and LIVE_TRADING:
                    usd_balance = await cdp_client.get_usd_balance()
                    if usd_balance < trade_amount:
                        continue
                
                # Execute trade
                price = await get_public_price(pair)
                
                if LIVE_TRADING and cdp_client:
                    result = await cdp_client.place_market_order(pair, "BUY", trade_amount)
                    
                    if result.get("success_response") or result.get("order_id"):
                        positions[pair] = {
                            "entry_price": price,
                            "highest_price": price,
                            "amount_usd": trade_amount,
                            "timestamp": datetime.now().isoformat(),
                            "live": True,
                            "autopilot": True
                        }
                        autopilot_settings["trades_today"] += 1
                        save_positions()
                        save_data("autopilot", autopilot_settings)
                        
                        # Alert
                        for chat_id in AUTO_SIGNAL_CHATS:
                            try:
                                await tg_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"🤖 <b>AUTOPILOT BUY</b>\n\n"
                                         f"📊 {pair}\n"
                                         f"💰 ${trade_amount:,.2f} @ ${price:,.2f}\n"
                                         f"💵 Balance: ${usd_balance:,.2f}\n"
                                         f"🎯 Confidence: {confidence}%\n"
                                         f"📝 {signal.get('reasoning', '')}\n\n"
                                         f"Trades: {autopilot_settings['trades_today']}/{autopilot_settings['max_daily_trades']}",
                                    parse_mode="HTML"
                                )
                            except:
                                pass
                else:
                    # Paper trade
                    positions[pair] = {
                        "entry_price": price,
                        "highest_price": price,
                        "amount_usd": trade_amount,
                        "timestamp": datetime.now().isoformat(),
                        "live": False,
                        "autopilot": True
                    }
                    autopilot_settings["trades_today"] += 1
                    save_positions()
                    save_data("autopilot", autopilot_settings)
                    
                    for chat_id in AUTO_SIGNAL_CHATS:
                        try:
                            await tg_app.bot.send_message(
                                chat_id=chat_id,
                                text=f"🤖 <b>AUTOPILOT BUY (PAPER)</b>\n\n"
                                     f"📊 {pair} @ ${price:,.2f}\n"
                                     f"💰 ${trade_amount:,.2f}\n"
                                     f"🎯 Confidence: {confidence}%",
                                parse_mode="HTML"
                            )
                        except:
                            pass
                
        except Exception as e:
            print(f"[AUTOPILOT ERROR] {e}")


# ============ NEW BACKGROUND TASKS ============
async def daily_summary_scheduler():
    """Send daily summary at 9 PM."""
    while True:
        try:
            tz = pytz.timezone("America/Chicago")
            now = datetime.now(tz)
            
            # Send at 9 PM
            if now.hour == 21 and now.minute < 5:
                for chat_id in AUTO_SIGNAL_CHATS:
                    await send_daily_summary(chat_id)
                await asyncio.sleep(3600)  # Wait an hour to avoid duplicates
            else:
                await asyncio.sleep(60)
        except Exception as e:
            print(f"[SUMMARY ERROR] {e}")


async def dca_autopilot_scanner():
    """DCA autopilot - buy dips automatically."""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            if not dca_settings["enabled"]:
                continue
            
            if dca_settings["today_count"] >= dca_settings["max_per_day"]:
                continue
            
            # Check balance
            usd_balance = 0
            if cdp_client:
                usd_balance = await cdp_client.get_usd_balance()
            
            if usd_balance < dca_settings["amount_usd"]:
                continue
            
            for pair in dca_settings["coins"]:
                try:
                    # Get 24h data
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"https://api.exchange.coinbase.com/products/{pair}/stats") as resp:
                            if resp.status == 200:
                                stats = await resp.json()
                                high_24h = float(stats.get("high", 0))
                                current = float(stats.get("last", 0))
                                
                                if high_24h > 0:
                                    drop_pct = ((high_24h - current) / high_24h) * 100
                                    
                                    if drop_pct >= dca_settings["drop_pct"]:
                                        # DCA buy!
                                        if LIVE_TRADING and cdp_client:
                                            result = await cdp_client.place_market_order(pair, "BUY", dca_settings["amount_usd"])
                                            
                                            if result.get("success_response") or result.get("order_id"):
                                                dca_settings["today_count"] += 1
                                                
                                                # Track position
                                                if pair not in positions:
                                                    positions[pair] = {
                                                        "entry_price": current,
                                                        "highest_price": current,
                                                        "amount_usd": dca_settings["amount_usd"],
                                                        "timestamp": datetime.now().isoformat(),
                                                        "live": True,
                                                        "strategy": "dca_auto"
                                                    }
                                                else:
                                                    # Average down
                                                    old_amount = positions[pair]["amount_usd"]
                                                    old_entry = positions[pair]["entry_price"]
                                                    new_amount = old_amount + dca_settings["amount_usd"]
                                                    positions[pair]["entry_price"] = ((old_entry * old_amount) + (current * dca_settings["amount_usd"])) / new_amount
                                                    positions[pair]["amount_usd"] = new_amount
                                                
                                                save_positions()
                                                
                                                # Alert
                                                for chat_id in AUTO_SIGNAL_CHATS:
                                                    try:
                                                        await tg_app.bot.send_message(
                                                            chat_id=chat_id,
                                                            text=f"📉 <b>DCA AUTO-BUY</b>\n\n"
                                                                 f"📊 {pair}\n"
                                                                 f"💰 ${dca_settings['amount_usd']} @ ${current:,.2f}\n"
                                                                 f"📉 Drop: {drop_pct:.1f}% from 24h high\n"
                                                                 f"🔢 DCA Today: {dca_settings['today_count']}/{dca_settings['max_per_day']}",
                                                            parse_mode="HTML"
                                                        )
                                                    except:
                                                        pass
                except Exception as e:
                    print(f"[DCA] Error for {pair}: {e}")
            
        except Exception as e:
            print(f"[DCA ERROR] {e}")


async def risk_monitor():
    """Monitor risk limits and pause trading if needed."""
    while True:
        try:
            await asyncio.sleep(120)  # Check every 2 minutes
            
            # Get current portfolio value
            total_value = 0
            if cdp_client:
                holdings = await cdp_client.get_all_holdings()
                total_value = holdings.get("total_value", 0)
            
            # Update highest balance
            if total_value > risk_state["highest_balance"]:
                risk_state["highest_balance"] = total_value
                save_positions()
            
            # Calculate drawdown
            if risk_state["highest_balance"] > 0:
                drawdown = ((risk_state["highest_balance"] - total_value) / risk_state["highest_balance"]) * 100
                risk_state["current_drawdown"] = drawdown
                
                # Check max drawdown
                if drawdown >= risk_state["max_drawdown_pct"]:
                    if autopilot_settings["enabled"]:
                        autopilot_settings["enabled"] = False
                        risk_state["paused_reason"] = f"Max drawdown ({drawdown:.1f}%)"
                        save_positions()
                        
                        for chat_id in AUTO_SIGNAL_CHATS:
                            try:
                                await tg_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"🚨 <b>RISK ALERT - AUTOPILOT PAUSED</b>\n\n"
                                         f"📉 Drawdown: {drawdown:.1f}%\n"
                                         f"⚠️ Exceeded {risk_state['max_drawdown_pct']}% limit\n\n"
                                         f"Use /autopilot on to resume",
                                    parse_mode="HTML"
                                )
                            except:
                                pass
            
            # Check daily loss limit
            if daily_pnl["realized"] <= -risk_state["daily_loss_limit"]:
                if autopilot_settings["enabled"]:
                    autopilot_settings["enabled"] = False
                    risk_state["paused_reason"] = f"Daily loss limit (${abs(daily_pnl['realized']):.2f})"
                    save_positions()
                    
                    for chat_id in AUTO_SIGNAL_CHATS:
                        try:
                            await tg_app.bot.send_message(
                                chat_id=chat_id,
                                text=f"🚨 <b>RISK ALERT - AUTOPILOT PAUSED</b>\n\n"
                                     f"💔 Today's Loss: ${abs(daily_pnl['realized']):.2f}\n"
                                     f"⚠️ Exceeded ${risk_state['daily_loss_limit']} limit\n\n"
                                     f"Use /autopilot on to resume",
                                parse_mode="HTML"
                            )
                        except:
                            pass
            
        except Exception as e:
            print(f"[RISK MONITOR ERROR] {e}")


async def tp_tiers_monitor():
    """Monitor positions for take profit tiers."""
    while True:
        try:
            await asyncio.sleep(60)
            
            if not tp_tiers["enabled"]:
                continue
            
            for pair, pos in list(positions.items()):
                current = await get_public_price(pair)
                entry = pos["entry_price"]
                pnl_pct = ((current - entry) / entry) * 100
                
                # Check each tier
                tiers_hit = pos.get("tiers_hit", [])
                
                for tier in tp_tiers["tiers"]:
                    tier_key = f"{tier['pct_gain']}"
                    
                    if tier_key in tiers_hit:
                        continue
                    
                    if pnl_pct >= tier["pct_gain"]:
                        sell_pct = tier["sell_pct"]
                        sell_amount = pos["amount_usd"] * (sell_pct / 100)
                        
                        if LIVE_TRADING and cdp_client:
                            result = await cdp_client.place_market_order(pair, "SELL", sell_amount)
                            
                            if result.get("success_response") or result.get("order_id"):
                                # Update position
                                positions[pair]["amount_usd"] -= sell_amount
                                if "tiers_hit" not in positions[pair]:
                                    positions[pair]["tiers_hit"] = []
                                positions[pair]["tiers_hit"].append(tier_key)
                                
                                # Track partial P&L
                                partial_pnl = sell_amount * (pnl_pct / 100)
                                daily_pnl["realized"] += partial_pnl
                                
                                save_positions()
                                
                                # Alert
                                for chat_id in AUTO_SIGNAL_CHATS:
                                    try:
                                        await tg_app.bot.send_message(
                                            chat_id=chat_id,
                                            text=f"🎯 <b>TP TIER HIT!</b>\n\n"
                                                 f"📊 {pair}\n"
                                                 f"📈 +{pnl_pct:.1f}% reached\n"
                                                 f"💸 Sold {sell_pct}% (${sell_amount:.2f})\n"
                                                 f"💰 P&L: ${partial_pnl:.2f}\n\n"
                                                 f"Remaining: ${positions[pair]['amount_usd']:.2f}",
                                            parse_mode="HTML"
                                        )
                                    except:
                                        pass
                        break  # Only hit one tier per cycle
                
        except Exception as e:
            log_error("tp_tiers", e, "tp_tiers_monitor")


# ============ FASTAPI SETUP ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track uptime
START_TIME = None


@app.on_event("startup")
async def on_startup():
    global START_TIME
    START_TIME = datetime.now()
    
    logger.info("🚀 Starting QuantSignals Ultra...")
    
    # Load state
    load_positions()
    logger.info(f"📦 Loaded {len(positions)} positions, {len(trade_history)} trades")
    
    # Initialize Telegram
    await tg_app.initialize()
    await tg_app.start()
    
    if BASE_URL:
        await tg_app.bot.set_webhook(url=f"{BASE_URL}/webhook/{WEBHOOK_SECRET}")
        logger.info(f"✅ Webhook set: {BASE_URL}/webhook/***")
    
    # Start all background tasks with error handling
    tasks = [
        ("stop_loss_monitor", stop_loss_monitor()),
        ("auto_signal_scheduler", auto_signal_scheduler()),
        ("price_alert_checker", price_alert_checker()),
        ("autopilot_scanner", autopilot_scanner()),
        ("daily_summary_scheduler", daily_summary_scheduler()),
        ("dca_autopilot_scanner", dca_autopilot_scanner()),
        ("risk_monitor", risk_monitor()),
        ("tp_tiers_monitor", tp_tiers_monitor()),
    ]
    
    for name, coro in tasks:
        asyncio.create_task(coro, name=name)
        logger.info(f"  ✓ {name} started")
    
    logger.info("✅ QuantSignals Ultra ready!")
    logger.info(f"  Mode: {'🔴 LIVE' if LIVE_TRADING else '🟡 PAPER'}")
    logger.info(f"  Autopilot: {'🟢 ON' if autopilot_settings.get('enabled') else '🔴 OFF'}")
    logger.info(f"  Coinbase: {'✅ Connected' if cdp_client else '❌ Not configured'}")
    logger.info(f"  Redis: {'✅ Connected' if redis_client else '⚠️ Memory only'}")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 Shutting down QuantSignals...")
    save_positions()
    logger.info("💾 State saved")
    await tg_app.stop()
    logger.info("👋 Goodbye!")


# ============ HEALTH & MONITORING ENDPOINTS ============
@app.get("/")
async def root():
    """Basic health check."""
    return {
        "status": "ok",
        "bot": "QuantSignals Ultra",
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check for monitoring."""
    uptime = (datetime.now() - START_TIME).total_seconds() if START_TIME else 0
    
    # Check Redis
    redis_ok = False
    if redis_client:
        try:
            redis_client.ping()
            redis_ok = True
        except:
            pass
    
    # Check Coinbase
    coinbase_ok = cdp_client is not None
    
    return {
        "status": "healthy",
        "uptime_seconds": int(uptime),
        "uptime_human": str(timedelta(seconds=int(uptime))),
        "components": {
            "redis": "ok" if redis_ok else "unavailable",
            "coinbase": "ok" if coinbase_ok else "not_configured",
            "telegram": "ok"
        },
        "trading": {
            "mode": "live" if LIVE_TRADING else "paper",
            "autopilot": autopilot_settings.get("enabled", False),
            "positions": len(positions),
            "trades_today": autopilot_settings.get("trades_today", 0)
        },
        "errors": {
            "total": sum(v for k, v in error_counts.items() if k.endswith("_errors")),
            "last": error_counts.get("last_error"),
            "last_time": error_counts.get("last_error_time")
        }
    }


@app.get("/stats")
async def get_stats():
    """Get trading statistics."""
    total_pnl = sum(t.get("pnl_usd", 0) for t in trade_history)
    wins = sum(1 for t in trade_history if t.get("pnl_usd", 0) > 0)
    total = len(trade_history)
    
    return {
        "portfolio": {
            "positions": len(positions),
            "pairs": list(positions.keys())
        },
        "performance": {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl": round(total_pnl, 2)
        },
        "today": {
            "pnl": round(daily_pnl.get("realized", 0), 2),
            "trades": daily_pnl.get("trades", 0),
            "wins": daily_pnl.get("wins", 0)
        },
        "autopilot": {
            "enabled": autopilot_settings.get("enabled", False),
            "trades_today": autopilot_settings.get("trades_today", 0),
            "total_profit": round(autopilot_settings.get("total_profit", 0), 2)
        },
        "risk": {
            "drawdown": round(risk_state.get("current_drawdown", 0), 1),
            "max_drawdown": risk_state.get("max_drawdown_pct", 10),
            "paused": risk_state.get("paused_reason")
        }
    }


@app.get("/positions")
async def get_positions():
    """Get current positions with live P&L."""
    result = []
    for pair, pos in positions.items():
        try:
            current = await get_public_price(pair)
            entry = pos["entry_price"]
            pnl_pct = ((current - entry) / entry) * 100
            pnl_usd = (pnl_pct / 100) * pos.get("amount_usd", 0)
            
            result.append({
                "pair": pair,
                "entry_price": entry,
                "current_price": current,
                "amount_usd": pos.get("amount_usd", 0),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_usd": round(pnl_usd, 2),
                "opened_at": pos.get("timestamp"),
                "strategy": pos.get("strategy", "manual"),
                "autopilot": pos.get("autopilot", False)
            })
        except Exception as e:
            log_error("api", e, f"get_positions({pair})")
    
    return {"positions": result, "count": len(result)}


@app.get("/signals")
async def get_signals_api():
    """Get current AI signals."""
    try:
        signals = await generate_trading_signals()
        return signals
    except Exception as e:
        log_error("api", e, "get_signals_api")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    """Telegram webhook handler."""
    if secret != WEBHOOK_SECRET:
        logger.warning(f"Invalid webhook secret attempted")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    try:
        data = await request.json()
        update = Update.de_json(data, tg_app.bot)
        await tg_app.process_update(update)
        return {"ok": True}
    except Exception as e:
        log_error("telegram", e, "webhook")
        return {"ok": False, "error": str(e)}


# ============ ADMIN ENDPOINTS ============
@app.post("/admin/pause")
async def admin_pause():
    """Pause autopilot (for emergencies)."""
    autopilot_settings["enabled"] = False
    risk_state["paused_reason"] = "Admin pause"
    save_positions()
    logger.warning("⚠️ Autopilot paused by admin")
    return {"status": "paused"}


@app.post("/admin/resume")
async def admin_resume():
    """Resume autopilot."""
    autopilot_settings["enabled"] = True
    risk_state["paused_reason"] = None
    save_positions()
    logger.info("✅ Autopilot resumed by admin")
    return {"status": "resumed"}


@app.get("/admin/errors")
async def admin_errors():
    """Get error log."""
    return error_counts
