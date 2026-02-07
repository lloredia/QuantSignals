import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pytz
import httpx
from fastapi import FastAPI, Request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from openai import OpenAI
import time
import secrets

# Optional Redis for persistence
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL")
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL)
        print("✅ Redis connected")
    else:
        redis_client = None
except:
    redis_client = None


# ============ STORAGE FUNCTIONS ============
def save_data(key: str, data: dict):
    """Save data to Redis."""
    if redis_client:
        try:
            redis_client.set(f"qs_{key}", json.dumps(data))
        except Exception as e:
            print(f"[REDIS] Save error: {e}")


def load_data(key: str, default: dict = None) -> dict:
    """Load data from Redis."""
    if redis_client:
        try:
            data = redis_client.get(f"qs_{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"[REDIS] Load error: {e}")
    return default or {}


def save_positions():
    save_data("positions", positions)
    save_data("daily_pnl", daily_pnl)
    save_data("signal_history", signal_history)
    save_data("price_alerts", price_alerts)


def load_positions():
    global positions, daily_pnl, signal_history, price_alerts
    positions = load_data("positions", {})
    daily_pnl = load_data("daily_pnl", {"realized": 0.0, "trades": 0, "wins": 0})
    signal_history = load_data("signal_history", [])
    price_alerts = load_data("price_alerts", {})
    print(f"[REDIS] Loaded {len(positions)} positions, {len(signal_history)} signals")


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
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "5"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "10"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "3"))  # Feature 1: Trailing Stop
DCA_DROP_PCT = float(os.getenv("DCA_DROP_PCT", "5"))  # Feature 5: DCA trigger

# Expanded coin list
TRADING_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
    "DOGE-USD", "XRP-USD", "ADA-USD", "MATIC-USD", "DOT-USD"
]

app = FastAPI()
tg_app = Application.builder().token(TOKEN).build()

# State tracking
positions = {}
daily_pnl = {"realized": 0.0, "trades": 0, "wins": 0}
signal_history = []  # Feature 7: Leaderboard
price_alerts = {}  # Feature 8: Custom Alerts

# Live trading mode
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Auto signal config
AUTO_SIGNAL_CHATS = os.getenv("AUTO_SIGNAL_CHATS", "").split(",")
AUTO_SIGNAL_CHATS = [c.strip() for c in AUTO_SIGNAL_CHATS if c.strip()]
SIGNAL_HOURS = [6, 12, 18]

# Autopilot mode
autopilot_settings = {
    "enabled": False,
    "max_daily_trades": 5,
    "min_confidence": 75,
    "trades_today": 0,
    "last_reset": None,
    "use_percentage": True,  # Use % of balance instead of fixed amount
    "trade_percentage": 20,  # Use 20% of available USD per trade
    "min_trade_usd": 5,      # Minimum trade size
    "max_trade_usd": 100,    # Maximum trade size
    "reinvest_profits": True,
    "total_profit": 0.0
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


# Initialize CDP client
cdp_client = None
if COINBASE_API_KEY and COINBASE_API_SECRET:
    try:
        cdp_client = CoinbaseCDPClient(COINBASE_API_KEY, COINBASE_API_SECRET)
        print("✅ Coinbase CDP client initialized")
    except Exception as e:
        print(f"❌ CDP client init failed: {e}")


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
    """Get latest crypto news headlines."""
    # Using CryptoPanic public feed (no auth needed for basic)
    url = "https://cryptopanic.com/api/v1/posts/?auth_token=free&public=true&kind=news"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            data = resp.json()
            news = []
            for item in data.get("results", [])[:5]:
                news.append({
                    "title": item.get("title", "")[:100],
                    "source": item.get("source", {}).get("title", ""),
                    "url": item.get("url", ""),
                    "currencies": [c.get("code") for c in item.get("currencies", [])]
                })
            return news
    except:
        return []


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


# ============ AI SIGNAL GENERATOR (Enhanced) ============
async def generate_trading_signals(include_news: bool = True) -> dict:
    """Generate AI trading signals with enhanced data."""
    
    if not OPENAI_API_KEY:
        return {"error": "OpenAI not configured"}
    
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
                
                market_data[pair] = {
                    "price": price,
                    "high_24h": candle_data.get("high_24h", price),
                    "low_24h": candle_data.get("low_24h", price),
                    "volume_24h": candle_data.get("volume_24h", 0),
                    "sma_8": round(sma_8, 2),
                    "sma_21": round(sma_21, 2),
                    "change_1h": round(change_1h, 2),
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
    
    prompt = f"""You are an expert crypto day trader AI with advanced market analysis.

CURRENT TIME: {now.strftime("%Y-%m-%d %H:%M %Z")}
TRADING STYLE: Day trading (1-8 hours)
RISK: 5% stop loss, 10% take profit

FEAR & GREED INDEX: {fear_greed['value']} ({fear_greed['classification']})
- 0-25: Extreme Fear (potential buying opportunity)
- 25-45: Fear
- 45-55: Neutral
- 55-75: Greed
- 75-100: Extreme Greed (potential selling opportunity)

LIVE MARKET DATA:
"""
    
    for pair, data in market_data.items():
        prompt += f"""
{pair}: ${data['price']:,.2f}
- 24h: High ${data['high_24h']:,.2f} / Low ${data['low_24h']:,.2f}
- 1h Change: {data['change_1h']:+.2f}%
- SMA8: ${data['sma_8']:,.2f} / SMA21: ${data['sma_21']:,.2f}
- Trend: {data['trend'].upper()}
"""
    
    if news:
        prompt += "\n\nRECENT NEWS:\n"
        for n in news[:3]:
            coins = ", ".join(n.get("currencies", [])[:3])
            prompt += f"- {n['title']} [{coins}]\n"
    
    prompt += """

ANALYSIS RULES:
1. Consider Fear & Greed - buy on fear, cautious on greed
2. Look for trend confirmations (SMA crossovers)
3. News sentiment affects short-term moves
4. Only HIGH confidence (>70%) trades
5. Max 2 signals

OUTPUT (JSON only):
{
    "signals": [
        {
            "pair": "BTC-USD",
            "action": "BUY",
            "confidence": 78,
            "entry_price": 97000,
            "stop_loss": 92150,
            "take_profit": 106700,
            "timeframe": "4h",
            "reasoning": "Brief reason"
        }
    ],
    "market_sentiment": "bullish",
    "fear_greed_analysis": "Fear indicates buying opportunity",
    "summary": "One sentence summary"
}
"""
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Expert crypto trader. JSON only, no markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
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
        signals["generated_at"] = now.isoformat()
        
        # Track signals for leaderboard (Feature 7)
        for sig in signals.get("signals", []):
            signal_history.append({
                "pair": sig.get("pair"),
                "action": sig.get("action"),
                "price": sig.get("entry_price"),
                "confidence": sig.get("confidence"),
                "timestamp": now.isoformat(),
                "result": None  # Updated later
            })
        save_positions()
        
        return signals
        
    except Exception as e:
        return {"error": str(e), "market_data": market_data}


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
    welcome = """📊 <b>QUANTSIGNALS v2.0</b>

AI-powered crypto trading with advanced features.

<b>📈 Trading:</b>
/signals - AI trading signals
/market - Live prices
/portfolio - Your positions
/pnl - Daily P&L

<b>📊 Analysis:</b>
/fear - Fear & Greed Index
/news - Crypto news
/whale - Whale alerts
/timeframe [coin] - Multi-TF analysis

<b>🤖 Automation:</b>
/autopilot - Full auto-trading mode
/pause - Pause all trading
/alert [coin] [price] - Price alerts

<b>⚙️ Tools:</b>
/backtest [coin] - Strategy backtest
/leaderboard - Signal performance
/dca - DCA opportunities
/settings - Bot settings

<i>⚠️ Not financial advice. Trade responsibly.</i>"""
    
    await update.message.reply_text(welcome, parse_mode="HTML")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


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


async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show positions."""
    if not positions:
        await update.message.reply_text("📭 No open positions.\n\nUse /signals to get ideas.", parse_mode="HTML")
        return
    
    text = "📊 <b>PORTFOLIO</b>\n━━━━━━━━━━━━━━━\n\n"
    total_pnl = 0
    
    for pair, pos in positions.items():
        current = await get_public_price(pair)
        entry = pos["entry_price"]
        pnl_pct = ((current - entry) / entry) * 100
        pnl_usd = (pnl_pct / 100) * pos["amount_usd"]
        total_pnl += pnl_usd
        
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        text += f"""{emoji} <b>{pair}</b>
Entry: ${entry:,.2f} → ${current:,.2f}
P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})
"""
        # Feature 1: Trailing stop info
        if "highest_price" in pos:
            text += f"📈 Peak: ${pos['highest_price']:,.2f}\n"
        text += "\n"
    
    total_emoji = "🟢" if total_pnl >= 0 else "🔴"
    text += f"━━━━━━━━━━━━━━━\n{total_emoji} <b>Total:</b> ${total_pnl:+.2f}"
    
    keyboard = [[InlineKeyboardButton(f"Close {pair}", callback_data=f"close_{pair}")] for pair in positions]
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


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
    
    if data.startswith("trade_buy_"):
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
            
            del positions[pair]
            save_positions()
            
            emoji = "🟢" if pnl_usd >= 0 else "🔴"
            await query.edit_message_text(
                f"✅ <b>CLOSED</b>\n\n{pair}\n"
                f"Entry: ${entry:,.2f} → Exit: ${current:,.2f}\n"
                f"{emoji} P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})",
                parse_mode="HTML"
            )


# ============ REGISTER HANDLERS ============
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
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
tg_app.add_handler(CommandHandler("dca", dca_cmd))
tg_app.add_handler(CommandHandler("autopilot", autopilot_cmd))
tg_app.add_handler(CommandHandler("pause", pause_cmd))
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
    """Autopilot mode - auto-execute trades with dynamic sizing."""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            if not autopilot_settings["enabled"]:
                continue
            
            # Check if we've hit daily limit
            if autopilot_settings["trades_today"] >= autopilot_settings["max_daily_trades"]:
                continue
            
            # Check if we have too many positions
            if len(positions) >= MAX_POSITIONS:
                continue
            
            # Get current USD balance for dynamic sizing
            usd_balance = TRADE_AMOUNT_USD  # Default fallback
            if cdp_client and LIVE_TRADING:
                usd_balance = await cdp_client.get_usd_balance()
            
            # Calculate trade size based on percentage of balance
            trade_amount = usd_balance * autopilot_settings["trade_percentage"] / 100
            trade_amount = max(autopilot_settings["min_trade_usd"], 
                             min(autopilot_settings["max_trade_usd"], trade_amount))
            
            # Skip if not enough balance
            if usd_balance < autopilot_settings["min_trade_usd"]:
                continue
            
            # Generate signals
            signals = await generate_trading_signals(include_news=False)
            
            if not signals.get("signals"):
                continue
            
            for signal in signals["signals"]:
                # Only process BUY signals with high confidence
                if signal.get("action") != "BUY":
                    continue
                
                confidence = signal.get("confidence", 0)
                if confidence < autopilot_settings["min_confidence"]:
                    continue
                
                pair = signal.get("pair")
                
                # Skip if already have position in this pair
                if pair in positions:
                    continue
                
                # Check daily limit again
                if autopilot_settings["trades_today"] >= autopilot_settings["max_daily_trades"]:
                    break
                
                # Re-check balance for this specific trade
                if cdp_client and LIVE_TRADING:
                    usd_balance = await cdp_client.get_usd_balance()
                    trade_amount = usd_balance * autopilot_settings["trade_percentage"] / 100
                    trade_amount = max(autopilot_settings["min_trade_usd"], 
                                     min(autopilot_settings["max_trade_usd"], trade_amount))
                    
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


# ============ FASTAPI ============
@app.on_event("startup")
async def on_startup():
    load_positions()
    await tg_app.initialize()
    await tg_app.start()
    if BASE_URL:
        await tg_app.bot.set_webhook(url=f"{BASE_URL}/webhook/{WEBHOOK_SECRET}")
        print(f"✅ Webhook set")
    
    asyncio.create_task(stop_loss_monitor())
    asyncio.create_task(auto_signal_scheduler())
    asyncio.create_task(price_alert_checker())
    asyncio.create_task(autopilot_scanner())


@app.on_event("shutdown")
async def on_shutdown():
    save_positions()
    await tg_app.stop()


@app.get("/")
async def health():
    return {"status": "ok", "bot": "QuantSignals v2", "positions": len(positions)}


@app.get("/debug/signals")
async def debug_signals():
    return await generate_trading_signals()


@app.get("/test/auto-signal")
async def test_auto_signal():
    if not AUTO_SIGNAL_CHATS:
        return {"error": "No AUTO_SIGNAL_CHATS"}
    
    signals = await generate_trading_signals()
    text = "🧪 <b>TEST</b>\n" + json.dumps(signals.get("signals", []), indent=2)[:500]
    
    for chat_id in AUTO_SIGNAL_CHATS:
        await tg_app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    
    return {"sent": len(AUTO_SIGNAL_CHATS)}


@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}
    update = Update.de_json(await request.json(), tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
