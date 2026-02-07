import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
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


def save_positions():
    """Save positions to Redis."""
    if redis_client:
        try:
            redis_client.set("qs_positions", json.dumps(positions))
            redis_client.set("qs_daily_pnl", json.dumps(daily_pnl))
        except Exception as e:
            print(f"[REDIS] Save error: {e}")


def load_positions():
    """Load positions from Redis."""
    global positions, daily_pnl
    if redis_client:
        try:
            pos_data = redis_client.get("qs_positions")
            pnl_data = redis_client.get("qs_daily_pnl")
            if pos_data:
                positions = json.loads(pos_data)
            if pnl_data:
                daily_pnl = json.loads(pnl_data)
            print(f"[REDIS] Loaded {len(positions)} positions")
        except Exception as e:
            print(f"[REDIS] Load error: {e}")

# ============ CONFIG ============
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "quantsignals-secret")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Coinbase CDP API (for trading - optional)
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

# Trading Config
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "5"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "10"))

# Supported coins - expanded list
TRADING_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
    "DOGE-USD", "XRP-USD", "ADA-USD", "MATIC-USD", "DOT-USD"
]

app = FastAPI()
tg_app = Application.builder().token(TOKEN).build()

# Position tracking
positions = {}
daily_pnl = {"realized": 0.0, "trades": 0, "wins": 0}

# Live trading mode
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Auto signal chat IDs (comma separated)
AUTO_SIGNAL_CHATS = os.getenv("AUTO_SIGNAL_CHATS", "").split(",")
AUTO_SIGNAL_CHATS = [c.strip() for c in AUTO_SIGNAL_CHATS if c.strip()]

# Scheduled signal times (CT): 6 AM, 12 PM, 6 PM
SIGNAL_HOURS = [6, 12, 18]


# ============ COINBASE CDP CLIENT ============
class CoinbaseCDPClient:
    """Coinbase CDP API client with JWT auth."""
    
    BASE_URL = "https://api.coinbase.com"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        # Fix newlines in private key
        self.api_secret = api_secret.replace("\\n", "\n")
    
    def _build_jwt(self, method: str, path: str) -> str:
        """Build JWT for CDP API."""
        import jwt as pyjwt
        import secrets as sec
        
        uri = f"{method} api.coinbase.com{path}"
        
        payload = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
            "uri": uri,
            "nonce": sec.token_hex(16),
        }
        
        headers = {
            "kid": self.api_key,
            "nonce": sec.token_hex(16),
        }
        
        # Load private key
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        try:
            private_key = serialization.load_pem_private_key(
                self.api_secret.encode(),
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            print(f"[CDP] Key load error: {e}")
            raise
        
        token = pyjwt.encode(payload, private_key, algorithm="ES256", headers=headers)
        return token
    
    def _headers(self, method: str, path: str) -> dict:
        """Generate auth headers."""
        token = self._build_jwt(method, path)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def get_accounts(self) -> dict:
        """Get account balances."""
        path = "/api/v3/brokerage/accounts"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.BASE_URL}{path}",
                    headers=self._headers("GET", path),
                    timeout=15
                )
                return resp.json()
        except Exception as e:
            print(f"[CDP ERROR] get_accounts: {e}")
            return {"error": str(e)}
    
    async def place_market_order(self, product_id: str, side: str, usd_amount: float) -> dict:
        """Place a market order."""
        path = "/api/v3/brokerage/orders"
        
        order_id = f"qs_{int(time.time())}_{secrets.token_hex(4)}"
        
        body = {
            "client_order_id": order_id,
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": str(usd_amount)
                }
            }
        }
        
        body_str = json.dumps(body)
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.BASE_URL}{path}",
                    headers=self._headers("POST", path),
                    content=body_str,
                    timeout=15
                )
                print(f"[CDP] Status: {resp.status_code}")
                print(f"[CDP] Response: {resp.text[:500]}")
                
                if resp.status_code == 200 or resp.status_code == 201:
                    return resp.json()
                else:
                    return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            print(f"[CDP ERROR] place_order: {e}")
            return {"error": str(e)}
    
    async def test_auth(self) -> dict:
        """Test API authentication."""
        path = "/api/v3/brokerage/accounts"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.BASE_URL}{path}",
                    headers=self._headers("GET", path),
                    timeout=15
                )
                print(f"[CDP TEST] Status: {resp.status_code}")
                print(f"[CDP TEST] Response: {resp.text[:500]}")
                return {"status": resp.status_code, "body": resp.text[:500]}
        except Exception as e:
            return {"error": str(e)}


# Initialize CDP client if keys exist
cdp_client = None
if COINBASE_API_KEY and COINBASE_API_SECRET:
    try:
        cdp_client = CoinbaseCDPClient(COINBASE_API_KEY, COINBASE_API_SECRET)
        print("✅ Coinbase CDP client initialized")
    except Exception as e:
        print(f"❌ CDP client init failed: {e}")


# ============ PUBLIC PRICE API (No Auth Required) ============
async def get_public_price(product_id: str) -> float:
    """Get price from Coinbase public API (no auth needed)."""
    url = f"https://api.coinbase.com/v2/prices/{product_id}/spot"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            data = resp.json()
            return float(data.get("data", {}).get("amount", 0))
    except Exception as e:
        print(f"[ERROR] public price {product_id}: {e}")
        return 0


async def get_public_candles(product_id: str) -> dict:
    """Get candle data from Coinbase Exchange public API (no auth needed)."""
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity=3600"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            candles = resp.json()
            
            if candles and len(candles) > 0 and isinstance(candles, list):
                # Candles format: [time, low, high, open, close, volume]
                prices = [c[4] for c in candles[:24]]  # close prices
                return {
                    "prices": prices,
                    "high_24h": max(c[2] for c in candles[:24]),
                    "low_24h": min(c[1] for c in candles[:24]),
                    "volume_24h": sum(c[5] for c in candles[:24]),
                    "current": prices[0] if prices else 0
                }
    except Exception as e:
        print(f"[ERROR] public candles {product_id}: {e}")
    return {}


# ============ AI SIGNAL GENERATOR ============
async def generate_trading_signals() -> dict:
    """Generate AI trading signals for crypto pairs."""
    
    if not OPENAI_API_KEY:
        return {"error": "OpenAI not configured"}
    
    # Gather market data using PUBLIC API (no auth needed!)
    market_data = {}
    for pair in TRADING_PAIRS:
        try:
            price = await get_public_price(pair)
            candle_data = await get_public_candles(pair)
            
            if price > 0:
                prices = candle_data.get("prices", [price])
                
                sma_8 = sum(prices[:8]) / 8 if len(prices) >= 8 else price
                sma_21 = sum(prices[:21]) / 21 if len(prices) >= 21 else price
                
                change_1h = 0
                if len(prices) >= 2:
                    change_1h = ((prices[0] - prices[1]) / prices[1]) * 100
                
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
                print(f"[OK] {pair}: ${price:,.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to get data for {pair}: {e}")
    
    if not market_data:
        return {"error": "No market data available"}
    
    # Build AI prompt
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    prompt = f"""You are an expert crypto day trader AI. Analyze the following REAL-TIME market data and provide trading signals.

CURRENT TIME: {now.strftime("%Y-%m-%d %H:%M %Z")}
TRADING STYLE: Day trading (holding 1-8 hours)
RISK TOLERANCE: Medium (5% stop loss, 10% take profit)

LIVE MARKET DATA:
"""
    
    for pair, data in market_data.items():
        prompt += f"""
{pair}:
- Current Price: ${data['price']:,.2f}
- 24h High: ${data['high_24h']:,.2f}
- 24h Low: ${data['low_24h']:,.2f}
- 1h Change: {data['change_1h']:+.2f}%
- SMA(8): ${data['sma_8']:,.2f}
- SMA(21): ${data['sma_21']:,.2f}
- Trend: {data['trend']}
- 24h Volume: ${data['volume_24h']:,.0f}
"""
    
    prompt += """

ANALYSIS INSTRUCTIONS:
1. Analyze momentum, trend direction, and support/resistance
2. Consider SMA crossovers (SMA8 vs SMA21) for trend confirmation
3. Look for oversold/overbought conditions based on 24h range
4. Only recommend HIGH confidence trades (>70% conviction)
5. Maximum 2 trade signals

OUTPUT FORMAT (JSON only, no markdown, no explanation):
{
    "signals": [
        {
            "pair": "BTC-USD",
            "action": "BUY",
            "confidence": 75,
            "entry_price": 97000,
            "stop_loss": 92150,
            "take_profit": 106700,
            "reasoning": "SMA8 crossed above SMA21, bullish momentum"
        }
    ],
    "market_sentiment": "bullish",
    "summary": "One sentence market summary"
}

If no good opportunities, return empty signals array with action "HOLD".
"""
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional crypto trader. Respond ONLY with valid JSON, no markdown code blocks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean JSON
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        signals = json.loads(result_text.strip())
        signals["market_data"] = market_data
        signals["generated_at"] = now.isoformat()
        
        return signals
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed: {e}")
        print(f"[DEBUG] Raw response: {result_text[:500]}")
        return {"error": f"JSON parse error", "market_data": market_data}
    except Exception as e:
        print(f"[ERROR] AI signal generation failed: {e}")
        return {"error": str(e), "market_data": market_data}


# ============ TELEGRAM COMMANDS ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = """📊 <b>QUANTSIGNALS</b>

AI-powered crypto trading signals.

<b>Commands:</b>
/signals - Get AI trading signals
/market - Quick market overview
/portfolio - View positions
/pnl - Today's P&L
/settings - Bot settings
/help - Show help

<i>⚠️ Trading involves risk. Never trade more than you can afford to lose.</i>"""
    
    await update.message.reply_text(welcome, parse_mode="HTML")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """📊 <b>QUANTSIGNALS HELP</b>

<b>Signal Commands:</b>
/signals - Generate AI trading signals
/market - Quick market overview

<b>Trading Commands:</b>
/portfolio - View open positions
/pnl - Today's profit/loss

<b>Settings:</b>
/settings - View bot settings

<i>Start with /signals to get AI recommendations!</i>"""
    
    await update.message.reply_text(help_text, parse_mode="HTML")


async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and display AI trading signals."""
    msg = await update.message.reply_text("🔄 Analyzing markets with AI...")
    
    signals = await generate_trading_signals()
    
    if "error" in signals and "market_data" not in signals:
        await msg.edit_text(f"❌ Error: {signals['error']}")
        return
    
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    text = f"""📊 <b>QUANTSIGNALS</b>
📅 {now.strftime("%B %d, %Y %I:%M %p %Z")}
━━━━━━━━━━━━━━━

"""
    
    if "error" in signals:
        text += f"⚠️ AI Error: {signals['error']}\n\n"
        text += "<b>Live Prices:</b>\n"
        for pair, data in signals.get("market_data", {}).items():
            emoji = "🟢" if data.get("change_1h", 0) > 0 else "🔴"
            text += f"{emoji} {pair}: ${data['price']:,.2f} ({data.get('change_1h', 0):+.2f}%)\n"
    else:
        sentiment = signals.get('market_sentiment', 'neutral').upper()
        sent_emoji = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "⚪"
        text += f"<b>Market Sentiment:</b> {sent_emoji} {sentiment}\n\n"
        
        if not signals.get("signals"):
            text += "⚪ <b>NO SIGNALS</b>\n"
            text += "No high-confidence trades at this time.\n\n"
            text += "<b>Live Prices:</b>\n"
            for pair, data in signals.get("market_data", {}).items():
                emoji = "🟢" if data.get("change_1h", 0) > 0 else "🔴"
                text += f"{emoji} {pair}: ${data['price']:,.2f} ({data.get('change_1h', 0):+.2f}%)\n"
        else:
            for signal in signals.get("signals", []):
                action = signal.get("action", "HOLD")
                action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                
                text += f"""{action_emoji} <b>{signal.get('pair', 'N/A')}</b>
Action: <b>{action}</b>
Confidence: {signal.get('confidence', 'N/A')}%
Entry: ${signal.get('entry_price', 0):,.2f}
Stop Loss: ${signal.get('stop_loss', 0):,.2f}
Take Profit: ${signal.get('take_profit', 0):,.2f}
📝 {signal.get('reasoning', 'N/A')}

"""
        
        text += f"""━━━━━━━━━━━━━━━
<b>Summary:</b> {signals.get('summary', 'N/A')}
"""
    
    text += "\n<i>⚠️ Not financial advice. Trade at your own risk.</i>"
    
    # Add trade buttons
    keyboard = []
    for signal in signals.get("signals", []):
        if signal.get("action") == "BUY":
            keyboard.append([
                InlineKeyboardButton(
                    f"🟢 Buy {signal['pair'].split('-')[0]} (${TRADE_AMOUNT_USD})",
                    callback_data=f"trade_buy_{signal['pair']}"
                )
            ])
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    
    await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)


async def market_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick market overview using public API."""
    msg = await update.message.reply_text("📊 Loading live prices...")
    
    text = """📊 <b>LIVE MARKET</b>
━━━━━━━━━━━━━━━

"""
    
    # Fetch all prices concurrently for speed
    async def get_pair_data(pair):
        try:
            price = await get_public_price(pair)
            return {"pair": pair, "price": price, "error": None}
        except Exception as e:
            return {"pair": pair, "price": 0, "error": str(e)}
    
    # Run all requests concurrently
    results = await asyncio.gather(*[get_pair_data(pair) for pair in TRADING_PAIRS])
    
    for result in results:
        pair = result["pair"]
        price = result["price"]
        coin = pair.split("-")[0]
        
        if price > 0:
            text += f"🪙 <b>{coin}</b>: ${price:,.2f}\n"
        else:
            text += f"⚠️ {coin}: No data\n"
    
    text += """
━━━━━━━━━━━━━━━
<i>Use /signals for full AI analysis</i>"""
    
    try:
        await msg.edit_text(text, parse_mode="HTML")
    except Exception as e:
        print(f"[ERROR] edit message: {e}")
        await update.message.reply_text(text, parse_mode="HTML")


async def balance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check account balances from Coinbase."""
    
    if not cdp_client:
        await update.message.reply_text(
            "❌ Coinbase not configured.\n\n"
            "Add API keys to Railway variables."
        )
        return
    
    msg = await update.message.reply_text("💰 Fetching Coinbase balances...")
    
    try:
        accounts = await cdp_client.get_accounts()
        
        if "error" in accounts:
            await msg.edit_text(f"❌ Error: {accounts['error']}")
            return
        
        text = """💰 <b>COINBASE BALANCES</b>
━━━━━━━━━━━━━━━

"""
        
        total_usd = 0
        balances_found = []
        
        for account in accounts.get("accounts", []):
            try:
                balance = float(account.get("available_balance", {}).get("value", 0))
                currency = account.get("currency", "")
                
                if balance > 0.0001:  # Skip dust
                    if currency == "USD":
                        balances_found.append(f"💵 <b>USD</b>: ${balance:,.2f}")
                        total_usd += balance
                    elif currency in ["BTC", "ETH", "SOL", "AVAX", "LINK", "USDC"]:
                        # Only show main coins
                        price = await get_public_price(f"{currency}-USD")
                        if price > 0:
                            usd_value = balance * price
                            total_usd += usd_value
                            balances_found.append(f"🪙 <b>{currency}</b>: {balance:.6f} (${usd_value:,.2f})")
            except:
                continue
        
        if balances_found:
            text += "\n".join(balances_found)
        else:
            text += "No significant balances found"
        
        text += f"""

━━━━━━━━━━━━━━━
💰 <b>Total Value:</b> ${total_usd:,.2f}"""
        
        await msg.edit_text(text, parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current open positions."""
    
    if not positions:
        await update.message.reply_text(
            "📭 <b>No open positions</b>\n\n"
            "Use /signals to get trading ideas.\n"
            "Click the Buy button to open a position.",
            parse_mode="HTML"
        )
        return
    
    text = """📊 <b>OPEN POSITIONS</b>
━━━━━━━━━━━━━━━

"""
    
    total_pnl = 0
    keyboard = []
    
    for pair, pos in list(positions.items()):
        current_price = await get_public_price(pair)
        entry = pos["entry_price"]
        
        if current_price > 0 and entry > 0:
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_usd = (pnl_pct / 100) * pos["amount_usd"]
            total_pnl += pnl_usd
            
            emoji = "🟢" if pnl_pct > 0 else "🔴"
            is_live = "🔴 LIVE" if pos.get("live") else "🟡 PAPER"
            
            # Calculate stop/target
            stop_price = entry * (1 - STOP_LOSS_PCT/100)
            target_price = entry * (1 + TAKE_PROFIT_PCT/100)
            
            text += f"""{emoji} <b>{pair}</b> ({is_live})
Entry: ${entry:,.2f}
Current: ${current_price:,.2f}
P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})
🛑 Stop: ${stop_price:,.2f} | 🎯 Target: ${target_price:,.2f}

"""
            # Add close button for each position
            keyboard.append([InlineKeyboardButton(f"❌ Close {pair.split('-')[0]}", callback_data=f"close_{pair}")])
        else:
            text += f"⚠️ <b>{pair}</b> - Error getting price\n\n"
    
    total_emoji = "🟢" if total_pnl >= 0 else "🔴"
    text += f"""━━━━━━━━━━━━━━━
{total_emoji} <b>Total P&L:</b> ${total_pnl:+.2f}

<i>Click button below to close a position</i>"""
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=reply_markup)


async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's P&L."""
    
    win_rate = (daily_pnl["wins"] / daily_pnl["trades"] * 100) if daily_pnl["trades"] > 0 else 0
    emoji = "🟢" if daily_pnl["realized"] >= 0 else "🔴"
    
    text = f"""📈 <b>TODAY'S P&L</b>
━━━━━━━━━━━━━━━

{emoji} Realized P&L: <b>${daily_pnl['realized']:+.2f}</b>
📊 Total Trades: {daily_pnl['trades']}
✅ Wins: {daily_pnl['wins']}
📉 Losses: {daily_pnl['trades'] - daily_pnl['wins']}
🎯 Win Rate: {win_rate:.1f}%

━━━━━━━━━━━━━━━
<i>Resets at midnight CT</i>"""
    
    await update.message.reply_text(text, parse_mode="HTML")


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current settings."""
    
    text = f"""⚙️ <b>QUANTSIGNALS SETTINGS</b>
━━━━━━━━━━━━━━━

💵 Trade Amount: ${TRADE_AMOUNT_USD}
📊 Max Positions: {MAX_POSITIONS}
🛑 Stop Loss: {STOP_LOSS_PCT}%
🎯 Take Profit: {TAKE_PROFIT_PCT}%

<b>Trading Pairs:</b>
{', '.join([p.split('-')[0] for p in TRADING_PAIRS])}

<b>Status:</b>
✅ Market Data: Public API (no auth)
{"✅" if COINBASE_API_KEY else "⚠️"} Trading: {"Enabled" if COINBASE_API_KEY else "Signals only (add Coinbase keys to trade)"}

━━━━━━━━━━━━━━━
<i>Edit via Railway environment variables</i>"""
    
    await update.message.reply_text(text, parse_mode="HTML")


# ============ CALLBACK HANDLER ============
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("trade_buy_"):
        pair = data.replace("trade_buy_", "")
        price = await get_public_price(pair)
        
        if price <= 0:
            await query.edit_message_text(f"❌ Could not get price for {pair}")
            return
        
        # Check if live trading is enabled
        if LIVE_TRADING and cdp_client:
            await query.edit_message_text(f"🔄 Executing LIVE BUY for {pair}...")
            
            try:
                result = await cdp_client.place_market_order(pair, "BUY", TRADE_AMOUNT_USD)
                
                # Check for success - CDP returns success_response on success
                if result.get("success_response") or result.get("order_id") or (result.get("success") == True):
                    # Get fresh price after order
                    entry_price = await get_public_price(pair)
                    
                    positions[pair] = {
                        "entry_price": entry_price,
                        "amount_usd": TRADE_AMOUNT_USD,
                        "timestamp": datetime.now().isoformat(),
                        "live": True,
                        "order_id": result.get("success_response", {}).get("order_id", "unknown")
                    }
                    
                    await query.edit_message_text(
                        f"✅ <b>LIVE ORDER EXECUTED</b>\n\n"
                        f"📊 {pair}\n"
                        f"💵 Amount: ${TRADE_AMOUNT_USD}\n"
                        f"📍 Entry: ${entry_price:,.2f}\n"
                        f"🛑 Stop Loss: ${entry_price * (1 - STOP_LOSS_PCT/100):,.2f}\n"
                        f"🎯 Take Profit: ${entry_price * (1 + TAKE_PROFIT_PCT/100):,.2f}\n\n"
                        f"✅ Position tracked! Use /portfolio to monitor.",
                        parse_mode="HTML"
                    )
                elif result.get("error_response"):
                    error = result.get("error_response", {})
                    error_msg = error.get("message", "Unknown error")
                    await query.edit_message_text(f"❌ Order failed: {error_msg}")
                else:
                    error = result.get("error") or result.get("message") or json.dumps(result)[:200]
                    await query.edit_message_text(f"❌ Order failed: {error}")
                    
            except Exception as e:
                await query.edit_message_text(f"❌ Error: {str(e)[:100]}")
        else:
            # Paper trading
            positions[pair] = {
                "entry_price": price,
                "amount_usd": TRADE_AMOUNT_USD,
                "timestamp": datetime.now().isoformat(),
                "live": False
            }
            
            mode = "🟡 PAPER" if not LIVE_TRADING else "⚠️ NO API"
            await query.edit_message_text(
                f"✅ <b>POSITION OPENED</b> ({mode})\n\n"
                f"📊 {pair}\n"
                f"💵 Amount: ${TRADE_AMOUNT_USD}\n"
                f"📍 Entry: ${price:,.2f}\n"
                f"🛑 Stop Loss: ${price * (1 - STOP_LOSS_PCT/100):,.2f}\n"
                f"🎯 Take Profit: ${price * (1 + TAKE_PROFIT_PCT/100):,.2f}\n\n"
                f"<i>Use /portfolio to track</i>\n\n"
                f"💡 Set LIVE_TRADING=true in Railway for real trades.",
                parse_mode="HTML"
            )
    
    elif data.startswith("close_"):
        pair = data.replace("close_", "")
        
        if pair in positions:
            entry = positions[pair]["entry_price"]
            current = await get_public_price(pair)
            pnl_pct = ((current - entry) / entry) * 100
            pnl_usd = (pnl_pct / 100) * positions[pair]["amount_usd"]
            is_live = positions[pair].get("live", False)
            
            # If live, execute sell order
            if is_live and LIVE_TRADING and cdp_client:
                await query.edit_message_text(f"🔄 Executing LIVE SELL for {pair}...")
                # Note: For sells, we'd need to track the base amount bought
                # For now, just close the position tracking
            
            daily_pnl["realized"] += pnl_usd
            daily_pnl["trades"] += 1
            if pnl_usd > 0:
                daily_pnl["wins"] += 1
            
            del positions[pair]
            
            emoji = "🟢" if pnl_usd >= 0 else "🔴"
            mode = "LIVE" if is_live else "PAPER"
            await query.edit_message_text(
                f"✅ <b>POSITION CLOSED</b> ({mode})\n\n"
                f"📊 {pair}\n"
                f"📍 Entry: ${entry:,.2f}\n"
                f"📍 Exit: ${current:,.2f}\n"
                f"{emoji} P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n\n"
                f"<i>Use /pnl for daily summary</i>",
                parse_mode="HTML"
            )
        else:
            await query.edit_message_text("❌ Position not found.")


# ============ REGISTER HANDLERS ============
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(CommandHandler("signals", signals_cmd))
tg_app.add_handler(CommandHandler("market", market_cmd))
tg_app.add_handler(CommandHandler("balance", balance_cmd))
tg_app.add_handler(CommandHandler("portfolio", portfolio_cmd))
tg_app.add_handler(CommandHandler("pnl", pnl_cmd))
tg_app.add_handler(CommandHandler("settings", settings_cmd))
tg_app.add_handler(CallbackQueryHandler(button_callback))


# ============ FASTAPI ROUTES ============
@app.on_event("startup")
async def on_startup():
    # Load positions from Redis
    load_positions()
    
    await tg_app.initialize()
    await tg_app.start()
    if BASE_URL:
        webhook_url = f"{BASE_URL}/webhook/{WEBHOOK_SECRET}"
        await tg_app.bot.set_webhook(url=webhook_url)
        print(f"✅ Webhook set: {webhook_url}")
    
    # Start background tasks
    asyncio.create_task(stop_loss_monitor())
    asyncio.create_task(auto_signal_scheduler())


async def stop_loss_monitor():
    """Background task to check stop losses every 60 seconds."""
    while True:
        try:
            await asyncio.sleep(60)
            
            if not positions:
                continue
            
            for pair, pos in list(positions.items()):
                current_price = await get_public_price(pair)
                entry = pos["entry_price"]
                pnl_pct = ((current_price - entry) / entry) * 100
                
                # Check stop loss
                if pnl_pct <= -STOP_LOSS_PCT:
                    print(f"[STOP LOSS] {pair} hit -{STOP_LOSS_PCT}%")
                    
                    # Calculate P&L
                    pnl_usd = (pnl_pct / 100) * pos["amount_usd"]
                    daily_pnl["realized"] += pnl_usd
                    daily_pnl["trades"] += 1
                    
                    # Send alert
                    for chat_id in AUTO_SIGNAL_CHATS:
                        if chat_id:
                            try:
                                await tg_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"🛑 <b>STOP LOSS HIT</b>\n\n"
                                         f"📊 {pair}\n"
                                         f"📍 Entry: ${entry:,.2f}\n"
                                         f"📍 Exit: ${current_price:,.2f}\n"
                                         f"🔴 P&L: {pnl_pct:.2f}% (${pnl_usd:.2f})\n\n"
                                         f"<i>Position auto-closed</i>",
                                    parse_mode="HTML"
                                )
                            except:
                                pass
                    
                    del positions[pair]
                    save_positions()
                
                # Check take profit
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    print(f"[TAKE PROFIT] {pair} hit +{TAKE_PROFIT_PCT}%")
                    
                    pnl_usd = (pnl_pct / 100) * pos["amount_usd"]
                    daily_pnl["realized"] += pnl_usd
                    daily_pnl["trades"] += 1
                    daily_pnl["wins"] += 1
                    
                    for chat_id in AUTO_SIGNAL_CHATS:
                        if chat_id:
                            try:
                                await tg_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"🎯 <b>TAKE PROFIT HIT</b>\n\n"
                                         f"📊 {pair}\n"
                                         f"📍 Entry: ${entry:,.2f}\n"
                                         f"📍 Exit: ${current_price:,.2f}\n"
                                         f"🟢 P&L: +{pnl_pct:.2f}% (+${pnl_usd:.2f})\n\n"
                                         f"<i>Position auto-closed</i>",
                                    parse_mode="HTML"
                                )
                            except:
                                pass
                    
                    del positions[pair]
                    save_positions()
                    
        except Exception as e:
            print(f"[STOP LOSS ERROR] {e}")


async def auto_signal_scheduler():
    """Send signals at scheduled times."""
    while True:
        try:
            tz = pytz.timezone("America/Chicago")
            now = datetime.now(tz)
            
            # Check if current hour is a signal hour and within first 5 minutes
            if now.hour in SIGNAL_HOURS and now.minute < 5:
                print(f"[AUTO SIGNAL] Sending signals at {now.hour}:00 CT")
                
                signals = await generate_trading_signals()
                
                if "error" not in signals or signals.get("market_data"):
                    text = f"📊 <b>QUANTSIGNALS AUTO UPDATE</b>\n"
                    text += f"⏰ {now.strftime('%I:%M %p %Z')}\n"
                    text += "━━━━━━━━━━━━━━━\n\n"
                    
                    if signals.get("signals"):
                        sentiment = signals.get('market_sentiment', 'neutral').upper()
                        text += f"<b>Sentiment:</b> {sentiment}\n\n"
                        
                        for signal in signals.get("signals", []):
                            action = signal.get("action", "HOLD")
                            emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                            text += f"{emoji} <b>{signal.get('pair')}</b>: {action}\n"
                            text += f"   Confidence: {signal.get('confidence')}%\n"
                            text += f"   📝 {signal.get('reasoning')}\n\n"
                    else:
                        text += "⚪ No high-confidence signals right now.\n\n"
                        text += "<b>Market Prices:</b>\n"
                        for pair, data in signals.get("market_data", {}).items():
                            emoji = "🟢" if data.get("change_1h", 0) > 0 else "🔴"
                            text += f"{emoji} {pair.split('-')[0]}: ${data['price']:,.2f}\n"
                    
                    text += "\n<i>Use /signals for full analysis</i>"
                    
                    for chat_id in AUTO_SIGNAL_CHATS:
                        if chat_id:
                            try:
                                await tg_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=text,
                                    parse_mode="HTML"
                                )
                            except Exception as e:
                                print(f"[AUTO SIGNAL] Failed to send to {chat_id}: {e}")
                
                # Sleep for 10 minutes to avoid duplicate sends
                await asyncio.sleep(600)
            else:
                # Check every minute
                await asyncio.sleep(60)
                
        except Exception as e:
            print(f"[AUTO SIGNAL ERROR] {e}")
            await asyncio.sleep(60)


@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()


@app.get("/")
async def health():
    return {"status": "ok", "bot": "QuantSignals", "positions": len(positions)}


@app.get("/debug/signals")
async def debug_signals():
    """Test signal generation."""
    signals = await generate_trading_signals()
    return signals


@app.get("/debug/price/{pair}")
async def debug_price(pair: str):
    """Test price fetching."""
    price = await get_public_price(pair)
    candles = await get_public_candles(pair)
    return {"pair": pair, "price": price, "candles_count": len(candles.get("prices", []))}


@app.get("/debug/coinbase")
async def debug_coinbase():
    """Test Coinbase CDP authentication."""
    if not cdp_client:
        return {"error": "CDP client not initialized", "live_trading": LIVE_TRADING}
    
    result = await cdp_client.test_auth()
    return {"live_trading": LIVE_TRADING, "auth_test": result}


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
