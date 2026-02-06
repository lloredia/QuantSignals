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
import hmac
import hashlib
import time
import base64

# ============ CONFIG ============
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "quantsignals-secret")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Coinbase API
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

# Trading Config
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", "10"))  # Start small
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "5"))  # 5% stop loss
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "10"))  # 10% take profit

# Supported coins for day trading (high volume, good volatility)
TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]

app = FastAPI()
tg_app = Application.builder().token(TOKEN).build()

# In-memory position tracking (use Redis/DB for production)
positions = {}
daily_pnl = {"realized": 0.0, "trades": 0, "wins": 0}


# ============ COINBASE CLIENT ============
class CoinbaseClient:
    """Coinbase Advanced Trade API client."""
    
    BASE_URL = "https://api.coinbase.com"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Create signature for Coinbase API."""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _headers(self, method: str, path: str, body: str = "") -> dict:
        """Generate auth headers."""
        timestamp = str(int(time.time()))
        signature = self._sign(timestamp, method, path, body)
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
    
    async def get_accounts(self) -> dict:
        """Get all accounts/balances."""
        path = "/api/v3/brokerage/accounts"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE_URL}{path}",
                headers=self._headers("GET", path),
                timeout=10
            )
            return resp.json()
    
    async def get_price(self, product_id: str) -> float:
        """Get current price for a trading pair."""
        path = f"/api/v3/brokerage/products/{product_id}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE_URL}{path}",
                headers=self._headers("GET", path),
                timeout=10
            )
            data = resp.json()
            return float(data.get("price", 0))
    
    async def get_candles(self, product_id: str, granularity: str = "ONE_HOUR", limit: int = 24) -> list:
        """Get historical candles for analysis."""
        path = f"/api/v3/brokerage/products/{product_id}/candles"
        end = int(time.time())
        start = end - (limit * 3600)  # Last N hours
        
        params = f"?start={start}&end={end}&granularity={granularity}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE_URL}{path}{params}",
                headers=self._headers("GET", path + params),
                timeout=10
            )
            return resp.json().get("candles", [])
    
    async def place_market_order(self, product_id: str, side: str, usd_amount: float) -> dict:
        """Place a market order."""
        path = "/api/v3/brokerage/orders"
        
        order_config = {
            "market_market_ioc": {
                "quote_size": str(usd_amount)
            }
        } if side == "BUY" else {
            "market_market_ioc": {
                "base_size": str(usd_amount)  # For sells, need base currency amount
            }
        }
        
        body = json.dumps({
            "client_order_id": f"tc_{int(time.time())}_{product_id}",
            "product_id": product_id,
            "side": side,
            "order_configuration": order_config
        })
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE_URL}{path}",
                headers=self._headers("POST", path, body),
                content=body,
                timeout=10
            )
            return resp.json()


# Initialize Coinbase client
coinbase = CoinbaseClient(COINBASE_API_KEY or "", COINBASE_API_SECRET or "")


# ============ AI SIGNAL GENERATOR ============
async def generate_trading_signals() -> dict:
    """Generate AI trading signals for crypto pairs."""
    
    if not OPENAI_API_KEY:
        return {"error": "OpenAI not configured"}
    
    # Gather market data
    market_data = {}
    for pair in TRADING_PAIRS:
        try:
            price = await coinbase.get_price(pair)
            candles = await coinbase.get_candles(pair, "ONE_HOUR", 24)
            
            if candles:
                prices = [float(c["close"]) for c in candles[:24]]
                high_24h = max(float(c["high"]) for c in candles[:24])
                low_24h = min(float(c["low"]) for c in candles[:24])
                volume_24h = sum(float(c["volume"]) for c in candles[:24])
                
                # Calculate basic indicators
                sma_8 = sum(prices[:8]) / 8 if len(prices) >= 8 else price
                sma_21 = sum(prices[:21]) / 21 if len(prices) >= 21 else price
                
                # Price change
                if len(prices) >= 2:
                    change_1h = ((price - prices[1]) / prices[1]) * 100
                else:
                    change_1h = 0
                
                market_data[pair] = {
                    "price": price,
                    "high_24h": high_24h,
                    "low_24h": low_24h,
                    "volume_24h": volume_24h,
                    "sma_8": sma_8,
                    "sma_21": sma_21,
                    "change_1h": change_1h,
                    "trend": "bullish" if sma_8 > sma_21 else "bearish"
                }
        except Exception as e:
            print(f"[ERROR] Failed to get data for {pair}: {e}")
    
    if not market_data:
        return {"error": "No market data available"}
    
    # Build prompt
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    prompt = f"""You are an expert crypto day trader AI. Analyze the following market data and provide trading signals.

CURRENT TIME: {now.strftime("%Y-%m-%d %H:%M %Z")}
TRADING STYLE: Day trading (holding 1-8 hours)
RISK TOLERANCE: Medium (5% stop loss, 10% take profit)

MARKET DATA:
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
"""
    
    prompt += """

INSTRUCTIONS:
1. Analyze each pair for day trading opportunities
2. Consider momentum, trend, support/resistance levels
3. Only recommend HIGH confidence trades (>70% conviction)
4. Maximum 2 trade signals

OUTPUT FORMAT (JSON only, no other text):
{
    "signals": [
        {
            "pair": "BTC-USD",
            "action": "BUY" or "SELL" or "HOLD",
            "confidence": 75,
            "entry_price": 50000,
            "stop_loss": 47500,
            "take_profit": 55000,
            "reasoning": "Brief reason"
        }
    ],
    "market_sentiment": "bullish" or "bearish" or "neutral",
    "summary": "One sentence market summary"
}
"""
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional crypto trader. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        result_text = response.choices[0].message.content
        
        # Clean up response
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        signals = json.loads(result_text)
        signals["market_data"] = market_data
        signals["generated_at"] = now.isoformat()
        
        return signals
        
    except Exception as e:
        print(f"[ERROR] AI signal generation failed: {e}")
        return {"error": str(e)}


# ============ TRADING FUNCTIONS ============
async def execute_trade(pair: str, action: str, amount_usd: float) -> dict:
    """Execute a trade on Coinbase."""
    
    if not COINBASE_API_KEY or not COINBASE_API_SECRET:
        return {"success": False, "error": "Coinbase not configured"}
    
    try:
        result = await coinbase.place_market_order(pair, action, amount_usd)
        
        if result.get("success"):
            # Track position
            if action == "BUY":
                positions[pair] = {
                    "entry_price": await coinbase.get_price(pair),
                    "amount_usd": amount_usd,
                    "timestamp": datetime.now().isoformat()
                }
            elif action == "SELL" and pair in positions:
                # Calculate P&L
                entry = positions[pair]["entry_price"]
                exit_price = await coinbase.get_price(pair)
                pnl = ((exit_price - entry) / entry) * amount_usd
                daily_pnl["realized"] += pnl
                daily_pnl["trades"] += 1
                if pnl > 0:
                    daily_pnl["wins"] += 1
                del positions[pair]
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ TELEGRAM COMMANDS ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = """🤖 <b>TRADECIRCLE BOT</b>

AI-powered crypto day trading assistant.

<b>Commands:</b>
/signals - Get AI trading signals
/portfolio - View current positions
/balance - Check account balance
/pnl - Today's P&L
/trade - Execute a trade
/settings - Bot settings
/help - Show help

<i>⚠️ Trading involves risk. Never trade more than you can afford to lose.</i>"""
    
    await update.message.reply_text(welcome, parse_mode="HTML")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """🤖 <b>TRADECIRCLE HELP</b>

<b>Signal Commands:</b>
/signals - Generate AI trading signals
/market - Quick market overview

<b>Trading Commands:</b>
/trade [pair] [buy/sell] - Execute trade
/portfolio - View open positions
/close [pair] - Close a position

<b>Account Commands:</b>
/balance - Check balances
/pnl - Today's profit/loss
/history - Recent trades

<b>Settings:</b>
/settings - View/change settings
/risk - Adjust risk parameters

<i>Example: /trade BTC-USD buy</i>"""
    
    await update.message.reply_text(help_text, parse_mode="HTML")


async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and display AI trading signals."""
    msg = await update.message.reply_text("🔄 Analyzing markets...")
    
    signals = await generate_trading_signals()
    
    if "error" in signals:
        await msg.edit_text(f"❌ Error: {signals['error']}")
        return
    
    tz = pytz.timezone("America/Chicago")
    now = datetime.now(tz)
    
    text = f"""🤖 <b>TRADECIRCLE SIGNALS</b>
📅 {now.strftime("%B %d, %Y %I:%M %p %Z")}
━━━━━━━━━━━━━━━

<b>Market Sentiment:</b> {signals.get('market_sentiment', 'N/A').upper()}

"""
    
    for signal in signals.get("signals", []):
        action_emoji = "🟢" if signal["action"] == "BUY" else "🔴" if signal["action"] == "SELL" else "⚪"
        
        text += f"""{action_emoji} <b>{signal['pair']}</b>
Action: <b>{signal['action']}</b>
Confidence: {signal['confidence']}%
Entry: ${signal['entry_price']:,.2f}
Stop Loss: ${signal['stop_loss']:,.2f}
Take Profit: ${signal['take_profit']:,.2f}
📝 {signal['reasoning']}

"""
    
    text += f"""━━━━━━━━━━━━━━━
<b>Summary:</b> {signals.get('summary', 'N/A')}

<i>⚠️ Not financial advice. Trade at your own risk.</i>"""
    
    # Add trade buttons for BUY signals
    keyboard = []
    for signal in signals.get("signals", []):
        if signal["action"] == "BUY":
            keyboard.append([
                InlineKeyboardButton(
                    f"🟢 Buy {signal['pair'].split('-')[0]} (${TRADE_AMOUNT_USD})",
                    callback_data=f"trade_buy_{signal['pair']}"
                )
            ])
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    
    await msg.edit_text(text, parse_mode="HTML", reply_markup=reply_markup)


async def balance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check account balances."""
    msg = await update.message.reply_text("💰 Fetching balances...")
    
    if not COINBASE_API_KEY:
        await msg.edit_text("❌ Coinbase not configured. Add API keys.")
        return
    
    try:
        accounts = await coinbase.get_accounts()
        
        text = """💰 <b>ACCOUNT BALANCES</b>
━━━━━━━━━━━━━━━

"""
        
        total_usd = 0
        for account in accounts.get("accounts", []):
            balance = float(account.get("available_balance", {}).get("value", 0))
            currency = account.get("currency", "")
            
            if balance > 0.01:  # Only show non-dust balances
                if currency == "USD":
                    text += f"💵 USD: ${balance:,.2f}\n"
                    total_usd += balance
                else:
                    # Get USD value
                    try:
                        price = await coinbase.get_price(f"{currency}-USD")
                        usd_value = balance * price
                        total_usd += usd_value
                        text += f"🪙 {currency}: {balance:.6f} (${usd_value:,.2f})\n"
                    except:
                        text += f"🪙 {currency}: {balance:.6f}\n"
        
        text += f"""
━━━━━━━━━━━━━━━
<b>Total Value:</b> ${total_usd:,.2f}"""
        
        await msg.edit_text(text, parse_mode="HTML")
        
    except Exception as e:
        await msg.edit_text(f"❌ Error fetching balances: {str(e)[:100]}")


async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current open positions."""
    
    if not positions:
        await update.message.reply_text("📭 No open positions.\n\nUse /signals to get trading ideas.")
        return
    
    text = """📊 <b>OPEN POSITIONS</b>
━━━━━━━━━━━━━━━

"""
    
    for pair, pos in positions.items():
        current_price = await coinbase.get_price(pair)
        entry = pos["entry_price"]
        pnl_pct = ((current_price - entry) / entry) * 100
        pnl_usd = (pnl_pct / 100) * pos["amount_usd"]
        
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        
        text += f"""{emoji} <b>{pair}</b>
Entry: ${entry:,.2f}
Current: ${current_price:,.2f}
P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})

"""
    
    text += "━━━━━━━━━━━━━━━"
    
    # Add close buttons
    keyboard = [[InlineKeyboardButton(f"Close {pair}", callback_data=f"close_{pair}")] for pair in positions]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
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


async def market_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick market overview."""
    msg = await update.message.reply_text("📊 Loading market data...")
    
    text = """📊 <b>MARKET OVERVIEW</b>
━━━━━━━━━━━━━━━

"""
    
    for pair in TRADING_PAIRS:
        try:
            price = await coinbase.get_price(pair)
            candles = await coinbase.get_candles(pair, "ONE_HOUR", 2)
            
            if candles and len(candles) >= 2:
                prev_price = float(candles[1]["close"])
                change = ((price - prev_price) / prev_price) * 100
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            else:
                change = 0
                emoji = "⚪"
            
            coin = pair.split("-")[0]
            text += f"{emoji} <b>{coin}</b>: ${price:,.2f} ({change:+.2f}%)\n"
            
        except Exception as e:
            text += f"⚠️ {pair}: Error\n"
    
    text += """
━━━━━━━━━━━━━━━
<i>Use /signals for AI analysis</i>"""
    
    await msg.edit_text(text, parse_mode="HTML")


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current settings."""
    
    text = f"""⚙️ <b>BOT SETTINGS</b>
━━━━━━━━━━━━━━━

💵 Trade Amount: ${TRADE_AMOUNT_USD}
📊 Max Positions: {MAX_POSITIONS}
🛑 Stop Loss: {STOP_LOSS_PCT}%
🎯 Take Profit: {TAKE_PROFIT_PCT}%

<b>Trading Pairs:</b>
{', '.join(TRADING_PAIRS)}

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
        await query.edit_message_text(f"🔄 Executing BUY for {pair}...")
        
        result = await execute_trade(pair, "BUY", TRADE_AMOUNT_USD)
        
        if result.get("success"):
            await query.edit_message_text(
                f"✅ <b>ORDER EXECUTED</b>\n\n"
                f"Bought ${TRADE_AMOUNT_USD} of {pair}\n\n"
                f"Use /portfolio to track position.",
                parse_mode="HTML"
            )
        else:
            await query.edit_message_text(
                f"❌ Order failed: {result.get('error', 'Unknown error')}"
            )
    
    elif data.startswith("close_"):
        pair = data.replace("close_", "")
        
        if pair in positions:
            pos = positions[pair]
            result = await execute_trade(pair, "SELL", pos["amount_usd"])
            
            if result.get("success"):
                await query.edit_message_text(f"✅ Closed {pair} position.")
            else:
                await query.edit_message_text(f"❌ Failed to close: {result.get('error')}")
        else:
            await query.edit_message_text("❌ Position not found.")


# ============ REGISTER HANDLERS ============
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(CommandHandler("signals", signals_cmd))
tg_app.add_handler(CommandHandler("balance", balance_cmd))
tg_app.add_handler(CommandHandler("portfolio", portfolio_cmd))
tg_app.add_handler(CommandHandler("pnl", pnl_cmd))
tg_app.add_handler(CommandHandler("market", market_cmd))
tg_app.add_handler(CommandHandler("settings", settings_cmd))
tg_app.add_handler(CallbackQueryHandler(button_callback))


# ============ FASTAPI ROUTES ============
@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    await tg_app.start()
    if BASE_URL:
        webhook_url = f"{BASE_URL}/webhook/{WEBHOOK_SECRET}"
        await tg_app.bot.set_webhook(url=webhook_url)
        print(f"✅ Webhook set: {webhook_url}")


@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()


@app.get("/")
async def health():
    return {"status": "ok", "bot": "QuantSignals", "positions": len(positions)}


@app.get("/debug/signals")
async def debug_signals():
    """Test endpoint to check signal generation."""
    signals = await generate_trading_signals()
    return signals


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
