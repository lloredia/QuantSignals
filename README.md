# 🚀 QuantSignals Ultra

**Elite Quantitative Crypto Trading Bot for Telegram**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://core.telegram.org/bots)
[![Coinbase](https://img.shields.io/badge/Coinbase-CDP-blue.svg)](https://docs.cdp.coinbase.com/)

---

## ✨ Features

### Trading
- **15 Trading Pairs**: BTC, ETH, SOL, AVAX, LINK, DOGE, XRP, ADA, MATIC, DOT, ATOM, UNI, AAVE, LTC, NEAR
- **Quick Buy/Sell**: One-tap trading with confirmation
- **Limit Orders**: Price alerts that notify when targets hit
- **Live Coinbase Integration**: Real trades via CDP API

### AI Signal Engine
- **7 Quantitative Strategies**: Momentum, Mean Reversion, Volatility, Trend, Liquidity, Sentiment, Correlation
- **Market Regime Detection**: Bull/Bear × High/Low Volatility
- **Self-Learning Weights**: Strategies adapt based on performance
- **Kelly Criterion Sizing**: Optimal position sizing

### Autopilot Mode
- **Fully Autonomous Trading**: Set it and forget it
- **Configurable Parameters**: Trade size, confidence threshold, max trades
- **Automatic Stop Loss & Trailing Stop**

### Risk Management
- **Max Drawdown Protection**: Auto-pause if portfolio drops X%
- **Daily Loss Limits**: Stop trading after hitting loss threshold
- **Take Profit Tiers**: Partial sells at +5%, +10%, +20%
- **DCA Autopilot**: Auto-buy dips

### Analytics
- **P&L Charts**: Visual performance tracking
- **Win Streak Tracking**: Current and best streaks
- **Daily Summaries**: Automatic 9 PM reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Telegram Bot Token
- Coinbase CDP API Keys
- Redis (optional)
- Anthropic API Key

### Environment Variables

```env
TELEGRAM_BOT_TOKEN=your_bot_token
WEBHOOK_SECRET=random_secret_string
BASE_URL=https://your-app.railway.app
CDP_API_KEY_NAME=your_cdp_key_name
CDP_API_KEY_PRIVATE=your_cdp_private_key
ANTHROPIC_API_KEY=your_anthropic_key
REDIS_URL=redis://...
AUTO_SIGNAL_CHATS=123456789
LIVE_TRADING=false
TRADE_AMOUNT_USD=25
STOP_LOSS_PCT=5
TAKE_PROFIT_PCT=10
```

### Deploy to Railway

1. Connect GitHub repo
2. Add environment variables
3. Deploy!

---

## 📱 Commands

| Command | Description |
|---------|-------------|
| `/menu` | Interactive button menu |
| `/buy BTC 50` | Buy $50 of BTC |
| `/sell ETH` | Sell all ETH |
| `/portfolio` | All holdings + P&L |
| `/signals` | AI trading signals |
| `/autopilot` | Autopilot control |
| `/risk` | Risk dashboard |
| `/performance` | Full report |

---

## 🛡️ Risk Management

- **Max Drawdown**: Auto-pause at X% loss
- **Daily Loss Limit**: Stop after $X loss
- **TP Tiers**: Sell 25% at +5%, +10%, +20%
- **DCA Auto**: Buy dips automatically

---

## ⚠️ Disclaimer

Educational purposes only. Crypto trading involves risk. Only trade what you can afford to lose.

---

**Built by LESLEADS Consulting**
