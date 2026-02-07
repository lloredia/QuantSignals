# 🚀 QuantSignals Ultra

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?logo=telegram&logoColor=white)
![Coinbase](https://img.shields.io/badge/Coinbase-CDP-0052FF?logo=coinbase&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Persistence-DC382D?logo=redis&logoColor=white)
![Railway](https://img.shields.io/badge/Railway-Deployed-0B0D0E?logo=railway&logoColor=white)

**Elite Quantitative Crypto Trading Intelligence**

*AI-Powered Signals • Autonomous Trading • Risk Management*

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-quick-start) • [Commands](#-commands) • [API](#-api-endpoints)

</div>

---

## 📋 Overview

QuantSignals Ultra is a professional-grade cryptocurrency trading bot that combines AI-powered signal generation with autonomous execution capabilities. Built for serious traders who want institutional-quality tools.

### Core Philosophy
```
Capital Preservation → Consistency → Compounding
```

---

## ✨ Features

### 🤖 AI Trading Engine
- **7 Quantitative Strategies** - Momentum, mean reversion, volatility expansion, trend following, liquidity sweep, sentiment anomaly, cross-asset correlation
- **Market Regime Detection** - Automatically adapts to bull/bear, high/low volatility conditions
- **Kelly Criterion Sizing** - Mathematically optimal position sizing based on win rate and edge
- **Self-Learning Weights** - Strategy weights adjust based on real performance

### 📊 Signal Generation
- **15 Trading Pairs** - BTC, ETH, SOL, AVAX, LINK, DOGE, XRP, ADA, MATIC, DOT, ATOM, UNI, AAVE, LTC, NEAR
- **Multi-Timeframe Analysis** - 1H, 4H, 1D confluence scoring
- **Technical Indicators** - SMA, RSI, Bollinger Bands, ATR, volume analysis
- **Sentiment Integration** - Fear & Greed Index, crypto news, whale alerts

### 🛡️ Risk Management
- **Max Drawdown Protection** - Auto-pause trading at configurable drawdown %
- **Daily Loss Limits** - Stop trading after daily loss threshold
- **Position Concentration** - Prevent overexposure to single assets
- **Trailing Stop Loss** - Dynamic stops that lock in profits
- **Take Profit Tiers** - Partial profit taking at multiple levels

### 🎯 Autonomous Trading
- **Autopilot Mode** - Fully autonomous signal execution
- **DCA Autopilot** - Automatically buy dips at configurable thresholds
- **Paper Trading** - Test strategies without real money
- **Live Execution** - Direct Coinbase integration via CDP API

### 📈 Analytics & Reporting
- **Real-time Portfolio** - View all Coinbase holdings with P&L
- **Performance Charts** - ASCII P&L visualization
- **Trade History** - Complete audit trail with CSV export
- **Win Streak Tracking** - Monitor hot and cold streaks
- **Daily Summaries** - Automated 9 PM reports

---

## 🏗️ Architecture

### System Overview

```mermaid
flowchart TB
    subgraph Interface["🔌 Interface Layer"]
        TG[📱 Telegram Bot]
        API[🌐 FastAPI Server]
        REDIS[(🗄️ Redis)]
    end

    subgraph Core["⚙️ Core Engine"]
        SIG[📊 Signal Generator]
        RISK[🛡️ Risk Manager]
        POS[💼 Position Tracker]
        STRAT[🧠 Strategy Engine]
        REG[📈 Regime Detector]
    end

    subgraph Tasks["🔄 Background Tasks"]
        SL[Stop Loss Monitor]
        AP[Autopilot Scanner]
        DCA[DCA Autopilot]
        RM[Risk Monitor]
        TP[TP Tiers Monitor]
        DS[Daily Summary]
        PA[Price Alerts]
        SS[Signal Scheduler]
    end

    subgraph External["🌐 External APIs"]
        CB[Coinbase CDP]
        CG[CoinGecko]
        CC[CryptoCompare]
        EX[Coinbase Exchange]
    end

    TG <--> API
    API <--> REDIS
    
    API --> Core
    Core --> Tasks
    
    Tasks --> External
    Core --> External

    style Interface fill:#1a1a2e,stroke:#16213e,color:#fff
    style Core fill:#0f3460,stroke:#16213e,color:#fff
    style Tasks fill:#533483,stroke:#16213e,color:#fff
    style External fill:#e94560,stroke:#16213e,color:#fff
```

### Data Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant T as 📱 Telegram
    participant B as 🤖 Bot
    participant E as ⚙️ Engine
    participant C as 💰 Coinbase
    participant R as 🗄️ Redis

    U->>T: /signals
    T->>B: Command
    B->>E: Generate Signals
    E->>C: Get Market Data
    C-->>E: Prices, Volume
    E->>E: Run 7 Strategies
    E->>E: Detect Regime
    E->>E: Calculate Confidence
    E-->>B: Trading Signals
    B->>R: Save Signal History
    B-->>T: Display Signals
    T-->>U: 📊 AI Signals

    U->>T: Click "Buy BTC"
    T->>B: Callback
    B->>C: Place Order
    C-->>B: Order Confirmed
    B->>R: Save Position
    B-->>T: ✅ Executed
    T-->>U: Confirmation
```

### Autopilot Flow

```mermaid
flowchart LR
    subgraph Scanner["🔄 Every 60s"]
        A[Check Enabled] --> B{Autopilot ON?}
        B -->|No| A
        B -->|Yes| C[Get Signals]
    end

    subgraph Filter["🎯 Signal Filter"]
        C --> D{Confidence >= 75%?}
        D -->|No| A
        D -->|Yes| E{EV >= 1.5%?}
        E -->|No| A
        E -->|Yes| F{Max Trades?}
        F -->|Hit| A
        F -->|OK| G[Calculate Size]
    end

    subgraph Execute["💰 Execution"]
        G --> H[Kelly Sizing]
        H --> I{Live Mode?}
        I -->|Paper| J[Save Position]
        I -->|Live| K[Coinbase Order]
        K --> L{Success?}
        L -->|Yes| J
        L -->|No| M[Log Error]
        J --> N[Send Alert]
    end

    style Scanner fill:#1a1a2e,stroke:#16213e,color:#fff
    style Filter fill:#0f3460,stroke:#16213e,color:#fff
    style Execute fill:#533483,stroke:#16213e,color:#fff
```

### Risk Management

```mermaid
flowchart TD
    subgraph Monitor["🛡️ Risk Monitor - Every 2min"]
        A[Get Portfolio Value] --> B[Calculate Drawdown]
        B --> C{Drawdown > Max?}
        C -->|Yes| D[🚨 Pause Autopilot]
        C -->|No| E{Daily Loss > Limit?}
        E -->|Yes| D
        E -->|No| F[✅ Continue Trading]
        D --> G[Send Alert]
    end

    subgraph Limits["⚙️ Configurable Limits"]
        H[Max Drawdown %]
        I[Daily Loss $]
        J[Max Position %]
        K[Take Profit Tiers]
    end

    Limits --> Monitor

    style Monitor fill:#e94560,stroke:#16213e,color:#fff
    style Limits fill:#0f3460,stroke:#16213e,color:#fff
```

### Strategy Engine

```mermaid
mindmap
  root((🧠 Strategy Engine))
    Momentum
      Breakout Detection
      Volume Confirmation
      Resistance Levels
    Mean Reversion
      RSI Oversold
      Bollinger Bands
      Support Bounce
    Trend Following
      SMA Crossover
      EMA Alignment
      ADX Strength
    Volatility
      Band Expansion
      ATR Breakout
      Squeeze Release
    Sentiment
      Fear & Greed
      News Analysis
      Whale Alerts
    Correlation
      BTC Dominance
      ETH Ratio
      Altcoin Beta
    Liquidity
      Stop Hunts
      Order Book
      Volume Profile
```

### State Management

```mermaid
erDiagram
    POSITIONS ||--o{ TRADES : generates
    POSITIONS {
        string pair
        float entry_price
        float highest_price
        float amount_usd
        datetime timestamp
        boolean live
        string strategy
    }
    
    TRADES {
        string pair
        float entry
        float exit
        float pnl_pct
        float pnl_usd
        string strategy
        datetime closed_at
    }
    
    AUTOPILOT ||--o{ POSITIONS : creates
    AUTOPILOT {
        boolean enabled
        int trade_percentage
        int min_confidence
        int max_daily_trades
        int trades_today
        float total_profit
    }
    
    RISK_STATE ||--|| AUTOPILOT : controls
    RISK_STATE {
        float max_drawdown_pct
        float daily_loss_limit
        float max_position_pct
        float highest_balance
        float current_drawdown
        string paused_reason
    }
```

### Component Breakdown

#### 🔌 Interface Layer
| Component | Technology | Purpose |
|-----------|------------|---------|
| Telegram Bot | python-telegram-bot | User interface, commands, alerts |
| FastAPI Server | FastAPI + Uvicorn | Webhooks, health checks, debug endpoints |
| Redis Client | redis-py | Persistent state storage |

#### ⚙️ Core Engine
| Module | Description |
|--------|-------------|
| Signal Generator | AI-powered trading signal generation with confidence scoring |
| Risk Manager | Drawdown protection, position limits, loss limits |
| Position Tracker | Open positions, P&L calculation, trade history |
| Strategy Engine | 7 quantitative strategies with adaptive weights |
| Regime Detector | Market condition classification |

#### 🔄 Background Tasks
| Task | Interval | Function |
|------|----------|----------|
| Stop Loss Monitor | 60s | Monitor positions for SL/TP/trailing stop |
| Autopilot Scanner | 60s | Execute high-confidence signals |
| DCA Autopilot | 5m | Buy dips automatically |
| Risk Monitor | 2m | Check drawdown and loss limits |
| TP Tiers Monitor | 60s | Execute partial profit taking |
| Daily Summary | 9 PM | Send daily performance report |
| Price Alert Checker | 60s | Check price alerts |
| Signal Scheduler | Hourly | Send scheduled signals |

#### 🌐 External Integrations
| Service | Purpose |
|---------|---------|
| Coinbase CDP | Trading execution, account balances |
| Coinbase Exchange | Market data, order books |
| CoinGecko | Fear & Greed Index |
| CryptoCompare | News feed |

---

## 📁 Project Structure

```
QuantSignals/
├── main.py              # Main application (4000+ lines)
│   ├── Configuration    # Environment variables, constants
│   ├── State Management # Redis persistence, in-memory state
│   ├── CDP Client       # Coinbase integration class
│   ├── Market Data      # Price feeds, technical analysis
│   ├── Signal Engine    # AI signal generation
│   ├── Risk Management  # Drawdown, limits, position sizing
│   ├── Commands         # 30+ Telegram command handlers
│   ├── Callbacks        # Button/menu handlers
│   ├── Background Tasks # 8 async monitors and scanners
│   └── FastAPI App      # Web server, webhooks
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration
├── railway.json         # Railway deployment config
├── Procfile             # Process configuration
├── tests.py             # Unit tests
└── README.md            # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Telegram Bot Token (from @BotFather)
- Coinbase CDP API Keys (optional for live trading)
- Redis instance (Railway provides free tier)

### Environment Variables

```bash
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
WEBHOOK_SECRET=random_secure_string

# Coinbase (for live trading)
CDP_API_KEY_NAME=your_cdp_key_name
CDP_API_KEY_SECRET=your_cdp_private_key

# Redis (for persistence)
REDIS_URL=redis://default:password@host:port

# Optional
BASE_URL=https://your-app.railway.app
AUTO_SIGNAL_CHATS=123456789,987654321
LIVE_TRADING=false
TRADE_AMOUNT_USD=25
STOP_LOSS_PCT=5
TAKE_PROFIT_PCT=10
TRAILING_STOP_PCT=3
```

### Local Development

```bash
# Clone repository
git clone https://github.com/lloredia/QuantSignals.git
cd QuantSignals

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TELEGRAM_BOT_TOKEN=your_token
export WEBHOOK_SECRET=your_secret

# Run locally (polling mode)
python main.py
```

### Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Set environment variables in Railway dashboard
```

---

## 📱 Commands

### 💰 Trading
| Command | Usage | Description |
|---------|-------|-------------|
| `/buy` | `/buy BTC 50` | Buy $50 of BTC |
| `/sell` | `/sell ETH 50` | Sell 50% of ETH |
| `/limit` | `/limit BTC 95000` | Alert when BTC hits $95k |
| `/signals` | `/signals` | Get AI trading signals |

### 📊 Portfolio
| Command | Description |
|---------|-------------|
| `/portfolio` | View all holdings with P&L |
| `/pnl` | Today's realized P&L |
| `/history` | Trade history with export |
| `/performance` | Full performance report |

### 🤖 Autopilot
| Command | Description |
|---------|-------------|
| `/autopilot` | Control center for autonomous trading |
| `/dcaauto` | Configure DCA autopilot |
| `/tptiers` | Set take profit tiers |
| `/pause` | Pause all trading |

### 🛡️ Risk
| Command | Description |
|---------|-------------|
| `/risk` | Risk dashboard |
| `/watchlist` | Manage coin watchlist |

### 📈 Analysis
| Command | Description |
|---------|-------------|
| `/regime` | Market regime detection |
| `/fear` | Fear & Greed Index |
| `/news` | Latest crypto news |
| `/tf BTC` | Multi-timeframe analysis |
| `/whale` | Whale alerts |

### 📊 Analytics
| Command | Description |
|---------|-------------|
| `/chart` | P&L chart |
| `/streak` | Win/loss streak |
| `/besttrades` | Best & worst trades |
| `/summary` | Daily summary |

### ⚙️ Tools
| Command | Description |
|---------|-------------|
| `/menu` | Interactive button menu |
| `/market` | Live prices |
| `/alerts` | View all alerts |
| `/backtest BTC` | Backtest strategy |
| `/settings` | Bot settings |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/debug/signals` | GET | Generate and return signals |
| `/test/auto-signal` | GET | Send test signal to chats |
| `/webhook/{secret}` | POST | Telegram webhook receiver |

---

## ⚙️ Configuration

### Trading Parameters
```python
TRADE_AMOUNT_USD = 25      # Default trade size
STOP_LOSS_PCT = 5          # Stop loss percentage
TAKE_PROFIT_PCT = 10       # Take profit percentage
TRAILING_STOP_PCT = 3      # Trailing stop percentage
```

### Risk Parameters
```python
MAX_POSITION_PCT = 25      # Max % per position
MAX_PORTFOLIO_RISK = 60    # Max % at risk
MAX_DAILY_DRAWDOWN = 10    # Auto-pause threshold
```

### Autopilot Defaults
```python
trade_percentage = 20      # % of balance per trade
min_confidence = 75        # Minimum signal confidence
max_daily_trades = 10      # Maximum trades per day
min_trade_usd = 10         # Minimum trade size
max_trade_usd = 100        # Maximum trade size
```

---

## 📊 Strategy Details

### 1. Momentum Breakout
Identifies price breaking above resistance with volume confirmation.

### 2. Mean Reversion
Buys oversold conditions (RSI < 30) expecting bounce.

### 3. Volatility Expansion
Trades Bollinger Band breakouts during volatility expansion.

### 4. Trend Following
Rides established trends using SMA crossovers.

### 5. Liquidity Sweep
Detects stop hunts and liquidity grabs for reversal entries.

### 6. Sentiment Anomaly
Trades extreme Fear & Greed readings as contrarian signals.

### 7. Cross-Asset Correlation
Uses BTC/ETH divergence for altcoin opportunities.

---

## 🔐 Security

- API keys stored as environment variables
- Webhook secret validation
- No keys in code or logs
- Redis password protected
- Rate limiting on API calls

---

## 🧪 Testing

```bash
# Run tests
python -m pytest tests.py -v

# Test specific module
python -m pytest tests.py::test_signal_generation -v
```

---

## 📈 Performance

Typical resource usage on Railway:
- **Memory**: ~150MB
- **CPU**: Minimal (async I/O)
- **Redis**: < 1MB storage

---

## 🗺️ Roadmap

- [ ] Web dashboard with real-time charts
- [ ] Mobile app (React Native)
- [ ] Multi-exchange support (Binance, Kraken)
- [ ] Options flow tracking
- [ ] Social copy trading
- [ ] Premium subscription tier

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

**This software is for educational purposes only. Cryptocurrency trading carries significant risk. Past performance does not guarantee future results. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses.**

---

## 🙏 Acknowledgments

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [Coinbase CDP SDK](https://docs.cdp.coinbase.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Redis](https://redis.io/)

---

<div align="center">

**Built with ❤️ by lloredia**

[GitHub](https://github.com/lloredia) • [Twitter](https://twitter.com/lloredia)

</div>
