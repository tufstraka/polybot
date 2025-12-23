# 🤖 Polybot - Spike Hunter

**Automated trading bot for Polymarket that detects and trades price spikes.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What It Does

Polybot watches Polymarket prediction markets and looks for **sudden price movements** (spikes). When news breaks, prices often overreact for a few seconds before calming down. This bot catches those moments and trades the bounce-back.

**The Strategy (Mean Reversion):**
1. Price spikes UP → Bot SELLS (expecting price to drop back)
2. Price spikes DOWN → Bot BUYS (expecting price to rise back)
3. Small positions ($1-2) with tight exits (2-4% profit target)

---

## Features

✅ **4-Layer Spike Detection**
- CUSUM algorithm (Bell Labs) for regime changes
- EWMA volatility bands (J.P. Morgan RiskMetrics)
- ROC momentum confirmation
- Liquidity validation

✅ **Risk Management**
- Daily loss limit ($2 default)
- Position sizing based on confidence
- Circuit breaker after losing streaks
- Capital protection

✅ **Paper Trading Mode**
- Test without real money
- Full simulation of trades
- Track what would have happened

✅ **Real-time Dashboard**
- See tracked markets
- View detected signals
- Monitor P&L and positions
- Check risk status

✅ **Notifications**
- Telegram alerts
- Discord webhooks
- Trade confirmations
- Daily summaries

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd polybot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example config
cp .env.example .env

# Edit with your credentials
nano .env
```

Add your Polymarket API credentials:
```env
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_PASSPHRASE=your_passphrase
```

### 3. Run (Paper Trading)

```bash
# Start bot in dry-run mode (no real trades)
python -m src.bot --dry-run
```

### 4. View Dashboard

```bash
# In another terminal
streamlit run src/dashboard/app.py
```

Open http://localhost:8501

---

## Configuration

### Main Config (`config/config.yaml`)

```yaml
# Trading settings
trading:
  dry_run: true              # Paper trading mode

# Spike detection sensitivity
detection:
  cusum_threshold: 2.5       # Lower = more sensitive
  ewma_band_width: 2.0       # Volatility band width
  roc_threshold: 0.02        # Minimum momentum (2%)
  min_volume: 50000          # Minimum 24h volume

# Risk limits
risk:
  max_daily_loss: 2.0        # Stop after $2 loss
  max_position_size: 2.0     # Max per trade
  min_position_size: 0.5     # Min per trade
  max_consecutive_losses: 3  # Circuit breaker trigger

# Your capital
money:
  initial_capital: 75.0      # Starting balance
```

### Environment Variables (`.env`)

```env
# Required - Polymarket API
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_api_secret
POLYMARKET_PASSPHRASE=your_passphrase

# Optional - Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=987654321

# Optional - Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## Project Structure

```
polybot/
├── src/
│   ├── bot.py              # Main orchestrator
│   ├── config/
│   │   └── settings.py     # Configuration management
│   ├── core/
│   │   ├── client.py       # Polymarket API wrapper
│   │   ├── scanner.py      # Market discovery
│   │   ├── tracker.py      # Price tracking & indicators
│   │   ├── detector.py     # Spike detection algorithm
│   │   ├── executor.py     # Order execution
│   │   ├── position_manager.py  # Position tracking
│   │   └── state_writer.py # Dashboard communication
│   ├── risk/
│   │   └── risk_manager.py # Risk management
│   ├── notifications/
│   │   ├── telegram.py     # Telegram alerts
│   │   └── discord.py      # Discord webhooks
│   └── dashboard/
│       └── app.py          # Streamlit dashboard
├── config/
│   └── config.yaml         # Main configuration
├── tests/                  # Unit tests
├── docs/
│   └── DEPLOYMENT.md       # Deployment guide
├── plans/
│   ├── ARCHITECTURE.md     # System design
│   └── SPIKE_DETECTION.md  # Algorithm details
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## How the Detection Works

### The 4-Layer System

```
Layer 1: CUSUM
│   Detects regime changes (stable → volatile)
│   Like a "change detector" for price behavior
│
└── Layer 2: EWMA Bands
    │   Dynamic support/resistance levels
    │   Adapts to current volatility
    │
    └── Layer 3: ROC Momentum
        │   Confirms the move has strength
        │   Filters out noise
        │
        └── Layer 4: Liquidity Check
            │   Verifies you can actually trade
            │   Checks orderbook depth
            │
            └── SIGNAL ✓
```

### Signal Confidence

Each layer adds to confidence:
- CUSUM trigger: +25%
- EWMA breakout: +25%
- ROC confirmation: +25%
- Good liquidity: +25%

**Minimum 50% confidence required to trade.**

---

## Docker Deployment

### Run with Docker Compose

```bash
# Build and start (paper trading)
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop
docker-compose down
```

### Access Dashboard

Open http://localhost:8501

### Live Trading (⚠️ Real Money)

```bash
# Edit .env first, then:
docker-compose --profile live up -d bot-live
```

---

## AWS EC2 Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete AWS setup guide.

**Quick Overview:**
1. Launch t3.micro EC2 instance
2. Install Docker
3. Clone repo and configure
4. Run with docker-compose
5. Access dashboard via public IP:8501

**Estimated Cost:** ~$8/month (t3.micro)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_detector.py -v

# Run with coverage
pytest tests/ --cov=src
```

---

## Risk Warning

⚠️ **IMPORTANT:**

- This bot trades with real money (when not in dry-run mode)
- Past performance doesn't guarantee future results
- Start with small amounts you can afford to lose
- Test thoroughly in dry-run mode first
- Monitor the bot regularly
- The authors are not responsible for financial losses

**Recommended:**
- Start with $75 capital (as designed)
- Keep daily loss limit at $2
- Run in dry-run mode for at least a week
- Understand the strategy before going live

---

## Troubleshooting

### Bot won't start
- Check API credentials in `.env`
- Verify `config/config.yaml` exists
- Check logs: `docker-compose logs bot`

### No signals detected
- Markets may be calm (no spikes)
- Try lowering `cusum_threshold` in config
- Check that markets meet volume minimum

### Dashboard not loading
- Verify bot is running first
- Check `data/bot_state.json` exists
- Try: `streamlit run src/dashboard/app.py`

### Circuit breaker triggered
- Normal after 3 consecutive losses
- Wait 30 minutes or restart bot
- Check `data/risk_state.json`

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by discussions about spike trading on prediction markets
- Uses [py-clob-client](https://github.com/Polymarket/py-clob-client) for Polymarket API
- CUSUM algorithm from Bell Labs statistical process control
- EWMA methodology from J.P. Morgan RiskMetrics

---

**Good luck! 🚀**