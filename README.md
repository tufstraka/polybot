# 🤖 Polybot - AI-Powered Spike Hunter

**AI-powered autonomous trading bot for Polymarket with Amazon Bedrock integration.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

---

## What It Does

Polybot watches Polymarket prediction markets using a **hybrid AI + technical analysis** approach. It combines foundation model reasoning (Claude, Titan, Mistral) with statistical spike detection to identify and trade market inefficiencies.

**The Strategy:**
1. **Technical Detection**: CUSUM + EWMA + ROC detect price spikes
2. **AI Analysis**: Bedrock models analyze market context and sentiment
3. **Ensemble Signals**: Weighted combination of all signal sources
4. **Smart Sizing**: Kelly Criterion + Monte Carlo risk validation
5. **Mean Reversion**: Trade the bounce-back with calculated confidence

---

## Features

### Core Trading
✅ **4-Layer Spike Detection**
- CUSUM algorithm (Bell Labs) for regime changes
- EWMA volatility bands (J.P. Morgan RiskMetrics)
- ROC momentum confirmation
- Liquidity validation

### AI Integration 
🤖 **Amazon Bedrock Models**
- Claude 3 Sonnet/Haiku for market reasoning
- Titan Text for cost-effective analysis
- Mistral Large for diverse perspectives

📊 **Ensemble Signal Generation**
- Combines technical + AI + sentiment + mean reversion
- Configurable autonomy level (0.0 = rules-only to 1.0 = AI-only)
- Weighted signal aggregation

📈 **Advanced Position Sizing**
- Kelly Criterion for optimal bet sizing
- Fractional Kelly for safety (default 25%)
- Monte Carlo validation (1000+ simulations)

📝 **Decision Transparency**
- Full reasoning audit trail
- All AI decisions logged to JSONL
- Dashboard shows AI rationale

### Risk Management
✅ **Risk Controls**
- Daily loss limit ($2 default)
- Position sizing based on confidence
- Circuit breaker after losing streaks
- Capital protection

✅ **Paper Trading Mode**
- Test without real money
- Full simulation of trades
- AI decisions tracked in dry-run

### Monitoring
✅ **Real-time Dashboard**
- See tracked markets
- View detected signals
- Monitor P&L and positions
- **AI Reasoning panel** (NEW)
- **Monte Carlo charts** (NEW)

✅ **Notifications**
- Telegram alerts
- Discord webhooks
- Trade confirmations
- AI decision summaries

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

Add your credentials:
```env
# Required - Polymarket API
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_PASSPHRASE=your_passphrase

# Required for AI features
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

### 3. Run (Paper Trading)

```bash
# Start bot with AI (paper trading)
python -m src.bot --dry-run

# Start bot without AI (technical only)
python -m src.bot --dry-run --no-ai
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
│   ├── bot.py              # Main orchestrator (AI-integrated)
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
│   ├── ai/                 # AI DECISION ENGINE (NEW)
│   │   ├── bedrock_client.py    # AWS Bedrock integration
│   │   ├── decision_engine.py   # Main AI coordinator
│   │   ├── signal_generator.py  # Ensemble signal generation
│   │   ├── monte_carlo.py       # Monte Carlo simulation
│   │   ├── reasoning_tracker.py # Decision audit trail
│   │   └── prompts.py           # LLM prompt templates
│   ├── risk/
│   │   └── risk_manager.py # Risk management
│   ├── notifications/
│   │   ├── telegram.py     # Telegram alerts
│   │   └── discord.py      # Discord webhooks
│   └── dashboard/
│       └── app.py          # Streamlit dashboard (AI panel)
├── config/
│   └── config.yaml         # Main configuration
├── tests/                  # Unit tests
├── docs/
│   ├── DEPLOYMENT.md       # Deployment guide
│   └── AI_INTEGRATION.md   # AI integration guide (NEW)
├── data/
│   └── reasoning/          # AI decision logs (NEW)
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

# Run AI component tests
pytest tests/test_ai_components.py -v

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

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md) - AWS EC2 setup and Docker deployment
- [AI Integration Guide](docs/AI_INTEGRATION.md) - Bedrock setup, Kelly Criterion, Monte Carlo

---

## Acknowledgments

- Inspired by discussions about spike trading on prediction markets
- Uses [py-clob-client](https://github.com/Polymarket/py-clob-client) for Polymarket API
- CUSUM algorithm from Bell Labs statistical process control
- EWMA methodology from J.P. Morgan RiskMetrics
- Amazon Bedrock for foundation model access
- Kelly Criterion from "Fortune's Formula" by William Poundstone

---

**Good luck! 🚀🤖**
