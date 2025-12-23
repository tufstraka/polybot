# Polybot Deployment Guide

This guide explains how to deploy Polybot to AWS EC2 and run it 24/7.

## Quick Start (Local)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd polybot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
- `POLYMARKET_API_KEY` - Your Polymarket API key
- `POLYMARKET_API_SECRET` - Your Polymarket API secret
- `POLYMARKET_PASSPHRASE` - Your Polymarket passphrase (if required)

Optional:
- `TELEGRAM_BOT_TOKEN` - For Telegram notifications
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID
- `DISCORD_WEBHOOK_URL` - For Discord notifications

### 3. Run in Dry-Run Mode (Paper Trading)

```bash
# Test with paper trading first!
python -m src.bot --dry-run
```

### 4. View Dashboard

```bash
# In a separate terminal
streamlit run src/dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## Docker Deployment (Recommended)

### Prerequisites

- Docker installed
- Docker Compose installed

### 1. Build and Run

```bash
# Build images
docker-compose build

# Start in dry-run mode (safe)
docker-compose up -d

# View logs
docker-compose logs -f bot
```

### 2. Access Dashboard

Open http://localhost:8501

### 3. Stop

```bash
docker-compose down
```

### 4. Run in Live Mode (Real Trading)

⚠️ **WARNING: This uses real money!**

```bash
# Edit .env and set your credentials
# Then run with live profile
docker-compose --profile live up -d bot-live
```

---

## AWS EC2 Deployment

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose settings:
   - **AMI**: Amazon Linux 2023 or Ubuntu 22.04
   - **Instance Type**: t3.micro (free tier) or t3.small
   - **Storage**: 20 GB (minimum)
   - **Security Group**: Allow ports 22 (SSH), 8501 (Dashboard)

3. Create or select a key pair for SSH access

### Step 2: Connect to Instance

```bash
# Connect via SSH
ssh -i your-key.pem ec2-user@your-instance-ip
```

### Step 3: Install Docker

For Amazon Linux 2023:
```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and back in for group changes
exit
```

For Ubuntu:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose-plugin

# Log out and back in
exit
```

### Step 4: Deploy Polybot

```bash
# Connect again
ssh -i your-key.pem ec2-user@your-instance-ip

# Clone repository
git clone <your-repo-url>
cd polybot

# Create .env file
nano .env
```

Add your configuration:
```env
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_api_secret
POLYMARKET_PASSPHRASE=your_passphrase
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

```bash
# Build and start
docker-compose build
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f bot
```

### Step 5: Access Dashboard

1. Get your EC2 public IP from AWS Console
2. Open http://your-ec2-ip:8501

### Step 6: Set Up Auto-Restart

The bot automatically restarts on failure (via Docker's `restart: unless-stopped`).

To survive instance reboots:
```bash
# Enable Docker to start on boot
sudo systemctl enable docker

# Optional: Add to crontab
crontab -e
# Add line:
@reboot cd /home/ec2-user/polybot && docker-compose up -d
```

---

## Configuration Reference

### config/config.yaml

```yaml
# Main configuration file
polymarket:
  api_url: "https://clob.polymarket.com"  # Don't change

trading:
  dry_run: true        # Set false for live trading

detection:
  lookback_window: 60   # Seconds of price history
  cusum_threshold: 2.5  # Spike sensitivity
  ewma_span: 20         # Smoothing period
  ewma_band_width: 2.0  # Volatility band width
  roc_threshold: 0.02   # Minimum momentum (2%)

risk:
  max_daily_loss: 2.0   # Stop trading after $2 loss
  max_position_size: 2.0  # Max per trade
  min_position_size: 0.5  # Min per trade

money:
  initial_capital: 75.0  # Your starting capital
```

### Environment Variables (.env)

```env
# Required
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_PASSPHRASE=your_passphrase

# Optional - Notifications
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=987654321
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Optional - Override config
POLYBOT_DRY_RUN=true
POLYBOT_INITIAL_CAPITAL=75.0
POLYBOT_MAX_DAILY_LOSS=2.0
```

---

## Monitoring

### Check Bot Status

```bash
# View logs
docker-compose logs -f bot

# Check container health
docker-compose ps

# View state file
cat data/bot_state.json | python -m json.tool
```

### Dashboard Features

- **Performance**: Track P&L, win rate, capital
- **Risk Status**: Daily limits, circuit breaker status
- **Markets**: See all tracked markets with indicators
- **Signals**: View detected spikes
- **Positions**: Monitor open trades

### Notifications

Set up Telegram or Discord to receive alerts:

1. **Signal detected**: When a spike is found
2. **Trade opened/closed**: Position updates
3. **Warnings**: Risk limit approaching
4. **Errors**: If something goes wrong

---

## Troubleshooting

### Bot won't start

```bash
# Check logs
docker-compose logs bot

# Common issues:
# - Invalid API credentials
# - Network connectivity
# - Missing config file
```

### No signals detected

- Markets may be calm (no spikes)
- Check that markets are being tracked
- Lower sensitivity in config (reduce cusum_threshold)

### Dashboard not loading

```bash
# Check dashboard container
docker-compose logs dashboard

# Verify port is open
curl http://localhost:8501
```

### Circuit breaker triggered

The bot pauses after 3 consecutive losses. Wait for reset or manually:
```bash
# View risk state
cat data/risk_state.json
```

---

## Going Live

⚠️ **Before trading with real money:**

1. ✅ Test thoroughly in dry-run mode
2. ✅ Verify notifications work
3. ✅ Understand the risk parameters
4. ✅ Start with minimum position sizes
5. ✅ Monitor closely for the first few days

To enable live trading:

```bash
# Option 1: Edit .env
POLYBOT_DRY_RUN=false

# Option 2: Use live profile
docker-compose --profile live up -d bot-live
```

---

## Cost Estimates

### AWS EC2

| Instance | vCPU | RAM | Cost/month |
|----------|------|-----|------------|
| t3.micro | 2 | 1 GB | ~$8 |
| t3.small | 2 | 2 GB | ~$15 |
| t3.medium | 2 | 4 GB | ~$30 |

t3.micro is sufficient for Polybot.

### Trading

- **Initial capital**: $75 (recommended minimum)
- **Per trade**: $0.50 - $2.00
- **Daily limit**: $2.00 max loss
- **Polymarket fees**: ~0.5% per trade

---

## Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Review the dashboard for error messages
3. Verify your API credentials
4. Check Polymarket API status

Good luck! 🚀