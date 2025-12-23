# AI Integration Guide

This guide covers the AI-powered trading features in Polybot, including Amazon Bedrock integration, ensemble signal generation, and risk management.

## Overview

Polybot's AI integration transforms the bot from a pure technical analysis system into an intelligent trading platform that combines:

1. **Foundation Model Analysis** - Amazon Bedrock (Claude, Titan, Mistral)
2. **Ensemble Signal Generation** - Weighted combination of multiple signal sources
3. **Advanced Position Sizing** - Kelly Criterion with Monte Carlo validation
4. **Decision Transparency** - Full reasoning audit trail

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Decision Engine                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Bedrock    │  │   Technical  │  │  Sentiment   │       │
│  │   Client     │  │   Signals    │  │   Analysis   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └────────────┬────┴─────────────────┘                │
│                      ▼                                       │
│              ┌───────────────┐                               │
│              │   Ensemble    │                               │
│              │   Generator   │                               │
│              └───────┬───────┘                               │
│                      │                                       │
│         ┌────────────┼────────────┐                          │
│         ▼            ▼            ▼                          │
│  ┌────────────┐ ┌──────────┐ ┌──────────────┐               │
│  │   Kelly    │ │  Monte   │ │  Reasoning   │               │
│  │  Criterion │ │  Carlo   │ │   Tracker    │               │
│  └────────────┘ └──────────┘ └──────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### AWS Credentials

Set your AWS credentials in `.env`:

```bash
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Model Selection
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

### AI Configuration in config.yaml

```yaml
ai:
  # Enable/disable AI features
  enabled: true
  
  # Model selection
  model: "claude-3-sonnet"  # Options: claude-3-sonnet, claude-3-haiku, titan-text, mistral-large
  
  # Autonomy level (0.0 = rules-only, 1.0 = AI-only)
  autonomy_level: 0.7
  
  # Minimum confidence to act on AI signals
  min_confidence: 0.6
  
  # Monte Carlo simulation
  monte_carlo_enabled: true
  monte_carlo_simulations: 1000
  
  # Analysis frequency (seconds between AI analyses)
  analysis_interval: 30
  
  # Model temperature (0.0 = deterministic, 1.0 = creative)
  temperature: 0.7

position_sizing:
  # Kelly Criterion settings
  kelly_fraction: 0.25  # Use 25% of Kelly for safety
  min_position: 1.0     # Minimum $1
  max_position: 10.0    # Maximum $10
```

## Supported Models

### Claude 3 Sonnet (Recommended)
```yaml
model: "claude-3-sonnet"
# Model ID: anthropic.claude-3-sonnet-20240229-v1:0
```
Best for: Balanced reasoning and speed, optimal for trading decisions.

### Claude 3 Haiku
```yaml
model: "claude-3-haiku"
# Model ID: anthropic.claude-3-haiku-20240307-v1:0
```
Best for: Faster responses, lower cost, suitable for high-frequency analysis.

### Amazon Titan Text
```yaml
model: "titan-text"
# Model ID: amazon.titan-text-express-v1
```
Best for: AWS-native integration, cost-effective.

### Mistral Large
```yaml
model: "mistral-large"
# Model ID: mistral.mistral-large-2402-v1:0
```
Best for: Alternative reasoning approach, diverse model ensemble.

## Components

### 1. Bedrock Client

The Bedrock client handles communication with AWS Bedrock:

```python
from src.ai.bedrock_client import BedrockClient

client = BedrockClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1"
)

response = await client.invoke(
    prompt="Analyze this market...",
    temperature=0.7,
    max_tokens=1000
)
```

### 2. Ensemble Signal Generator

Combines multiple signal sources with configurable weights:

```python
from src.ai.signal_generator import EnsembleSignalGenerator

generator = EnsembleSignalGenerator(autonomy_level=0.7)

signal = generator.generate(
    technical=technical_signals,
    ai=ai_signal,
    sentiment=sentiment_signal,
    mean_reversion=mean_reversion_signal
)

# Returns:
# - direction: BUY/SELL/HOLD
# - strength: 0.0 to 1.0
# - confidence: 0.0 to 1.0
# - components: breakdown of each signal
```

**Weight Distribution by Autonomy Level:**

| Autonomy | Technical | AI | Sentiment | Mean Reversion |
|----------|-----------|-----|-----------|----------------|
| 0.0      | 70%       | 0%  | 15%       | 15%            |
| 0.5      | 40%       | 35% | 10%       | 15%            |
| 1.0      | 10%       | 65% | 10%       | 15%            |

### 3. Monte Carlo Simulator

Simulates price paths to assess risk:

```python
from src.ai.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator(
    num_simulations=1000,
    time_horizon=10
)

result = simulator.simulate(
    current_price=0.50,
    volatility=0.02,
    fair_value=0.55
)

# Returns:
# - prob_profit: probability of profit
# - expected_return: mean return across simulations
# - var_95: 95% Value at Risk
# - var_99: 99% Value at Risk
# - max_loss: worst case loss
# - max_gain: best case gain
```

### 4. Kelly Criterion Position Sizing

Calculates optimal position size:

```python
# Kelly formula: f* = (p * b - q) / b
# Where:
#   p = probability of winning
#   q = probability of losing (1 - p)
#   b = win/loss ratio

kelly_fraction = generator.calculate_kelly(
    win_probability=0.65,
    win_loss_ratio=1.5,
    fraction=0.25  # Use 25% Kelly for safety
)

position_size = generator.calculate_position_size(
    capital=1000,
    kelly_fraction=kelly_fraction,
    max_position=100
)
```

### 5. Reasoning Tracker

Logs all AI decisions for audit and learning:

```python
from src.ai.reasoning_tracker import ReasoningTracker, DecisionOutcome

tracker = ReasoningTracker(log_dir="data/reasoning")

# Log decision
entry_id = tracker.log_decision(
    market_id="market_123",
    decision="BUY",
    confidence=0.75,
    reasoning="Strong bullish divergence detected",
    signals={
        "technical": 0.8,
        "ai": 0.7,
        "sentiment": 0.6
    }
)

# Record outcome when trade closes
tracker.record_outcome(
    entry_id=entry_id,
    outcome=DecisionOutcome.PROFIT,
    pnl=15.50,
    exit_price=0.65,
    exit_reason="take_profit"
)

# Get statistics
stats = tracker.get_stats()
# Returns win_rate, avg_confidence, profitable_trades, etc.
```

### 6. AI Decision Engine

Main coordinator that orchestrates all components:

```python
from src.ai import AIDecisionEngine, MarketContext, TradingContext

engine = AIDecisionEngine(settings=config)

# Create contexts
market_ctx = MarketContext(
    market_id="test_market",
    question="Will X happen?",
    current_price=0.50,
    # ... other fields
)

trading_ctx = TradingContext(
    current_capital=1000,
    available_capital=800,
    # ... other fields
)

# Get AI decision
decision = await engine.analyze_market(market_ctx, trading_ctx)

# Decision includes:
# - recommendation: BUY/SELL/HOLD
# - confidence: 0.0 to 1.0
# - position_size: calculated from Kelly
# - stop_loss, take_profit: suggested levels
# - reasoning: full explanation
```

## Dashboard Integration

The AI reasoning panel displays:

1. **Recent AI Decisions** - Table of decisions with reasoning
2. **Model Performance** - Win rate, average confidence
3. **Monte Carlo Results** - Risk simulation charts
4. **Decision Breakdown** - Signal component weights

Access via the "AI Analysis" tab in the Streamlit dashboard.

## Running with AI

### Enable AI Mode
```bash
python -m src.bot --dry-run
```

### Disable AI (Technical Only)
```bash
python -m src.bot --dry-run --no-ai
```

### Environment Requirements
```bash
# Install dependencies
pip install boto3 botocore scipy scikit-learn

# Verify AWS credentials
aws bedrock list-foundation-models --region us-east-1
```

## Best Practices

### 1. Start Conservative
Begin with low autonomy (0.3-0.5) and increase gradually:
```yaml
ai:
  autonomy_level: 0.3  # Start here
```

### 2. Use Fractional Kelly
Never use full Kelly - 25% is recommended:
```yaml
position_sizing:
  kelly_fraction: 0.25
```

### 3. Monitor AI Reasoning
Regularly review the reasoning logs:
```bash
cat data/reasoning/reasoning_$(date +%Y-%m-%d).jsonl | jq .
```

### 4. Validate with Paper Trading
Always test AI changes in dry-run mode first:
```bash
python -m src.bot --dry-run --config config/ai_test.yaml
```

### 5. Set Appropriate Confidence Thresholds
Don't act on low-confidence signals:
```yaml
ai:
  min_confidence: 0.6  # Minimum 60% confidence
```

## Troubleshooting

### AWS Credentials Error
```
Error: Unable to locate credentials
```
Solution: Ensure AWS credentials are set:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Model Not Available
```
Error: Model not found in region
```
Solution: Check model availability in your region:
```bash
aws bedrock list-foundation-models --region us-east-1 --query "modelSummaries[].modelId"
```

### Rate Limiting
```
Error: ThrottlingException
```
Solution: Increase analysis interval:
```yaml
ai:
  analysis_interval: 60  # Analyze every 60 seconds
```

### High Latency
```
Warning: AI analysis took >5 seconds
```
Solution: Use a faster model:
```yaml
ai:
  model: "claude-3-haiku"  # Faster than Sonnet
```

## Cost Estimation

| Model | Cost per 1K tokens | Typical Usage |
|-------|-------------------|---------------|
| Claude 3 Sonnet | $0.003 input, $0.015 output | ~$5-15/day active |
| Claude 3 Haiku | $0.00025 input, $0.00125 output | ~$0.50-2/day active |
| Titan Text | $0.0008 input, $0.0016 output | ~$1-3/day active |

## Security Considerations

1. **Never commit credentials** - Use environment variables
2. **Rotate keys regularly** - Set up IAM key rotation
3. **Use least privilege** - Bedrock invoke permission only
4. **Audit logs** - Enable CloudTrail for Bedrock calls
5. **VPC endpoints** - Use private endpoints in production

## API Reference

### MarketContext Fields
| Field | Type | Description |
|-------|------|-------------|
| market_id | str | Unique market identifier |
| question | str | Market question text |
| current_price | float | Current YES token price |
| ewma_price | float | EWMA smoothed price |
| roc | float | Rate of change |
| volatility | float | Current volatility |
| volume_24h | float | 24h trading volume |

### TradingContext Fields
| Field | Type | Description |
|-------|------|-------------|
| current_capital | float | Total account value |
| available_capital | float | Capital available to trade |
| daily_pnl | float | Today's P&L |
| open_positions_count | int | Number of open positions |
| win_rate | float | Historical win rate |

### AIDecision Fields
| Field | Type | Description |
|-------|------|-------------|
| recommendation | enum | BUY, SELL, or HOLD |
| confidence | float | 0.0 to 1.0 |
| position_size | float | Suggested position size |
| stop_loss | float | Suggested stop loss |
| take_profit | float | Suggested take profit |
| reasoning | str | Full explanation |
| is_actionable | bool | Whether to execute |