# Polymarket Spike Detection Algorithm

## The Simple Idea

**Prediction markets overreact to news. This bot catches those overreactions and profits when prices calm down.**

```
📰 News Hits → 😱 Panic/Excitement → 📈 Price Spikes → 😌 Calm Down → 💰 Profit
```

## How It Works (Plain English)

### Step 1: Watch for Unusual Moves
The bot watches prices every second. When a price moves **much more than usual**, that's a spike.

Think of it like this:
- If a market normally moves 0.1% per second
- And suddenly it moves 2% in one second
- That's 20x the normal movement = SPIKE DETECTED

### Step 2: Confirm It's Real
Not every big move is worth trading. The bot checks:
- ✅ Is there enough money in the market to trade? (Liquidity)
- ✅ Is the move continuing in one direction? (Momentum)
- ✅ Have we traded this market recently? (Cooldown)

### Step 3: Enter the Trade
When all checks pass:
- Enter a small position ($1-3)
- Set automatic exit points (take profit + stop loss)

### Step 4: Exit Smart
The bot exits when:
- 🎯 **Target hit**: Price moved in your favor (take profit)
- 🛑 **Stop hit**: Price moved against you (stop loss)
- ⏰ **Time out**: Too long without movement (close at current price)

---

## The Algorithm (Simplified)

### Core Formula: Spike Score

```
                    (Current Price - Average Price)
Spike Score = ─────────────────────────────────────────
                        Normal Volatility

If Spike Score > Threshold → SPIKE DETECTED
```

**What this means:**
- We compare current price to the recent average
- We divide by "normal volatility" (how much prices usually move)
- A high score means "this move is unusual"

### Visual Example

```
Price History: [0.50, 0.51, 0.50, 0.49, 0.50, 0.51, 0.50]  ← Normal bouncing
Average: 0.50
Normal Volatility: 0.01 (prices usually vary by 1 cent)

Then suddenly: 0.55  ← New price!

Spike Score = (0.55 - 0.50) / 0.01 = 5.0

If our threshold is 2.0, then 5.0 > 2.0 = SPIKE DETECTED!
```

---

## Trading Strategy: Mean Reversion

**Why Mean Reversion Works in Prediction Markets:**

1. **Overreaction is common**: People panic-buy or panic-sell on news
2. **Smart money corrects**: Experienced traders bring prices back to fair value
3. **Bounded prices**: Prices can't go below 0 or above 1, limiting extreme moves

**The Strategy:**

```
When price SPIKES UP:
  → Bet it will come back DOWN (Sell/Short)
  
When price SPIKES DOWN:
  → Bet it will come back UP (Buy/Long)
```

**Example:**
```
Market: "Will Bitcoin hit $100k by December?"
Normal price: 0.45 (45% probability)

News: "Bitcoin ETF approved!"
Price spikes to: 0.55 (+10 cents in seconds)

Bot action: SELL at 0.55
Why: The spike was emotional. Price will likely settle around 0.50-0.52.

Result: Price settles at 0.51
Profit: 0.55 - 0.51 = $0.04 per share (4 cents = ~7% gain)
```

---

## Configuration (Plain English)

```yaml
# config/config.yaml

# ===========================================
# MODE: Are we testing or trading for real?
# ===========================================
mode:
  paper_trading: true    # true = fake money (testing), false = real money
  
# ===========================================
# MONEY: How much are we working with?
# ===========================================
money:
  starting_balance: 75   # Your total capital in USD
  bet_size: 2            # How much to bet per trade (USD)
  max_daily_loss: 2      # Stop trading if we lose this much today

# ===========================================
# SPIKE DETECTION: How sensitive should we be?
# ===========================================
detection:
  # How unusual must a move be to trigger a trade?
  # Lower = more trades (but more false signals)
  # Higher = fewer trades (but higher quality signals)
  # Recommended: 2.0 for beginners, 1.5 for aggressive
  sensitivity: 2.0
  
  # How many seconds of price history to analyze?
  # Lower = react faster to new moves
  # Higher = smoother, ignores tiny spikes
  # Recommended: 30 seconds
  lookback_seconds: 30
  
  # Minimum price change to even consider (percentage)
  # Filters out tiny moves that aren't worth trading
  # Recommended: 1% minimum
  min_move_percent: 1.0

# ===========================================
# RISK: How do we protect our money?
# ===========================================
risk:
  # When to take profits (percentage gain)
  # Example: 3 means exit when up 3%
  take_profit_percent: 3.0
  
  # When to cut losses (percentage loss)
  # Example: 2 means exit when down 2%
  stop_loss_percent: 2.0
  
  # Maximum trades open at once
  # More = more potential profit but more risk
  max_open_trades: 2
  
  # Seconds to wait before trading same market again
  # Prevents over-trading the same spike
  cooldown_seconds: 60

# ===========================================
# MARKET FILTERS: Which markets to trade?
# ===========================================
filters:
  # Only trade markets with this much daily volume (USD)
  # Higher = more liquid markets, easier to trade
  min_daily_volume: 5000
  
  # Only trade if bid-ask spread is below this (percentage)
  # Lower = cheaper to trade, less slippage
  max_spread_percent: 3.0

# ===========================================
# NOTIFICATIONS: How to get alerts?
# ===========================================
notifications:
  telegram_enabled: true
  discord_enabled: true
  
  # What to be notified about
  notify_on_trade: true       # When bot opens/closes a trade
  notify_on_daily_summary: true  # Daily P&L report
```

---

## Dashboard (Simplified)

```
┌──────────────────────────────────────────────────────────────────┐
│  🤖 POLYBOT                              📋 Paper Trading Mode   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TODAY'S PERFORMANCE                                             │
│  ────────────────────                                            │
│  💰 Profit/Loss:  +$0.45                                         │
│  📊 Win Rate:     67% (4 wins / 6 trades)                        │
│  🎯 Open Trades:  1 of 2 max                                     │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WATCHING THESE MARKETS                                          │
│  ──────────────────────                                          │
│  🔴 Trump Win 2024     0.52 → 0.55  (+3.1%)  ⚡ SPIKE!           │
│  🟢 ETH > $4000        0.31 → 0.30  (-0.5%)  Normal              │
│  🟢 Fed Rate Cut       0.78 → 0.78  ( 0.0%)  Normal              │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PRICE CHART: Trump Win 2024                                     │
│  ───────────────────────────                                     │
│                                                                  │
│  0.56 │                         ╭─╮                              │
│       │                        ╭╯ ╰╮  ← Spike detected here      │
│  0.54 │                      ╭─╯   ╰─╮                           │
│       │                    ╭─╯       │                           │
│  0.52 │  ────────────────╭─╯         ╰───                        │
│       │                                                          │
│  0.50 │                                                          │
│       └──────────────────────────────────────────────            │
│         10:00    10:05    10:10    10:15    10:20                │
│                                                                  │
│  ══════ Upper Band (sell zone)                                   │
│  ────── Normal Price                                             │
│  ══════ Lower Band (buy zone)                                    │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RECENT TRADES                                                   │
│  ─────────────                                                   │
│  ✅ 10:15  Trump Win    SELL  $2.00  →  +$0.12 (+6%)            │
│  ❌ 10:02  ETH > $4k    BUY   $2.00  →  -$0.08 (-4%)            │
│  ✅ 09:45  Fed Rate     SELL  $2.00  →  +$0.06 (+3%)            │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SAFETY STATUS                                                   │
│  ─────────────                                                   │
│  Daily Loss:   [$0.32]═══════════░░░░░░░░░░░░░░░░ $2.00 max     │
│  Spike Score:  [2.3]═════════════════░░░░░░░░░░░░ 2.0 threshold │
│  Cooldowns:    1 market waiting                                  │
│  Circuit:      ✅ All systems go                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why This Strategy is Profitable

### Academic Evidence

1. **Overreaction Hypothesis** (De Bondt & Thaler, 1985)
   - Markets systematically overreact to news
   - Extreme moves are followed by reversals
   - This effect is stronger in smaller, less efficient markets (like Polymarket)

2. **Mean Reversion** (Poterba & Summers, 1988)
   - Prices tend to return to their long-term average
   - Short-term spikes don't reflect true probability changes
   - Emotional trading creates temporary mispricings

3. **Prediction Market Inefficiencies** (Wolfers & Zitzewitz, 2004)
   - Prediction markets are less efficient than traditional markets
   - Fewer professional traders means more mispricings
   - News events create predictable overreaction patterns

### Why Prediction Markets Are Especially Prone to Spikes

| Factor | Effect |
|--------|--------|
| **Retail traders** | More emotional, panic buy/sell |
| **News-driven** | Single tweets can move markets |
| **Bounded prices** | Can't go below 0 or above 1 |
| **Low liquidity** | Easier to move prices with small orders |
| **No shorting friction** | Both directions equally easy |

### Expected Performance

Based on mean reversion strategies in similar markets:

| Metric | Conservative | Moderate | Aggressive |
|--------|-------------|----------|------------|
| Win Rate | 55-60% | 50-55% | 45-50% |
| Avg Win | 2-3% | 3-4% | 4-6% |
| Avg Loss | 1-2% | 2-3% | 3-4% |
| Trades/Day | 2-5 | 5-15 | 15-30 |
| Expected Daily | +0.5-1% | +0.5-2% | -1% to +3% |

**Note**: These are estimates. Actual results depend on market conditions.

---

## Configuration Presets

### 🛡️ Conservative (Recommended for Beginners)

```yaml
detection:
  sensitivity: 2.5      # Only trade obvious spikes
  lookback_seconds: 45  # Smoother detection
  min_move_percent: 1.5

risk:
  take_profit_percent: 2.0  # Take small wins
  stop_loss_percent: 1.5    # Cut losses quick
  max_open_trades: 1        # One trade at a time
  cooldown_seconds: 120     # Long cooldown
```

**Expected**: 2-3 trades/day, 55-60% win rate, slow but steady

### ⚖️ Balanced (Recommended After Testing)

```yaml
detection:
  sensitivity: 2.0
  lookback_seconds: 30
  min_move_percent: 1.0

risk:
  take_profit_percent: 3.0
  stop_loss_percent: 2.0
  max_open_trades: 2
  cooldown_seconds: 60
```

**Expected**: 5-10 trades/day, 50-55% win rate, good risk/reward

### ⚡ Aggressive (Only After Proven Profitability)

```yaml
detection:
  sensitivity: 1.5      # Catch smaller spikes
  lookback_seconds: 20  # React faster
  min_move_percent: 0.5

risk:
  take_profit_percent: 4.0
  stop_loss_percent: 2.5
  max_open_trades: 3
  cooldown_seconds: 30
```

**Expected**: 15-25 trades/day, 45-50% win rate, high variance

---

## Risk Warnings

### What Can Go Wrong

1. **Trend continuation**: Sometimes spikes are the start of a real move, not an overreaction
2. **Liquidity gaps**: Price might not return to normal before your stop is hit
3. **News events**: Major news can cause permanent price shifts
4. **Fees/spread**: Trading costs eat into small gains
5. **API issues**: Connection problems can leave positions open

### How We Mitigate These Risks

| Risk | Mitigation |
|------|------------|
| Wrong direction | Stop loss limits maximum loss per trade |
| Over-trading | Cooldown prevents revenge trading |
| Big daily loss | $2 daily loss limit stops the bot |
| Illiquid markets | Volume/spread filters avoid bad markets |
| Technical issues | State persistence, auto-reconnect |

### The Golden Rule

**Never risk money you can't afford to lose.**

Start with paper trading. Run it for at least a week. Only go live with small amounts after you see consistent simulated profits.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Spike** | An unusually large price movement in a short time |
| **Mean Reversion** | The tendency of prices to return to their average |
| **Sensitivity** | How big a move needs to be to trigger a trade |
| **Lookback** | How far back in time to analyze prices |
| **Take Profit** | The price level where we automatically sell for a gain |
| **Stop Loss** | The price level where we automatically sell to limit loss |
| **Cooldown** | Wait time before trading the same market again |
| **Liquidity** | How easily you can buy/sell without affecting price |
| **Spread** | The difference between buy and sell prices |
| **Paper Trading** | Trading with fake money to test strategies |