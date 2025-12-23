"""
Polybot Dashboard - See what your bot is doing.

Plain English Explanation:
==========================

This is your "control center" where you can see:
- Is the bot running?
- What markets is it watching?
- What spikes has it detected?
- How much money have you made/lost?
- What's the current risk status?

Run with:
    streamlit run src/dashboard/app.py

The dashboard automatically refreshes every few seconds to show
the latest data from the bot.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.state_writer import StateReader

# Page config
st.set_page_config(
    page_title="Polybot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
    .status-running { color: #00ff00; }
    .status-stopped { color: #ff0000; }
    .status-paused { color: #ffff00; }
    .signal-up { color: #00ff00; }
    .signal-down { color: #ff0000; }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)


def load_state():
    """Load current bot state from shared file."""
    reader = StateReader("data/bot_state.json")
    return reader.read()


def format_timedelta(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def render_header(state: dict):
    """Render the top header with bot status."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Bot status
    status = state.get("bot_status", "unknown")
    status_emoji = {
        "running": "🟢",
        "paused": "🟡",
        "stopped": "🔴",
        "error": "🔴"
    }.get(status, "⚪")
    
    with col1:
        st.metric(
            label="Bot Status",
            value=f"{status_emoji} {status.upper()}"
        )
    
    # Mode
    mode = state.get("bot_mode", "unknown")
    mode_emoji = "🧪" if mode == "dry_run" else "💰"
    
    with col2:
        st.metric(
            label="Mode",
            value=f"{mode_emoji} {mode.replace('_', ' ').title()}"
        )
    
    # Uptime
    uptime = state.get("uptime_seconds", 0)
    with col3:
        st.metric(
            label="Uptime",
            value=format_timedelta(uptime)
        )
    
    # Markets tracked
    with col4:
        st.metric(
            label="Markets Tracked",
            value=state.get("markets_tracked", 0)
        )
    
    # Signals today
    with col5:
        st.metric(
            label="Signals Today",
            value=state.get("signals_today", 0)
        )


def render_performance(state: dict):
    """Render performance metrics card."""
    st.subheader("📊 Performance")
    
    perf = state.get("performance")
    if not perf:
        st.info("No performance data yet. Start the bot to begin trading.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        daily_pnl = perf.get("daily_pnl", 0)
        st.metric(
            label="Today's P&L",
            value=f"${daily_pnl:+.2f}",
            delta=f"{'profit' if daily_pnl > 0 else 'loss' if daily_pnl < 0 else 'flat'}"
        )
    
    with col2:
        total_pnl = perf.get("total_pnl", 0)
        st.metric(
            label="Total P&L",
            value=f"${total_pnl:+.2f}"
        )
    
    with col3:
        win_rate = perf.get("win_rate", 0)
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%"
        )
    
    with col4:
        capital = perf.get("current_capital", 0)
        initial = perf.get("initial_capital", 75)
        change = ((capital - initial) / initial * 100) if initial > 0 else 0
        st.metric(
            label="Capital",
            value=f"${capital:.2f}",
            delta=f"{change:+.1f}%"
        )
    
    # Additional stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Trades",
            value=perf.get("total_trades", 0)
        )
    
    with col2:
        st.metric(
            label="Wins / Losses",
            value=f"{perf.get('winning_trades', 0)} / {perf.get('losing_trades', 0)}"
        )
    
    with col3:
        st.metric(
            label="Best Trade",
            value=f"${perf.get('best_trade', 0):+.2f}"
        )
    
    with col4:
        st.metric(
            label="Worst Trade",
            value=f"${perf.get('worst_trade', 0):+.2f}"
        )


def render_risk_status(state: dict):
    """Render risk management status."""
    st.subheader("⚠️ Risk Status")
    
    risk = state.get("risk")
    if not risk:
        st.info("No risk data available.")
        return
    
    # Status indicator
    status = risk.get("status", "unknown")
    status_colors = {
        "ok": "🟢",
        "warning": "🟡",
        "daily_limit": "🔴",
        "circuit_breaker": "🔴",
        "low_capital": "🔴"
    }
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"### {status_colors.get(status, '⚪')} {status.upper()}")
    
    with col2:
        st.markdown(f"*{risk.get('status_message', '')}*")
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        daily_pnl = risk.get("daily_pnl", 0)
        daily_limit = risk.get("daily_limit", 2)
        st.metric(
            label="Daily P&L",
            value=f"${daily_pnl:+.2f}",
            delta=f"Limit: ${daily_limit:.2f}"
        )
    
    with col2:
        remaining = risk.get("remaining_risk", 0)
        st.metric(
            label="Remaining Risk",
            value=f"${remaining:.2f}"
        )
    
    with col3:
        consec = risk.get("consecutive_losses", 0)
        st.metric(
            label="Losing Streak",
            value=f"{consec} trades"
        )
    
    with col4:
        can_trade = risk.get("can_trade", False)
        st.metric(
            label="Can Trade",
            value="✅ YES" if can_trade else "❌ NO"
        )
    
    # Circuit breaker
    if risk.get("circuit_breaker_active"):
        remaining_mins = risk.get("circuit_breaker_remaining", 0)
        st.warning(f"⏰ Circuit breaker active. Resuming in {remaining_mins:.1f} minutes.")


def render_markets(state: dict):
    """Render tracked markets table with indicators."""
    st.subheader("📈 Tracked Markets")
    
    markets = state.get("markets", [])
    if not markets:
        st.info("No markets being tracked. Start the bot to begin scanning.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(markets)
    
    # Format columns
    if "current_price" in df.columns:
        df["Price"] = df["current_price"].apply(lambda x: f"${x:.3f}")
    if "ewma_price" in df.columns:
        df["EWMA"] = df["ewma_price"].apply(lambda x: f"${x:.3f}")
    if "roc" in df.columns:
        df["ROC"] = df["roc"].apply(lambda x: f"{x:+.2%}")
    if "volume_24h" in df.columns:
        df["Volume"] = df["volume_24h"].apply(lambda x: f"${x:,.0f}")
    
    # Select columns to display
    display_cols = ["name", "Price", "EWMA", "ROC", "Volume"]
    display_cols = [c for c in display_cols if c in df.columns or c.lower() in df.columns]
    
    if display_cols:
        st.dataframe(
            df[display_cols] if display_cols else df,
            use_container_width=True,
            hide_index=True
        )


def render_price_chart(state: dict):
    """Render price chart with EWMA bands for selected market."""
    st.subheader("📊 Price Chart with EWMA Bands")
    
    markets = state.get("markets", [])
    if not markets:
        st.info("No market data available for charting.")
        return
    
    # Market selector
    market_names = [m.get("name", m.get("market_id", "Unknown")) for m in markets]
    selected = st.selectbox("Select Market", market_names)
    
    # Find selected market
    market = next((m for m in markets if m.get("name") == selected), None)
    if not market:
        return
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & EWMA Bands", "CUSUM Indicator")
    )
    
    # For demo, we'll show current values as points
    # In production, you'd have historical data
    current_price = market.get("current_price", 0)
    ewma_price = market.get("ewma_price", 0)
    ewma_upper = market.get("ewma_upper", 0)
    ewma_lower = market.get("ewma_lower", 0)
    cusum_pos = market.get("cusum_pos", 0)
    cusum_neg = market.get("cusum_neg", 0)
    
    # Price and bands
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[current_price],
            mode="markers+text",
            name="Current Price",
            marker=dict(size=15, color="white"),
            text=[f"${current_price:.3f}"],
            textposition="top center"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[ewma_price],
            mode="markers",
            name="EWMA",
            marker=dict(size=10, color="yellow")
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[ewma_upper],
            mode="markers",
            name="Upper Band",
            marker=dict(size=8, color="green", symbol="triangle-up")
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[ewma_lower],
            mode="markers",
            name="Lower Band",
            marker=dict(size=8, color="red", symbol="triangle-down")
        ),
        row=1, col=1
    )
    
    # CUSUM
    fig.add_trace(
        go.Bar(
            x=["CUSUM+", "CUSUM-"],
            y=[cusum_pos, cusum_neg],
            marker_color=["green", "red"],
            name="CUSUM"
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        height=500,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Indicator values
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.3f}")
    with col2:
        st.metric("EWMA", f"${ewma_price:.3f}")
    with col3:
        roc = market.get("roc", 0)
        st.metric("ROC", f"{roc:+.2%}")
    with col4:
        vol = market.get("volume_24h", 0)
        st.metric("24h Volume", f"${vol:,.0f}")


def render_signals(state: dict):
    """Render recent signals log."""
    st.subheader("🎯 Recent Signals")
    
    signals = state.get("recent_signals", [])
    if not signals:
        st.info("No signals detected yet. The bot will show detected spikes here.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(signals)
    
    # Format columns
    if "direction" in df.columns:
        df["Direction"] = df["direction"].apply(
            lambda x: "🚀 UP" if x == "up" else "📉 DOWN"
        )
    if "price" in df.columns:
        df["Price"] = df["price"].apply(lambda x: f"${x:.3f}")
    if "confidence" in df.columns:
        df["Confidence"] = df["confidence"].apply(lambda x: f"{x:.0%}")
    if "detected_at" in df.columns:
        df["Time"] = pd.to_datetime(df["detected_at"]).dt.strftime("%H:%M:%S")
    if "status" in df.columns:
        df["Status"] = df["status"].apply(
            lambda x: {"detected": "🔵", "traded": "🟢", "expired": "⚪"}.get(x, "⚪") + f" {x}"
        )
    
    # Select columns
    display_cols = ["market_name", "Direction", "Price", "Confidence", "Time", "Status"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    if display_cols:
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True
        )


def render_positions(state: dict):
    """Render open positions."""
    st.subheader("💼 Positions")
    
    positions = state.get("positions", [])
    if not positions:
        st.info("No positions open.")
        return
    
    # Filter open positions
    open_positions = [p for p in positions if p.get("status") == "open"]
    
    if not open_positions:
        st.info("No open positions. Recent closed positions:")
        # Show recent closed
        closed = [p for p in positions if p.get("status") == "closed"][:5]
        if closed:
            df = pd.DataFrame(closed)
            st.dataframe(df, use_container_width=True, hide_index=True)
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(open_positions)
    
    # Format columns
    if "entry_price" in df.columns:
        df["Entry"] = df["entry_price"].apply(lambda x: f"${x:.3f}")
    if "current_price" in df.columns:
        df["Current"] = df["current_price"].apply(lambda x: f"${x:.3f}")
    if "unrealized_pnl" in df.columns:
        df["P&L"] = df["unrealized_pnl"].apply(
            lambda x: f"${x:+.2f}" if x else "$0.00"
        )
    if "size" in df.columns:
        df["Size"] = df["size"].apply(lambda x: f"${x:.2f}")
    if "stop_loss" in df.columns:
        df["SL"] = df["stop_loss"].apply(lambda x: f"${x:.3f}")
    if "take_profit" in df.columns:
        df["TP"] = df["take_profit"].apply(lambda x: f"${x:.3f}")
    
    # Select columns
    display_cols = ["market_name", "side", "Entry", "Current", "P&L", "Size", "SL", "TP"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    if display_cols:
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True
        )


def render_events(state: dict):
    """Render recent events/activity log."""
    st.subheader("📝 Activity Log")
    
    events = state.get("recent_events", [])
    if not events:
        st.info("No recent activity.")
        return
    
    # Show last 10 events
    for event in events[-10:][::-1]:
        event_type = event.get("type", "info")
        message = event.get("message", "")
        timestamp = event.get("timestamp", "")
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp
        else:
            time_str = "?"
        
        # Type emoji
        emoji = {
            "signal": "🎯",
            "trade": "💰",
            "error": "🚨",
            "warning": "⚠️",
            "system": "ℹ️"
        }.get(event_type, "📌")
        
        st.text(f"{time_str} {emoji} {message}")


def render_sidebar():
    """Render sidebar with controls and info."""
    with st.sidebar:
        st.title("🤖 Polybot")
        st.markdown("---")
        
        # Refresh rate
        refresh_rate = st.slider(
            "Refresh Rate (seconds)",
            min_value=1,
            max_value=30,
            value=5
        )
        
        st.markdown("---")
        
        # Quick info
        st.markdown("### Quick Guide")
        st.markdown("""
        **Status Indicators:**
        - 🟢 Running/OK
        - 🟡 Warning/Paused
        - 🔴 Error/Stopped
        
        **Signal Types:**
        - 🚀 UP spike detected
        - 📉 DOWN spike detected
        
        **Signal Status:**
        - 🔵 Detected (pending)
        - 🟢 Traded
        - ⚪ Expired
        """)
        
        st.markdown("---")
        
        # Last updated
        st.markdown(f"*Last refresh: {datetime.now().strftime('%H:%M:%S')}*")
        
        return refresh_rate


def main():
    """Main dashboard application."""
    # Sidebar
    refresh_rate = render_sidebar()
    
    # Title
    st.title("🤖 Polybot Spike Hunter Dashboard")
    
    # Load state
    state = load_state()
    
    if not state:
        st.warning("⚠️ Bot is not running or state file not found.")
        st.info("""
        To start the bot, run:
        ```bash
        python -m src.bot
        ```
        
        The dashboard will automatically show data once the bot is running.
        """)
        
        # Show placeholder data for demo
        st.markdown("---")
        st.markdown("### 📊 Demo Mode (no live data)")
        
        # Create demo state
        demo_state = {
            "bot_status": "stopped",
            "bot_mode": "dry_run",
            "uptime_seconds": 0,
            "markets_tracked": 0,
            "signals_today": 0,
            "markets": [],
            "recent_signals": [],
            "positions": [],
            "performance": {
                "daily_pnl": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "current_capital": 75.0,
                "initial_capital": 75.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "best_trade": 0,
                "worst_trade": 0
            },
            "risk": {
                "status": "ok",
                "status_message": "Bot is stopped",
                "can_trade": False,
                "daily_pnl": 0,
                "daily_limit": 2.0,
                "remaining_risk": 2.0,
                "consecutive_losses": 0,
                "circuit_breaker_active": False,
                "circuit_breaker_remaining": 0
            },
            "recent_events": []
        }
        state = demo_state
    
    # Header
    render_header(state)
    
    st.markdown("---")
    
    # Main content in two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance
        render_performance(state)
        
        st.markdown("---")
        
        # Markets table
        render_markets(state)
        
        st.markdown("---")
        
        # Price chart
        render_price_chart(state)
    
    with col2:
        # Risk status
        render_risk_status(state)
        
        st.markdown("---")
        
        # Signals
        render_signals(state)
        
        st.markdown("---")
        
        # Positions
        render_positions(state)
        
        st.markdown("---")
        
        # Events
        render_events(state)
    
    # Auto-refresh
    st.empty()
    import time
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()