
"""
Polybot AI Dashboard - Modern Trading Dashboard
===============================================

A sleek, modern dashboard for monitoring your AI-powered trading bot.
Features glassmorphism design, real-time charts, AI reasoning display,
and intuitive UX.

Run with:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.state_writer import StateReader

# Try to import AI components
try:
    from src.ai.reasoning_tracker import get_reasoning_tracker
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Polybot | Spike Hunter",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# MODERN CSS STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255, 255, 255, 0.03);
        --bg-card-hover: rgba(255, 255, 255, 0.06);
        --border-color: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-muted: #666666;
        --accent-green: #00d4aa;
        --accent-red: #ff4757;
        --accent-blue: #3b82f6;
        --accent-yellow: #fbbf24;
        --accent-purple: #8b5cf6;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0a0a0f 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .modern-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, #00d4aa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 8px 0;
    }
    .stat-label {
        font-size: 12px;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .stat-delta {
        font-size: 13px;
        font-weight: 500;
        margin-top: 8px;
        color: #666;
    }
    .stat-delta.positive { color: #00d4aa; }
    .stat-delta.negative { color: #ff4757; }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 100px;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .status-running {
        background: rgba(0, 212, 170, 0.15);
        color: #00d4aa;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .status-stopped {
        background: rgba(255, 71, 87, 0.15);
        color: #ff4757;
        border: 1px solid rgba(255, 71, 87, 0.3);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .section-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    
    .progress-container {
        background: #12121a;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 8px;
    }
    .progress-green { background: linear-gradient(90deg, #00d4aa, #00ff88); }
    .progress-red { background: linear-gradient(90deg, #ff4757, #ff6b6b); }
    .progress-blue { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    
    .position-row {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
        align-items: center;
        gap: 16px;
    }
    
    .signal-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 32px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #a0a0a0;
        margin: 0;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
    }
    
    .dataframe th {
        background: #12121a !important;
        color: #a0a0a0 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 11px !important;
    }
    .dataframe td {
        background: transparent !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_state():
    """Load current bot state from shared file."""
    try:
        reader = StateReader("data/bot_state.json")
        return reader.read()
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_money(value: float, include_sign: bool = False) -> str:
    """Format money value."""
    if include_sign and value >= 0:
        return f"+${value:.2f}"
    elif value < 0:
        return f"-${abs(value):.2f}"
    return f"${value:.2f}"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_hero_header(state: dict):
    """Render the hero header with status."""
    status = state.get("bot_status", "stopped")
    mode = state.get("bot_mode", "dry_run")
    ai_enabled = state.get("ai_enabled", False)
    
    status_class = "status-running" if status == "running" else "status-stopped"
    status_dot = "●" if status == "running" else "○"
    mode_text = "Paper Trading" if mode == "dry_run" else "Live Trading"
    ai_badge = "🤖 AI Enabled" if ai_enabled else ""
    
    st.markdown(f"""
    <div class="hero-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 class="hero-title">⚡ Polybot AI Trading</h1>
                <p class="hero-subtitle">AI-powered autonomous trading with Amazon Bedrock</p>
            </div>
            <div style="text-align: right;">
                <span class="status-badge {status_class}">{status_dot} {status.upper()}</span>
                <p style="margin: 8px 0 0 0; font-size: 14px; color: #a0a0a0;">{mode_text} {ai_badge}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_key_metrics(state: dict):
    """Render the key metrics row."""
    perf = state.get("performance") or {}
    
    daily_pnl = perf.get("daily_pnl", 0) or 0
    total_pnl = perf.get("total_pnl", 0) or 0
    capital = perf.get("current_capital", 75) or 75
    initial = perf.get("initial_capital", 75) or 75
    win_rate = perf.get("win_rate", 0) or 0
    trades = perf.get("total_trades", 0) or 0
    markets = state.get("markets_tracked", 0)
    signals = state.get("signals_today", 0)
    
    return_pct = ((capital - initial) / initial * 100) if initial > 0 else 0
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        delta_class = "positive" if daily_pnl >= 0 else "negative"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Today's P&L</div>
            <div class="stat-value">{format_money(daily_pnl, True)}</div>
            <div class="stat-delta {delta_class}">Daily Result</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_class = "positive" if return_pct >= 0 else "negative"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Return</div>
            <div class="stat-value">{return_pct:+.1f}%</div>
            <div class="stat-delta {delta_class}">{format_money(total_pnl, True)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Win Rate</div>
            <div class="stat-value">{win_rate:.0f}%</div>
            <div class="stat-delta">{trades} trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Capital</div>
            <div class="stat-value">{format_money(capital)}</div>
            <div class="stat-delta">Started {format_money(initial)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Markets</div>
            <div class="stat-value">{markets}</div>
            <div class="stat-delta">Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Signals</div>
            <div class="stat-value">{signals}</div>
            <div class="stat-delta">Today</div>
        </div>
        """, unsafe_allow_html=True)


def render_risk_panel(state: dict):
    """Render the risk management panel using native Streamlit components."""
    risk = state.get("risk") or {}
    
    status = risk.get("status", "ok") or "ok"
    daily_pnl = risk.get("daily_pnl", 0) or 0
    daily_limit = risk.get("daily_limit", 2) or 2
    remaining = risk.get("remaining_risk", daily_limit) or daily_limit
    can_trade = risk.get("can_trade", True)
    consec = risk.get("consecutive_losses", 0) or 0
    cb_active = risk.get("circuit_breaker_active", False)
    
    risk_used = abs(daily_pnl) / daily_limit * 100 if daily_limit > 0 else 0
    risk_used_fraction = min(risk_used / 100, 1.0)
    
    status_info = {
        "ok": ("✅", "All Systems Go", "success"),
        "warning": ("⚠️", "Approaching Limits", "warning"),
        "daily_limit": ("🛑", "Daily Limit Reached", "error"),
        "circuit_breaker": ("🔴", "Circuit Breaker Active", "error"),
    }
    status_icon, status_msg, status_type = status_info.get(status, ("❓", "Unknown", "info"))
    
    with st.container():
        st.subheader("⚠️ Risk Management")
        
        # Status row
        col1, col2 = st.columns(2)
        with col1:
            if status_type == "success":
                st.success(f"{status_icon} {status_msg}")
            elif status_type == "warning":
                st.warning(f"{status_icon} {status_msg}")
            elif status_type == "error":
                st.error(f"{status_icon} {status_msg}")
            else:
                st.info(f"{status_icon} {status_msg}")
        
        with col2:
            if can_trade:
                st.success("✓ Trading Enabled")
            else:
                st.error("✕ Trading Disabled")
        
        # Progress bar for daily risk
        st.markdown("**Daily Risk Used**")
        st.progress(risk_used_fraction)
        st.caption(f"{format_money(abs(daily_pnl))} / {format_money(daily_limit)} ({risk_used:.1f}%)")
        
        # Metrics row
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Remaining", format_money(remaining))
        
        with col2:
            st.metric("Losing Streak", str(consec), delta=None if consec == 0 else f"-{consec}")
        
        with col3:
            cb_status = "🔴 ACTIVE" if cb_active else "🟢 OFF"
            st.metric("Circuit Breaker", cb_status)
        
        st.markdown("---")


def render_positions(state: dict):
    """Render open positions using native Streamlit components."""
    positions = state.get("positions", [])
    open_positions = [p for p in positions if p.get("status") == "open"]
    
    with st.container():
        st.subheader("💼 Open Positions")
        
        if not open_positions:
            st.info("📭 No open positions - Waiting for signals...")
        else:
            for pos in open_positions:
                pnl = pos.get("unrealized_pnl", 0)
                side_emoji = "🔼" if pos.get("side", "").lower() == "buy" else "🔽"
                name = pos.get("market_name", "Unknown")
                if len(name) > 30:
                    name = name[:30] + "..."
                
                with st.expander(f"{side_emoji} {name}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Entry", f"${pos.get('entry_price', 0):.3f}")
                    
                    with col2:
                        st.metric("Current", f"${pos.get('current_price', 0):.3f}")
                    
                    with col3:
                        st.metric("Size", f"${pos.get('size', 0):.2f}")
                    
                    with col4:
                        delta_color = "normal" if pnl >= 0 else "inverse"
                        st.metric("P&L", format_money(pnl, True), delta=f"{pnl/pos.get('size', 1)*100:.1f}%" if pos.get('size', 0) > 0 else None, delta_color=delta_color)
        
        st.markdown("---")


def render_signals(state: dict):
    """Render recent signals using native Streamlit components with pagination."""
    signals = state.get("recent_signals", [])
    
    with st.container():
        st.subheader("🎯 Recent Signals")
        
        if not signals:
            st.info("📡 No signals detected - Scanning for spikes...")
        else:
            # Pagination for signals
            signals_per_page = 10
            total_signals = len(signals)
            total_pages = max(1, (total_signals + signals_per_page - 1) // signals_per_page)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="signals_page"
                )
            with col2:
                st.caption(f"Page {current_page}/{total_pages} ({total_signals} signals)")
            
            # Calculate slice for current page (reverse order - newest first)
            reversed_signals = list(reversed(signals))
            start_idx = (current_page - 1) * signals_per_page
            end_idx = min(start_idx + signals_per_page, total_signals)
            page_signals = reversed_signals[start_idx:end_idx]
            
            for sig in page_signals:
                direction = sig.get("direction", "").lower()
                is_up = direction == "up" or direction == "buy"
                icon = "🚀" if is_up else "📉"
                
                timestamp = sig.get("detected_at", "")
                time_str = "?"
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except Exception:
                        pass
                
                name = sig.get("market_name", "Unknown")  # FULL name
                
                conf = sig.get("confidence", 0)
                price = sig.get("price", 0)
                status = sig.get("status", "detected")
                status_icon = {"detected": "🔵", "traded": "🟢", "expired": "⚪", "failed": "🔴"}.get(status, "⚪")
                trigger = sig.get("trigger_reason", "")
                
                with st.expander(f"{icon} {name}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Price:** ${price:.3f}")
                        st.markdown(f"**Confidence:** {conf:.0%}")
                        st.markdown(f"**Trigger:** {trigger}")
                    with col2:
                        st.markdown(f"**Time:** {time_str}")
                        st.markdown(f"**Status:** {status_icon} {status}")
        
        st.markdown("---")


def render_markets_table(state: dict):
    """Render tracked markets table with pagination."""
    markets = state.get("markets", [])
    
    st.markdown("""
    <div class="modern-card">
        <div class="section-header">
            <div class="section-icon">📈</div>
            <h3 class="section-title">Tracked Markets</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if not markets:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; color: #666;">
            <div style="font-size: 40px; margin-bottom: 12px;">📊</div>
            <div style="font-size: 15px;">No markets tracked</div>
            <div style="font-size: 13px; margin-top: 4px; color: #555;">Start the bot to begin scanning...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Pagination for markets
        markets_per_page = 25
        total_markets = len(markets)
        total_pages = max(1, (total_markets + markets_per_page - 1) // markets_per_page)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            current_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="markets_page"
            )
        with col2:
            st.caption(f"Showing page {current_page} of {total_pages} ({total_markets} total markets)")
        
        # Calculate slice for current page
        start_idx = (current_page - 1) * markets_per_page
        end_idx = min(start_idx + markets_per_page, total_markets)
        page_markets = markets[start_idx:end_idx]
        
        # Format columns for current page
        display_data = []
        for m in page_markets:
            name = m.get("name", "Unknown")  # Show FULL name
            
            display_data.append({
                "Market": name,
                "Price": f"${m.get('current_price', 0):.3f}",
                "EWMA": f"${m.get('ewma_price', 0):.3f}",
                "ROC": f"{m.get('roc', 0):+.2%}",
                "CUSUM+": f"{m.get('cusum_pos', 0):.4f}",
                "CUSUM-": f"{m.get('cusum_neg', 0):.4f}",
                "Volume 24h": f"${m.get('volume_24h', 0):,.0f}",
            })
        
        if display_data:
            st.dataframe(
                pd.DataFrame(display_data),
                use_container_width=True,
                hide_index=True,
                height=400  # Scrollable table
            )
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_price_chart(state: dict):
    """Render price chart with EWMA bands."""
    markets = state.get("markets", [])
    
    st.markdown("""
    <div class="modern-card">
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h3 class="section-title">Price & Indicators</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if not markets:
        st.info("No market data available")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Market selector
    market_names = [m.get("name", m.get("market_id", "Unknown"))[:50] for m in markets]
    selected = st.selectbox("Select Market", market_names, key="market_select")
    
    idx = market_names.index(selected) if selected in market_names else 0
    market = markets[idx]
    
    # Get values
    price = market.get("current_price", 0.5)
    ewma = market.get("ewma_price", 0.5)
    upper = market.get("ewma_upper", 0.55)
    lower = market.get("ewma_lower", 0.45)
    cusum_pos = market.get("cusum_pos", 0)
    cusum_neg = market.get("cusum_neg", 0)
    roc = market.get("roc", 0)
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
        subplot_titles=("Price & EWMA Bands", "CUSUM Indicator")
    )
    
    # Price visualization
    now = datetime.now()
    x_vals = [now]
    
    # Add bands as area
    fig.add_trace(go.Scatter(
        x=x_vals, y=[upper], mode="markers+text",
        name="Upper Band", marker=dict(size=12, color="#00d4aa", symbol="triangle-up"),
        text=[f"Upper: ${upper:.3f}"], textposition="top center", textfont=dict(size=10, color="#00d4aa")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=[lower], mode="markers+text",
        name="Lower Band", marker=dict(size=12, color="#ff4757", symbol="triangle-down"),
        text=[f"Lower: ${lower:.3f}"], textposition="bottom center", textfont=dict(size=10, color="#ff4757")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=[ewma], mode="markers+text",
        name="EWMA", marker=dict(size=14, color="#fbbf24"),
        text=[f"EWMA: ${ewma:.3f}"], textposition="middle right", textfont=dict(size=10, color="#fbbf24")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=[price], mode="markers+text",
        name="Current Price", marker=dict(size=20, color="#ffffff", line=dict(color="#3b82f6", width=3)),
        text=[f"Price: ${price:.3f}"], textposition="top center", textfont=dict(size=12, color="#ffffff")
    ), row=1, col=1)
    
    # CUSUM bars
    fig.add_trace(go.Bar(
        x=["CUSUM+", "CUSUM-"], y=[cusum_pos, abs(cusum_neg)],
        marker_color=["#00d4aa", "#ff4757"], name="CUSUM",
        text=[f"{cusum_pos:.4f}", f"{cusum_neg:.4f}"], textposition="outside"
    ), row=2, col=1)
    
    fig.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics below chart
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price", f"${price:.3f}")
    with col2:
        st.metric("EWMA", f"${ewma:.3f}")
    with col3:
        st.metric("ROC", f"{roc:+.2%}")
    with col4:
        st.metric("Volume", f"${market.get('volume_24h', 0):,.0f}")
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_activity_log(state: dict):
    """Render activity log using native Streamlit components."""
    events = state.get("recent_events", [])
    
    with st.container():
        st.subheader("📝 Activity Log")
        
        if not events:
            st.info("No recent activity")
        else:
            log_entries = []
            for event in events[-10:][::-1]:
                event_type = event.get("type", "info")
                message = event.get("message", "")
                timestamp = event.get("timestamp", "")
                
                time_str = "?"
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except Exception:
                        pass
                
                emoji = {"signal": "🎯", "trade": "💰", "error": "🚨", "warning": "⚠️", "system": "ℹ️"}.get(event_type, "📌")
                log_entries.append({"Time": time_str, "": emoji, "Event": message})
            
            if log_entries:
                df = pd.DataFrame(log_entries)
                st.dataframe(df, use_container_width=True, hide_index=True, height=300)


def render_ai_reasoning(state: dict):
    """Render AI reasoning and decision panel with full text and pagination."""
    with st.container():
        st.subheader("🤖 AI Decision Engine")
        
        # Check if AI is available
        if not AI_AVAILABLE:
            st.warning("AI module not available. Configure AWS Bedrock credentials to enable.")
            return
        
        # Get AI stats from state
        ai_stats = state.get("ai_stats", {})
        
        if not ai_stats:
            st.info("🧠 AI analysis not yet active. Start the bot with AI enabled.")
            return
        
        # AI Status Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model = ai_stats.get("model", "claude-3-sonnet")
            st.metric("Model", model)
        
        with col2:
            decisions = ai_stats.get("decisions_today", 0)
            st.metric("Decisions Today", str(decisions))
        
        with col3:
            avg_latency = ai_stats.get("avg_latency_ms", 0)
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        
        st.markdown("---")
        
        # Recent AI Decisions with pagination
        st.markdown("**AI Reasoning History**")
        
        reasoning_entries = state.get("ai_reasoning", [])
        
        if not reasoning_entries:
            # Try to load from reasoning tracker
            try:
                tracker = get_reasoning_tracker()
                reasoning_entries = tracker.get_for_dashboard()
            except Exception:
                reasoning_entries = []
        
        if not reasoning_entries:
            st.info("No AI decisions recorded yet.")
        else:
            # Pagination settings
            entries_per_page = 10
            total_entries = len(reasoning_entries)
            total_pages = max(1, (total_entries + entries_per_page - 1) // entries_per_page)
            
            # Page selector
            col1, col2 = st.columns([1, 3])
            with col1:
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="ai_reasoning_page"
                )
            with col2:
                st.caption(f"Showing page {current_page} of {total_pages} ({total_entries} total entries)")
            
            # Calculate slice for current page
            start_idx = (current_page - 1) * entries_per_page
            end_idx = min(start_idx + entries_per_page, total_entries)
            page_entries = reasoning_entries[start_idx:end_idx]
            
            for entry in page_entries:
                action = entry.get("action", "HOLD")
                confidence = entry.get("confidence", "0%")
                market = entry.get("market", "Unknown")
                reasoning = entry.get("reasoning", "No reasoning available")  # FULL TEXT
                time_str = entry.get("time", "?")
                outcome = entry.get("outcome", "pending")
                pnl = entry.get("pnl", "-")
                
                # Color code action
                action_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
                action_icon = action_colors.get(action, "⚪")
                
                # Outcome colors
                outcome_colors = {
                    "pending": "🔵",
                    "executed": "🟡",
                    "profitable": "🟢",
                    "unprofitable": "🔴",
                    "breakeven": "⚪"
                }
                outcome_icon = outcome_colors.get(outcome, "⚪")
                
                # Show full market name and reasoning
                with st.expander(f"{action_icon} {action} - {market} ({confidence})", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Market:** {market}")
                        st.markdown(f"**Action:** {action_icon} {action}")
                        st.markdown(f"**Confidence:** {confidence}")
                    
                    with col2:
                        st.markdown(f"**Time:** {time_str}")
                        st.markdown(f"**Outcome:** {outcome_icon} {outcome}")
                        if pnl != "-":
                            st.markdown(f"**P&L:** {pnl}")
                    
                    st.markdown("---")
                    st.markdown("**📝 Full AI Reasoning:**")
                    
                    # Show full reasoning text in a scrollable container
                    if reasoning and len(reasoning) > 0:
                        st.text_area(
                            "AI Analysis",
                            value=reasoning,
                            height=300,  # Larger area for full text
                            disabled=True,
                            key=f"reasoning_{time_str}_{hash(market) % 10000}"
                        )
                    else:
                        st.info("No reasoning text available")
                    
                    # Show additional details in a collapsed section
                    with st.expander("📊 Technical Details", expanded=False):
                        entry_price = entry.get("entry_price", 0) or 0
                        stop_loss = entry.get("stop_loss", 0) or 0
                        take_profit = entry.get("take_profit", 0) or 0
                        position_size = entry.get("position_size", 0) or 0
                        # Handle both formats: tokens_used from state or input_tokens+output_tokens from tracker
                        tokens = entry.get("tokens_used", 0) or (entry.get("input_tokens", 0) + entry.get("output_tokens", 0))
                        latency = entry.get("latency_ms", 0) or 0
                        
                        tech_col1, tech_col2 = st.columns(2)
                        with tech_col1:
                            st.markdown(f"- **Entry Price:** ${float(entry_price):.4f}")
                            st.markdown(f"- **Stop Loss:** ${float(stop_loss):.4f}")
                            st.markdown(f"- **Take Profit:** ${float(take_profit):.4f}")
                        with tech_col2:
                            st.markdown(f"- **Position Size:** ${float(position_size):.2f}")
                            st.markdown(f"- **Tokens Used:** {int(tokens)}")
                            st.markdown(f"- **AI Latency:** {float(latency):.0f}ms")
        
        st.markdown("---")
        
        # AI Performance Summary
        if ai_stats:
            st.markdown("**AI Performance**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                win_rate = ai_stats.get("win_rate", 0)
                st.metric("Win Rate", f"{win_rate:.0%}")
            
            with col2:
                profitable = ai_stats.get("profitable_trades", 0)
                st.metric("Profitable", str(profitable))
            
            with col3:
                total_tokens = ai_stats.get("total_tokens", 0)
                st.metric("Tokens Used", f"{total_tokens:,}")
            
            with col4:
                avg_conf = ai_stats.get("avg_confidence", 0)
                st.metric("Avg Confidence", f"{avg_conf:.0%}")


def render_monte_carlo(state: dict):
    """Render Monte Carlo simulation results."""
    with st.container():
        st.subheader("📊 Risk Analysis (Monte Carlo)")
        
        mc_results = state.get("monte_carlo", {})
        
        if not mc_results:
            st.info("No Monte Carlo simulation results available.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prob_profit = mc_results.get("prob_profit", 0)
            st.metric("P(Profit)", f"{prob_profit:.0%}")
        
        with col2:
            var_95 = mc_results.get("var_95", 0)
            st.metric("VaR (95%)", f"${abs(var_95):.2f}")
        
        with col3:
            risk_assessment = mc_results.get("risk_assessment", "Unknown")
            st.metric("Risk Level", risk_assessment)
        
        # Risk distribution chart if data available
        if "distribution" in mc_results:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=mc_results["distribution"],
                nbinsx=50,
                marker_color="#3b82f6",
                opacity=0.7
            ))
            fig.update_layout(
                title="Return Distribution",
                xaxis_title="Return ($)",
                yaxis_title="Frequency",
                height=200,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main dashboard application."""
    # Load state
    state = load_state()
    
    if not state:
        # Demo state when bot not running
        state = {
            "bot_status": "stopped",
            "bot_mode": "dry_run",
            "uptime_seconds": 0,
            "markets_tracked": 0,
            "signals_today": 0,
            "markets": [],
            "recent_signals": [],
            "positions": [],
            "performance": {
                "daily_pnl": 0, "total_pnl": 0, "win_rate": 0,
                "current_capital": 75.0, "initial_capital": 75.0,
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0
            },
            "risk": {
                "status": "ok", "status_message": "Bot stopped",
                "can_trade": False, "daily_pnl": 0, "daily_limit": 2.0,
                "remaining_risk": 2.0, "consecutive_losses": 0,
                "circuit_breaker_active": False
            },
            "recent_events": []
        }
    
    # Render components
    render_hero_header(state)
    render_key_metrics(state)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["📈 Trading", "🤖 AI Analysis"])
    
    with tab1:
        # Main content - Trading view
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_markets_table(state)
            render_price_chart(state)
        
        with col2:
            render_risk_panel(state)
            render_positions(state)
            render_signals(state)
            render_activity_log(state)
    
    with tab2:
        # AI Analysis view
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_ai_reasoning(state)
        
        with col2:
            render_monte_carlo(state)
            
            # AI Stats Summary
            st.subheader("🧠 AI Stats")
            ai_stats = state.get("ai_stats", {})
            
            if ai_stats:
                st.json(ai_stats)
            else:
                st.info("AI stats will appear once the bot starts with AI enabled.")
    
    # Auto-refresh
    refresh_rate = st.sidebar.slider("Refresh (seconds)", 2, 30, 5, key="refresh")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"*Last update: {datetime.now().strftime('%H:%M:%S')}*")
    
    import time
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()