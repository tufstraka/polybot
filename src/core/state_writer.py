"""
Shared State Writer - Bridge between Bot and Dashboard.

Plain English Explanation:
==========================

The bot and dashboard are separate programs. They need a way to talk
to each other. This "state writer" is like a shared bulletin board:

1. The BOT writes updates to a JSON file every second
2. The DASHBOARD reads that same file to show you what's happening

The JSON file contains everything the dashboard needs:
- Current prices being tracked
- Active signals detected
- Open positions
- P&L and performance metrics
- Risk status

This is simpler and more reliable than using a database or 
network connection between the bot and dashboard.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from threading import Lock
import time

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """State of a single market being tracked."""
    market_id: str
    name: str
    current_price: float
    ewma_price: float           # Smoothed price
    ewma_upper: float           # Upper volatility band
    ewma_lower: float           # Lower volatility band
    roc: float                  # Rate of change (momentum)
    cusum_pos: float            # Cumulative sum (positive)
    cusum_neg: float            # Cumulative sum (negative)
    volume_24h: float
    last_updated: str


@dataclass
class SignalState:
    """A detected trading signal."""
    signal_id: str
    market_id: str
    market_name: str
    direction: str              # "up" or "down"
    price: float
    confidence: float
    detected_at: str
    trigger_reason: str         # What triggered the signal
    status: str = "detected"    # detected, traded, expired


@dataclass
class PositionState:
    """An open or recent position."""
    position_id: str
    market_id: str
    market_name: str
    side: str                   # "BUY" or "SELL"
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    opened_at: str
    status: str = "open"        # open, closed
    unrealized_pnl_pct: float = 0.0  # P&L as percentage
    exit_price: float = 0.0     # Price at close (for closed positions)
    exit_reason: str = ""       # stop_loss, take_profit, manual
    closed_at: str = ""         # When position was closed


@dataclass
class RiskState:
    """Current risk management status."""
    status: str                 # ok, warning, limit_hit, circuit_breaker
    status_message: str
    can_trade: bool
    daily_pnl: float
    daily_limit: float
    remaining_risk: float
    consecutive_losses: int
    circuit_breaker_active: bool
    circuit_breaker_remaining: float


@dataclass
class PerformanceState:
    """Trading performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    current_capital: float
    initial_capital: float
    max_drawdown: float
    best_trade: float
    worst_trade: float


@dataclass
class AIStatsState:
    """AI decision engine statistics."""
    model: str = "claude-3-sonnet"
    decisions_today: int = 0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    avg_confidence: float = 0.0
    win_rate: float = 0.0
    profitable_trades: int = 0


@dataclass
class AIReasoningEntry:
    """A single AI reasoning entry."""
    action: str                 # BUY, SELL, HOLD
    confidence: str             # "75%"
    market: str
    reasoning: str
    time: str
    outcome: str = "pending"    # pending, executed, profitable, unprofitable
    pnl: str = "-"
    # Technical details
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class MonteCarloState:
    """Monte Carlo simulation results."""
    prob_profit: float = 0.0
    var_95: float = 0.0
    risk_assessment: str = "Unknown"
    distribution: List[float] = field(default_factory=list)


@dataclass
class BotState:
    """
    Complete state of the bot.
    
    This is what gets written to the shared JSON file.
    The dashboard reads this to display everything.
    """
    # Bot info
    bot_status: str = "stopped"     # running, paused, stopped, error
    bot_mode: str = "dry_run"       # live, dry_run
    ai_enabled: bool = False        # AI decision engine status
    uptime_seconds: float = 0.0
    last_heartbeat: str = ""
    
    # Markets being tracked
    markets_tracked: int = 0
    markets: List[MarketState] = field(default_factory=list)
    
    # Signals
    signals_today: int = 0
    recent_signals: List[SignalState] = field(default_factory=list)
    
    # Positions
    open_positions: int = 0
    positions: List[PositionState] = field(default_factory=list)
    
    # Risk
    risk: Optional[RiskState] = None
    
    # Performance
    performance: Optional[PerformanceState] = None
    
    # AI Decision Engine
    ai_stats: Optional[AIStatsState] = None
    ai_reasoning: List[AIReasoningEntry] = field(default_factory=list)
    monte_carlo: Optional[MonteCarloState] = None
    
    # Recent logs/events
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Errors
    last_error: Optional[str] = None
    error_count: int = 0


class StateWriter:
    """
    Writes bot state to a JSON file for the dashboard.
    
    Usage:
        writer = StateWriter("data/bot_state.json")
        
        # Update individual parts
        writer.update_bot_status("running", "live")
        writer.update_market(market_state)
        writer.add_signal(signal_state)
        
        # Flush writes to disk
        writer.flush()  # Call periodically
    """
    
    def __init__(
        self,
        state_file: str = "data/bot_state.json",
        max_signals: int = 50,
        max_events: int = 100,
        auto_flush_interval: float = 1.0
    ):
        """
        Initialize state writer.
        
        Args:
            state_file: Path to the shared state file
            max_signals: Maximum recent signals to keep
            max_events: Maximum recent events to keep
            auto_flush_interval: Seconds between auto-flushes (0 to disable)
        """
        self.state_file = Path(state_file)
        self.max_signals = max_signals
        self.max_events = max_events
        self.auto_flush_interval = auto_flush_interval
        
        self._state = BotState()
        self._lock = Lock()
        self._dirty = False
        self._last_flush = time.time()
        self._start_time = time.time()
        
        # Create directory if needed
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize with existing state if available
        self._load_existing()
        
        logger.info(f"State writer initialized: {state_file}")
    
    def _load_existing(self):
        """Load existing state file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                # We don't fully restore, just note it existed
                logger.debug("Found existing state file")
            except Exception as e:
                logger.warning(f"Could not load existing state: {e}")
    
    def _to_dict(self) -> dict:
        """Convert state to dictionary for JSON serialization."""
        state_dict = {
            "bot_status": self._state.bot_status,
            "bot_mode": self._state.bot_mode,
            "ai_enabled": self._state.ai_enabled,
            "uptime_seconds": time.time() - self._start_time,
            "last_heartbeat": datetime.utcnow().isoformat(),
            "markets_tracked": self._state.markets_tracked,
            "markets": [asdict(m) for m in self._state.markets],
            "signals_today": self._state.signals_today,
            "recent_signals": [asdict(s) for s in self._state.recent_signals],
            "open_positions": self._state.open_positions,
            "positions": [asdict(p) for p in self._state.positions],
            "risk": asdict(self._state.risk) if self._state.risk else None,
            "performance": asdict(self._state.performance) if self._state.performance else None,
            "ai_stats": asdict(self._state.ai_stats) if self._state.ai_stats else None,
            "ai_reasoning": [asdict(r) for r in self._state.ai_reasoning],  # Keep ALL entries
            "monte_carlo": asdict(self._state.monte_carlo) if self._state.monte_carlo else None,
            "recent_events": self._state.recent_events[-self.max_events:],
            "last_error": self._state.last_error,
            "error_count": self._state.error_count
        }
        return state_dict
    
    def flush(self, force: bool = False):
        """
        Write current state to disk.
        
        Args:
            force: Write even if not dirty
        """
        with self._lock:
            # Check if we should flush
            now = time.time()
            if not force and not self._dirty:
                return
            
            if not force and self.auto_flush_interval > 0:
                if now - self._last_flush < self.auto_flush_interval:
                    return
            
            try:
                state_dict = self._to_dict()
                
                # Write atomically (write to temp, then rename)
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state_dict, f, indent=2)
                temp_file.rename(self.state_file)
                
                self._dirty = False
                self._last_flush = now
                
                logger.debug("State flushed to disk")
                
            except Exception as e:
                logger.error(f"Failed to flush state: {e}")
    
    def update_bot_status(self, status: str, mode: str = None):
        """
        Update overall bot status.
        
        Args:
            status: "running", "paused", "stopped", "error"
            mode: "live" or "dry_run"
        """
        with self._lock:
            self._state.bot_status = status
            if mode:
                self._state.bot_mode = mode
            self._dirty = True
    
    def update_market(self, market: MarketState):
        """
        Update or add a market's state.
        
        The dashboard shows this as a table of tracked markets
        with their current prices and indicator values.
        """
        with self._lock:
            # Find and update existing, or add new
            for i, m in enumerate(self._state.markets):
                if m.market_id == market.market_id:
                    self._state.markets[i] = market
                    break
            else:
                self._state.markets.append(market)
            
            self._state.markets_tracked = len(self._state.markets)
            self._dirty = True
    
    def remove_market(self, market_id: str):
        """Stop tracking a market."""
        with self._lock:
            self._state.markets = [
                m for m in self._state.markets
                if m.market_id != market_id
            ]
            self._state.markets_tracked = len(self._state.markets)
            self._dirty = True
    
    def add_signal(self, signal: SignalState):
        """
        Add a newly detected signal.
        
        Signals are shown in the dashboard's signal log,
        helping you see what the bot is detecting.
        """
        with self._lock:
            self._state.recent_signals.insert(0, signal)
            self._state.recent_signals = self._state.recent_signals[:self.max_signals]
            self._state.signals_today += 1
            self._dirty = True
    
    def update_signal_status(self, signal_id: str, status: str):
        """Update a signal's status (e.g., from 'detected' to 'traded')."""
        with self._lock:
            for signal in self._state.recent_signals:
                if signal.signal_id == signal_id:
                    signal.status = status
                    break
            self._dirty = True
    
    def update_position(self, position: PositionState):
        """
        Update or add a position.
        
        The dashboard shows open positions with their
        current P&L and exit levels.
        """
        with self._lock:
            # Find and update existing, or add new
            for i, p in enumerate(self._state.positions):
                if p.position_id == position.position_id:
                    self._state.positions[i] = position
                    break
            else:
                self._state.positions.append(position)
            
            # Count open positions
            self._state.open_positions = sum(
                1 for p in self._state.positions if p.status == "open"
            )
            self._dirty = True
    
    def close_position(
        self,
        position_id: str,
        exit_price: float = 0.0,
        exit_reason: str = "",
        realized_pnl: float = 0.0
    ):
        """
        Mark a position as closed with exit details.
        
        Args:
            position_id: ID of position to close
            exit_price: Price at which position was closed
            exit_reason: Reason for exit (stop_loss, take_profit, manual)
            realized_pnl: The realized profit/loss
        """
        with self._lock:
            for position in self._state.positions:
                if position.position_id == position_id:
                    position.status = "closed"
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    position.closed_at = datetime.utcnow().isoformat()
                    # Update unrealized to realized (for display)
                    if realized_pnl != 0:
                        position.unrealized_pnl = realized_pnl
                    break
            
            self._state.open_positions = sum(
                1 for p in self._state.positions if p.status == "open"
            )
            self._dirty = True
    
    def update_risk(self, risk: RiskState):
        """
        Update risk management state.
        
        The dashboard shows this as a risk status panel
        with daily limits and circuit breaker status.
        """
        with self._lock:
            self._state.risk = risk
            self._dirty = True
    
    def update_performance(self, performance: PerformanceState):
        """
        Update performance metrics.
        
        The dashboard shows this as your P&L card
        with win rate and trade counts.
        """
        with self._lock:
            self._state.performance = performance
            self._dirty = True
    
    def update_ai_enabled(self, enabled: bool):
        """Update AI enabled status."""
        with self._lock:
            self._state.ai_enabled = enabled
            self._dirty = True
    
    def update_ai_stats(self, stats: AIStatsState):
        """
        Update AI decision engine statistics.
        
        The dashboard shows this in the AI Analysis tab.
        """
        with self._lock:
            self._state.ai_stats = stats
            self._dirty = True
    
    def add_ai_reasoning(self, reasoning: AIReasoningEntry):
        """
        Add an AI reasoning entry.
        
        Shows the AI's decision-making process in the dashboard.
        """
        with self._lock:
            # Check if entry already exists (by time and market)
            existing = any(
                r.time == reasoning.time and r.market == reasoning.market
                for r in self._state.ai_reasoning
            )
            if not existing:
                self._state.ai_reasoning.append(reasoning)
            # Keep last 500 entries (enough for long sessions)
            self._state.ai_reasoning = self._state.ai_reasoning[-500:]
            self._dirty = True
    
    def update_monte_carlo(self, results: MonteCarloState):
        """
        Update Monte Carlo simulation results.
        
        The dashboard shows this as risk analysis charts.
        """
        with self._lock:
            self._state.monte_carlo = results
            self._dirty = True
    
    def add_event(self, event_type: str, message: str, details: dict = None):
        """
        Add an event to the log.
        
        Events are shown in the dashboard's activity feed.
        """
        event = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        with self._lock:
            self._state.recent_events.append(event)
            self._state.recent_events = self._state.recent_events[-self.max_events:]
            self._dirty = True
    
    def record_error(self, error: str):
        """Record an error that occurred."""
        with self._lock:
            self._state.last_error = error
            self._state.error_count += 1
            self._dirty = True
        
        self.add_event("error", error)
    
    def reset_daily_counters(self):
        """Reset counters for a new day."""
        with self._lock:
            self._state.signals_today = 0
            # Keep some recent signals but mark them as from yesterday
            self._dirty = True
        
        self.add_event("system", "Daily counters reset")
    
    def get_state(self) -> BotState:
        """Get current state (for internal use)."""
        with self._lock:
            return self._state
    
    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        with self._lock:
            return self._to_dict()


class StateReader:
    """
    Reads bot state from the shared JSON file.
    
    Used by the dashboard to get current bot state.
    
    Usage:
        reader = StateReader("data/bot_state.json")
        state = reader.read()
        print(f"Bot is {state['bot_status']}")
    """
    
    def __init__(self, state_file: str = "data/bot_state.json"):
        """Initialize state reader."""
        self.state_file = Path(state_file)
    
    def read(self) -> Optional[dict]:
        """
        Read current state from file.
        
        Returns None if file doesn't exist or is invalid.
        """
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read state: {e}")
            return None
    
    def is_bot_running(self) -> bool:
        """Check if bot appears to be running."""
        state = self.read()
        if not state:
            return False
        
        # Check if heartbeat is recent (within 10 seconds)
        try:
            heartbeat = datetime.fromisoformat(state.get("last_heartbeat", ""))
            age = (datetime.utcnow() - heartbeat).total_seconds()
            return state.get("bot_status") == "running" and age < 10
        except:
            return False
    
    def get_markets(self) -> List[dict]:
        """Get list of tracked markets."""
        state = self.read()
        return state.get("markets", []) if state else []
    
    def get_signals(self) -> List[dict]:
        """Get recent signals."""
        state = self.read()
        return state.get("recent_signals", []) if state else []
    
    def get_positions(self) -> List[dict]:
        """Get positions."""
        state = self.read()
        return state.get("positions", []) if state else []
    
    def get_risk(self) -> Optional[dict]:
        """Get risk state."""
        state = self.read()
        return state.get("risk") if state else None
    
    def get_performance(self) -> Optional[dict]:
        """Get performance metrics."""
        state = self.read()
        return state.get("performance") if state else None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test writer
    writer = StateWriter("test_state.json")
    
    # Update bot status
    writer.update_bot_status("running", "dry_run")
    
    # Add a market
    market = MarketState(
        market_id="test123",
        name="Trump wins election",
        current_price=0.65,
        ewma_price=0.64,
        ewma_upper=0.68,
        ewma_lower=0.60,
        roc=0.02,
        cusum_pos=0.5,
        cusum_neg=0.0,
        volume_24h=150000,
        last_updated=datetime.utcnow().isoformat()
    )
    writer.update_market(market)
    
    # Add a signal
    signal = SignalState(
        signal_id="sig001",
        market_id="test123",
        market_name="Trump wins election",
        direction="up",
        price=0.65,
        confidence=0.85,
        detected_at=datetime.utcnow().isoformat(),
        trigger_reason="EWMA band breakout + CUSUM trigger"
    )
    writer.add_signal(signal)
    
    # Add risk state
    risk = RiskState(
        status="ok",
        status_message="All systems go",
        can_trade=True,
        daily_pnl=0.25,
        daily_limit=2.0,
        remaining_risk=1.75,
        consecutive_losses=0,
        circuit_breaker_active=False,
        circuit_breaker_remaining=0
    )
    writer.update_risk(risk)
    
    # Add performance
    perf = PerformanceState(
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=60.0,
        total_pnl=1.25,
        daily_pnl=0.25,
        current_capital=76.25,
        initial_capital=75.0,
        max_drawdown=-0.50,
        best_trade=0.35,
        worst_trade=-0.25
    )
    writer.update_performance(perf)
    
    # Flush to disk
    writer.flush(force=True)
    
    print("State written to test_state.json")
    
    # Test reader
    reader = StateReader("test_state.json")
    state = reader.read()
    print(f"\nBot status: {state['bot_status']}")
    print(f"Markets tracked: {state['markets_tracked']}")
    print(f"Signals today: {state['signals_today']}")
    
    # Clean up
    import os
    os.remove("test_state.json")
    print("\n✅ State writer test complete")