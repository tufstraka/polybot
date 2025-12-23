"""
Risk Manager - Protects your capital with strict limits.

Plain English Explanation:
==========================

Think of this as your "safety officer" that prevents you from losing too much money.
It enforces several key rules:

1. DAILY LOSS LIMIT: "Stop trading if you lose $2 today"
   - Tracks all your wins and losses for the day
   - Automatically stops the bot when you hit your daily limit
   - Resets at midnight so you can trade again tomorrow

2. POSITION SIZING: "Only risk $2 per trade"
   - Never put more than a small amount on any single trade
   - Keeps you in the game even if several trades go wrong

3. CIRCUIT BREAKER: "Take a break after too many losses"
   - If you lose 3 trades in a row, pause for a while
   - Prevents emotional/momentum-based losing streaks

4. CAPITAL PROTECTION: "Don't bet more than you can afford"
   - Tracks your total account balance
   - Reduces position sizes when capital gets low
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RiskStatus(Enum):
    """Current risk status of the bot."""
    OK = "ok"                          # All good, can trade
    WARNING = "warning"                # Getting close to limits
    DAILY_LIMIT_HIT = "daily_limit"   # Stop trading today
    CIRCUIT_BREAKER = "circuit_breaker"  # Too many consecutive losses
    LOW_CAPITAL = "low_capital"        # Account balance too low


@dataclass
class DailyStats:
    """
    Tracks today's trading performance.
    
    Plain English: This is your daily scorecard.
    """
    date: str                          # Today's date (YYYY-MM-DD)
    trades_opened: int = 0             # How many trades you started
    trades_closed: int = 0             # How many trades you finished
    wins: int = 0                      # Number of winning trades
    losses: int = 0                    # Number of losing trades
    realized_pnl: float = 0.0          # Actual money made/lost today
    consecutive_losses: int = 0        # Losses in a row (resets on a win)
    max_drawdown: float = 0.0          # Biggest dip from peak today


@dataclass
class RiskState:
    """
    Complete state of the risk manager.
    
    This gets saved to disk so we can recover after restarts.
    """
    current_capital: float             # How much money you have now
    initial_capital: float             # How much you started with
    daily_stats: DailyStats            # Today's performance
    total_realized_pnl: float = 0.0    # All-time profit/loss
    total_trades: int = 0              # All-time trade count
    status: str = "ok"                 # Current risk status
    last_updated: str = ""             # When this was last saved


class RiskManager:
    """
    Manages trading risk and protects your capital.
    
    Usage:
        risk_mgr = RiskManager(
            initial_capital=75.0,
            max_daily_loss=2.0,
            max_position_size=2.0,
            min_position_size=0.5
        )
        
        # Before every trade:
        if risk_mgr.can_open_trade():
            size = risk_mgr.calculate_position_size(confidence=0.8)
            # ... execute trade ...
        
        # After trade closes:
        risk_mgr.record_trade_result(profit_or_loss=0.15)
    """
    
    def __init__(
        self,
        initial_capital: float = 75.0,
        max_daily_loss: float = 2.0,
        max_position_size: float = 2.0,
        min_position_size: float = 0.5,
        max_consecutive_losses: int = 3,
        circuit_breaker_minutes: int = 30,
        min_capital_ratio: float = 0.1,
        state_file: str = "data/risk_state.json"
    ):
        """
        Initialize the risk manager.
        
        Args:
            initial_capital: Starting account balance ($75)
            max_daily_loss: Maximum loss allowed per day ($2)
            max_position_size: Biggest single trade size ($2)
            min_position_size: Smallest trade size ($0.50)
            max_consecutive_losses: Losses in a row before pause (3)
            circuit_breaker_minutes: How long to pause after streak (30)
            min_capital_ratio: Minimum capital before stopping (10%)
            state_file: Where to save risk state
        """
        self.initial_capital = initial_capital
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_consecutive_losses = max_consecutive_losses
        self.circuit_breaker_minutes = circuit_breaker_minutes
        self.min_capital_ratio = min_capital_ratio
        self.state_file = Path(state_file)
        
        # Track when circuit breaker was triggered
        self._circuit_breaker_time: Optional[datetime] = None
        
        # Load or initialize state
        self.state = self._load_state()
        
        logger.info(
            f"Risk Manager initialized: "
            f"capital=${self.state.current_capital:.2f}, "
            f"daily_limit=${max_daily_loss:.2f}, "
            f"max_position=${max_position_size:.2f}"
        )
    
    def _load_state(self) -> RiskState:
        """Load risk state from disk or create new."""
        today = date.today().isoformat()
        
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                
                # Parse daily stats
                daily_data = data.get("daily_stats", {})
                daily_stats = DailyStats(**daily_data)
                
                # Check if it's a new day - reset daily stats
                if daily_stats.date != today:
                    logger.info(f"New day detected, resetting daily stats")
                    daily_stats = DailyStats(date=today)
                
                state = RiskState(
                    current_capital=data.get("current_capital", self.initial_capital),
                    initial_capital=data.get("initial_capital", self.initial_capital),
                    daily_stats=daily_stats,
                    total_realized_pnl=data.get("total_realized_pnl", 0.0),
                    total_trades=data.get("total_trades", 0),
                    status=data.get("status", "ok"),
                    last_updated=data.get("last_updated", "")
                )
                
                logger.info(f"Loaded risk state: {state.current_capital:.2f} capital")
                return state
                
            except Exception as e:
                logger.warning(f"Failed to load risk state: {e}")
        
        # Create new state
        return RiskState(
            current_capital=self.initial_capital,
            initial_capital=self.initial_capital,
            daily_stats=DailyStats(date=today),
            last_updated=datetime.utcnow().isoformat()
        )
    
    def _save_state(self):
        """Save risk state to disk."""
        self.state.last_updated = datetime.utcnow().isoformat()
        
        # Convert to serializable format
        data = {
            "current_capital": self.state.current_capital,
            "initial_capital": self.state.initial_capital,
            "daily_stats": asdict(self.state.daily_stats),
            "total_realized_pnl": self.state.total_realized_pnl,
            "total_trades": self.state.total_trades,
            "status": self.state.status,
            "last_updated": self.state.last_updated
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _check_new_day(self):
        """Check if it's a new day and reset daily stats if needed."""
        today = date.today().isoformat()
        if self.state.daily_stats.date != today:
            logger.info(f"New trading day: {today}")
            self.state.daily_stats = DailyStats(date=today)
            self._circuit_breaker_time = None  # Reset circuit breaker
            self._save_state()
    
    def get_status(self) -> RiskStatus:
        """
        Get current risk status.
        
        Returns what state the risk manager is in right now.
        """
        self._check_new_day()
        
        stats = self.state.daily_stats
        
        # Check circuit breaker first (has timeout)
        if self._circuit_breaker_time:
            elapsed = (datetime.utcnow() - self._circuit_breaker_time).seconds / 60
            if elapsed < self.circuit_breaker_minutes:
                return RiskStatus.CIRCUIT_BREAKER
            else:
                # Circuit breaker expired
                self._circuit_breaker_time = None
                logger.info("Circuit breaker expired, resuming trading")
        
        # Check daily loss limit
        if stats.realized_pnl <= -self.max_daily_loss:
            return RiskStatus.DAILY_LIMIT_HIT
        
        # Check consecutive losses
        if stats.consecutive_losses >= self.max_consecutive_losses:
            self._circuit_breaker_time = datetime.utcnow()
            logger.warning(
                f"Circuit breaker triggered: {stats.consecutive_losses} consecutive losses"
            )
            return RiskStatus.CIRCUIT_BREAKER
        
        # Check minimum capital
        min_capital = self.initial_capital * self.min_capital_ratio
        if self.state.current_capital < min_capital:
            return RiskStatus.LOW_CAPITAL
        
        # Check if close to daily limit (warning)
        warning_threshold = self.max_daily_loss * 0.7  # 70% of limit
        if stats.realized_pnl <= -warning_threshold:
            return RiskStatus.WARNING
        
        return RiskStatus.OK
    
    def can_open_trade(self) -> bool:
        """
        Check if we're allowed to open a new trade.
        
        Plain English: "Is it safe to trade right now?"
        
        Returns True if:
        - We haven't hit the daily loss limit
        - Circuit breaker isn't active
        - We have enough capital
        """
        status = self.get_status()
        
        if status == RiskStatus.OK or status == RiskStatus.WARNING:
            return True
        
        if status == RiskStatus.DAILY_LIMIT_HIT:
            logger.warning("Cannot trade: Daily loss limit reached")
        elif status == RiskStatus.CIRCUIT_BREAKER:
            logger.warning("Cannot trade: Circuit breaker active")
        elif status == RiskStatus.LOW_CAPITAL:
            logger.warning("Cannot trade: Capital too low")
        
        return False
    
    def calculate_position_size(
        self,
        confidence: float = 1.0,
        volatility_factor: float = 1.0
    ) -> float:
        """
        Calculate safe position size based on current conditions.
        
        Plain English: "How much should I bet on this trade?"
        
        Args:
            confidence: How confident is the signal? (0.0 to 1.0)
                       Higher confidence = bigger position
            volatility_factor: How volatile is the market? (0.5 to 2.0)
                              Higher volatility = smaller position (for safety)
        
        Returns:
            Dollar amount to trade (between min and max position size)
        
        Example:
            - Base position: $2.00
            - Confidence 0.5: $1.00
            - Confidence 0.5 + high volatility: $0.50
        """
        # Start with max position
        base_size = self.max_position_size
        
        # Scale by confidence
        confidence_adjusted = base_size * max(0.25, min(1.0, confidence))
        
        # Scale by inverse of volatility (higher vol = smaller size)
        vol_multiplier = 1.0 / max(0.5, min(2.0, volatility_factor))
        vol_adjusted = confidence_adjusted * vol_multiplier
        
        # Capital-based scaling: reduce size if capital is low
        capital_ratio = self.state.current_capital / self.initial_capital
        if capital_ratio < 0.5:
            # Below 50% capital, reduce position size proportionally
            capital_multiplier = max(0.25, capital_ratio * 2)
            vol_adjusted *= capital_multiplier
        
        # Risk-per-trade scaling: reduce size if close to daily limit
        remaining_risk = self.max_daily_loss + self.state.daily_stats.realized_pnl
        if remaining_risk < self.max_position_size:
            # Can't risk more than remaining daily limit
            vol_adjusted = min(vol_adjusted, remaining_risk)
        
        # Clamp to min/max
        final_size = max(self.min_position_size, min(self.max_position_size, vol_adjusted))
        
        logger.debug(
            f"Position size: base=${base_size:.2f}, "
            f"conf_adj=${confidence_adjusted:.2f}, "
            f"vol_adj=${vol_adjusted:.2f}, "
            f"final=${final_size:.2f}"
        )
        
        return round(final_size, 2)
    
    def record_trade_opened(self):
        """
        Record that a new trade was opened.
        
        Call this when you enter a position.
        """
        self._check_new_day()
        self.state.daily_stats.trades_opened += 1
        self._save_state()
        
        logger.info(f"Trade opened (#{self.state.daily_stats.trades_opened} today)")
    
    def record_trade_result(self, pnl: float, is_win: bool = None):
        """
        Record the result of a closed trade.
        
        Plain English: "Tell the risk manager how the trade went"
        
        Args:
            pnl: Profit (positive) or loss (negative) in dollars
            is_win: Force win/loss classification (auto-detected if None)
        
        Example:
            risk_mgr.record_trade_result(0.15)   # Won $0.15
            risk_mgr.record_trade_result(-0.50)  # Lost $0.50
        """
        self._check_new_day()
        
        stats = self.state.daily_stats
        
        # Auto-detect win/loss if not specified
        if is_win is None:
            is_win = pnl > 0
        
        # Update daily stats
        stats.trades_closed += 1
        stats.realized_pnl += pnl
        
        if is_win:
            stats.wins += 1
            stats.consecutive_losses = 0  # Reset losing streak
            logger.info(f"Trade WIN: +${pnl:.2f}")
        else:
            stats.losses += 1
            stats.consecutive_losses += 1
            logger.info(f"Trade LOSS: -${abs(pnl):.2f} ({stats.consecutive_losses} in a row)")
        
        # Track max drawdown
        if stats.realized_pnl < stats.max_drawdown:
            stats.max_drawdown = stats.realized_pnl
        
        # Update overall stats
        self.state.current_capital += pnl
        self.state.total_realized_pnl += pnl
        self.state.total_trades += 1
        self.state.status = self.get_status().value
        
        self._save_state()
        
        logger.info(
            f"Daily P&L: ${stats.realized_pnl:.2f} | "
            f"Capital: ${self.state.current_capital:.2f} | "
            f"W/L: {stats.wins}/{stats.losses}"
        )
    
    def get_daily_stats(self) -> DailyStats:
        """Get today's trading statistics."""
        self._check_new_day()
        return self.state.daily_stats
    
    def get_summary(self) -> dict:
        """
        Get a complete summary of risk status.
        
        Returns a dictionary with all relevant info for display.
        """
        self._check_new_day()
        
        stats = self.state.daily_stats
        status = self.get_status()
        
        # Calculate win rate
        total_closed = stats.wins + stats.losses
        win_rate = (stats.wins / total_closed * 100) if total_closed > 0 else 0.0
        
        # Calculate remaining daily risk
        remaining_daily_risk = self.max_daily_loss + stats.realized_pnl
        
        # Circuit breaker info
        circuit_breaker_remaining = 0
        if self._circuit_breaker_time:
            elapsed = (datetime.utcnow() - self._circuit_breaker_time).seconds / 60
            circuit_breaker_remaining = max(0, self.circuit_breaker_minutes - elapsed)
        
        return {
            "status": status.value,
            "status_description": self._get_status_description(status),
            "can_trade": self.can_open_trade(),
            
            # Capital
            "current_capital": self.state.current_capital,
            "initial_capital": self.state.initial_capital,
            "capital_change_pct": ((self.state.current_capital - self.state.initial_capital) 
                                   / self.state.initial_capital * 100),
            
            # Daily stats
            "daily_pnl": stats.realized_pnl,
            "daily_limit": self.max_daily_loss,
            "remaining_daily_risk": remaining_daily_risk,
            "daily_trades": stats.trades_closed,
            "daily_wins": stats.wins,
            "daily_losses": stats.losses,
            "daily_win_rate": win_rate,
            "consecutive_losses": stats.consecutive_losses,
            "max_drawdown": stats.max_drawdown,
            
            # All-time stats
            "total_pnl": self.state.total_realized_pnl,
            "total_trades": self.state.total_trades,
            
            # Position sizing
            "max_position_size": self.max_position_size,
            "recommended_position": self.calculate_position_size(),
            
            # Circuit breaker
            "circuit_breaker_active": status == RiskStatus.CIRCUIT_BREAKER,
            "circuit_breaker_minutes_remaining": round(circuit_breaker_remaining, 1),
            
            "last_updated": self.state.last_updated
        }
    
    def _get_status_description(self, status: RiskStatus) -> str:
        """Get human-readable description of status."""
        descriptions = {
            RiskStatus.OK: "All systems go - safe to trade",
            RiskStatus.WARNING: "Approaching daily limit - trade carefully",
            RiskStatus.DAILY_LIMIT_HIT: "Daily loss limit reached - no more trades today",
            RiskStatus.CIRCUIT_BREAKER: "Cooling off after losing streak - wait for reset",
            RiskStatus.LOW_CAPITAL: "Account balance too low - deposit more or reduce risk"
        }
        return descriptions.get(status, "Unknown status")
    
    def reset_daily_stats(self):
        """
        Force reset daily stats (for testing or manual override).
        
        Warning: This clears today's trading history!
        """
        logger.warning("Manually resetting daily stats")
        self.state.daily_stats = DailyStats(date=date.today().isoformat())
        self._circuit_breaker_time = None
        self._save_state()
    
    def update_capital(self, new_capital: float):
        """
        Manually update current capital (e.g., after deposit/withdrawal).
        
        Args:
            new_capital: New account balance
        """
        old_capital = self.state.current_capital
        self.state.current_capital = new_capital
        self._save_state()
        
        logger.info(f"Capital updated: ${old_capital:.2f} -> ${new_capital:.2f}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Create risk manager with user's settings
    rm = RiskManager(
        initial_capital=75.0,
        max_daily_loss=2.0,
        max_position_size=2.0,
        min_position_size=0.5,
        state_file="test_risk_state.json"
    )
    
    print("\n=== Risk Manager Demo ===\n")
    
    # Check initial status
    print(f"Status: {rm.get_status().value}")
    print(f"Can trade: {rm.can_open_trade()}")
    print(f"Recommended position: ${rm.calculate_position_size():.2f}")
    
    # Simulate some trades
    print("\n--- Simulating trades ---")
    
    # Win a trade
    rm.record_trade_opened()
    rm.record_trade_result(0.20)  # Won $0.20
    
    # Lose a trade
    rm.record_trade_opened()
    rm.record_trade_result(-0.30)  # Lost $0.30
    
    # Another loss
    rm.record_trade_opened()
    rm.record_trade_result(-0.50)  # Lost $0.50
    
    # Print summary
    print("\n--- Summary ---")
    summary = rm.get_summary()
    print(f"Status: {summary['status']} - {summary['status_description']}")
    print(f"Daily P&L: ${summary['daily_pnl']:.2f}")
    print(f"Remaining risk: ${summary['remaining_daily_risk']:.2f}")
    print(f"Win rate: {summary['daily_win_rate']:.1f}%")
    print(f"Consecutive losses: {summary['consecutive_losses']}")
    print(f"Can trade: {summary['can_trade']}")
    
    # Test circuit breaker
    print("\n--- Testing circuit breaker ---")
    rm.record_trade_result(-0.30)  # 2nd consecutive loss
    rm.record_trade_result(-0.30)  # 3rd consecutive loss - triggers circuit breaker
    
    print(f"Status after 3 losses: {rm.get_status().value}")
    print(f"Can trade: {rm.can_open_trade()}")
    
    # Clean up test file
    import os
    if os.path.exists("test_risk_state.json"):
        os.remove("test_risk_state.json")
    
    print("\n✅ Risk Manager test complete")