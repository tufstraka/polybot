"""
Position Manager Module
=======================

Tracks open positions, calculates P&L, and manages exit logic
(stop-loss and take-profit).
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from enum import Enum
import logging

from src.config.settings import get_settings, Settings
from src.core.client import Side
from src.core.detector import TradingSignal, SignalDirection
from src.core.executor import ExecutionResult, ExecutionStatus


logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    """Status of a position."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Position:
    """
    Represents an open or closed trading position.
    
    Tracks entry, exit, and P&L for a single trade.
    """
    id: str
    market_id: str
    token_id: str
    market_name: str
    
    # Direction
    side: SignalDirection
    
    # Entry details
    entry_price: float
    entry_size: float
    entry_time: datetime
    
    # Exit levels (adaptive)
    stop_loss: float
    take_profit: float
    
    # Current state
    status: PositionStatus = PositionStatus.OPEN
    current_price: float = 0.0
    
    # Exit details (if closed)
    exit_price: float = 0.0
    exit_size: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    # P&L tracking
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status == PositionStatus.OPEN
    
    @property
    def entry_value(self) -> float:
        """Total value at entry."""
        return self.entry_price * self.entry_size
    
    @property
    def current_value(self) -> float:
        """Current market value."""
        return self.current_price * self.entry_size
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss (for open positions)."""
        if not self.is_open:
            return 0.0
        
        if self.side == SignalDirection.BUY:
            return (self.current_price - self.entry_price) * self.entry_size
        else:
            return (self.entry_price - self.current_price) * self.entry_size
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized P&L as percentage."""
        if self.entry_value == 0:
            return 0.0
        return (self.unrealized_pnl / self.entry_value) * 100
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized - fees)."""
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid
    
    def should_stop_loss(self) -> bool:
        """Check if stop-loss should trigger."""
        if not self.is_open:
            return False
        
        if self.side == SignalDirection.BUY:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Check if take-profit should trigger."""
        if not self.is_open:
            return False
        
        if self.side == SignalDirection.BUY:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit
    
    def update_trailing_stop(self, volatility: float, multiplier: float = 1.5):
        """
        Update trailing stop based on current price and volatility.
        
        Only moves stop in favorable direction (locks in profit).
        """
        if not self.is_open:
            return
        
        if self.side == SignalDirection.BUY:
            # For long positions, trail stop up
            new_stop = self.current_price - (volatility * multiplier)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                logger.debug(f"Trailing stop updated to {self.stop_loss:.4f}")
        else:
            # For short positions, trail stop down
            new_stop = self.current_price + (volatility * multiplier)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
                logger.debug(f"Trailing stop updated to {self.stop_loss:.4f}")
    
    def close(
        self,
        exit_price: float,
        exit_size: Optional[float] = None,
        reason: str = "manual",
    ):
        """
        Close the position.
        
        Args:
            exit_price: Price at which position was closed
            exit_size: Size closed (defaults to full position)
            reason: Reason for closing
        """
        self.exit_price = exit_price
        self.exit_size = exit_size or self.entry_size
        self.exit_time = datetime.now()
        self.exit_reason = reason
        
        # Calculate realized P&L
        if self.side == SignalDirection.BUY:
            self.realized_pnl = (exit_price - self.entry_price) * self.exit_size
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.exit_size
        
        # Update status
        if reason == "stop_loss":
            self.status = PositionStatus.STOP_LOSS
        elif reason == "take_profit":
            self.status = PositionStatus.TAKE_PROFIT
        else:
            self.status = PositionStatus.CLOSED
        
        logger.info(
            f"Position {self.id} closed: {reason} @ {exit_price:.4f}, "
            f"P&L: ${self.realized_pnl:.2f}"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "market_name": self.market_name,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "entry_size": self.entry_size,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "exit_size": self.exit_size,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            market_id=data["market_id"],
            token_id=data["token_id"],
            market_name=data["market_name"],
            side=SignalDirection(data["side"]),
            entry_price=data["entry_price"],
            entry_size=data["entry_size"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"],
            status=PositionStatus(data["status"]),
            current_price=data["current_price"],
            exit_price=data["exit_price"],
            exit_size=data["exit_size"],
            exit_time=datetime.fromisoformat(data["exit_time"]) if data["exit_time"] else None,
            exit_reason=data["exit_reason"],
            realized_pnl=data["realized_pnl"],
            fees_paid=data["fees_paid"],
        )


@dataclass
class DailyStats:
    """Daily performance statistics."""
    date: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def net_pnl(self) -> float:
        """Net P&L after fees."""
        return self.total_pnl - self.total_fees


class PositionManager:
    """
    Manages all trading positions.
    
    Responsibilities:
    - Track open and closed positions
    - Calculate P&L
    - Check stop-loss and take-profit conditions
    - Persist state to disk for recovery
    - Calculate daily statistics
    
    Usage:
        manager = PositionManager()
        position = manager.open_position(signal, execution)
        manager.update_price(market_id, new_price)
        exits = manager.check_exits()
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_file: str = "data/positions.json",
    ):
        """
        Initialize the position manager.
        
        Args:
            settings: Optional Settings object
            state_file: Path to persist position state
        """
        self.settings = settings or get_settings()
        self.state_file = Path(state_file)
        
        # Position storage
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        
        # Daily stats
        self._daily_stats: Dict[str, DailyStats] = {}
        
        # Callbacks
        self._on_position_close: Optional[Callable[[Position], None]] = None
        
        # Load persisted state
        self._load_state()
    
    def set_close_callback(self, callback: Callable[[Position], None]):
        """Set callback for position close events."""
        self._on_position_close = callback
    
    def open_position(
        self,
        signal: TradingSignal,
        execution: ExecutionResult,
        market_name: str = "",
    ) -> Position:
        """
        Open a new position from an executed signal.
        
        Args:
            signal: The trading signal
            execution: The execution result
            market_name: Human-readable market name
            
        Returns:
            The new Position object
        """
        position_id = f"pos_{datetime.now().timestamp()}"
        
        position = Position(
            id=position_id,
            market_id=signal.market_id,
            token_id=signal.token_id,
            market_name=market_name,
            side=signal.direction,
            entry_price=execution.executed_price,
            entry_size=execution.executed_size,
            entry_time=datetime.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_price=execution.executed_price,
            fees_paid=execution.fees,
        )
        
        self._positions[position_id] = position
        self._save_state()
        
        logger.info(
            f"Opened position {position_id}: {signal.direction.value} "
            f"{execution.executed_size:.2f} @ {execution.executed_price:.4f}"
        )
        
        return position
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[Position]:
        """
        Close a position.
        
        Args:
            position_id: ID of position to close
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            The closed Position or None if not found
        """
        if position_id not in self._positions:
            logger.warning(f"Position {position_id} not found")
            return None
        
        position = self._positions[position_id]
        position.close(exit_price, reason=reason)
        
        # Move to closed positions
        del self._positions[position_id]
        self._closed_positions.append(position)
        
        # Update daily stats
        self._update_daily_stats(position)
        
        # Save state
        self._save_state()
        
        # Notify callback
        if self._on_position_close:
            self._on_position_close(position)
        
        return position
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary mapping token_id to current price
        """
        for position in self._positions.values():
            if position.token_id in prices:
                position.current_price = prices[position.token_id]
    
    def update_price(self, token_id: str, price: float):
        """Update price for a specific token."""
        for position in self._positions.values():
            if position.token_id == token_id:
                position.current_price = price
    
    def check_exits(self) -> List[Position]:
        """
        Check all positions for exit conditions.
        
        Returns:
            List of positions that should be closed
        """
        to_exit = []
        
        for position in self._positions.values():
            if position.should_stop_loss():
                logger.info(f"Stop-loss triggered for {position.id}")
                to_exit.append((position, "stop_loss"))
            elif position.should_take_profit():
                logger.info(f"Take-profit triggered for {position.id}")
                to_exit.append((position, "take_profit"))
        
        # Close positions
        closed = []
        for position, reason in to_exit:
            closed_pos = self.close_position(
                position.id,
                position.current_price,
                reason=reason,
            )
            if closed_pos:
                closed.append(closed_pos)
        
        return closed
    
    def update_trailing_stops(self, volatilities: Dict[str, float]):
        """
        Update trailing stops for all positions.
        
        Args:
            volatilities: Dictionary mapping market_id to current volatility
        """
        for position in self._positions.values():
            if position.market_id in volatilities:
                position.update_trailing_stop(
                    volatilities[position.market_id],
                    multiplier=1.5,
                )
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self._positions.get(position_id)
    
    def get_positions_for_market(self, market_id: str) -> List[Position]:
        """Get all positions for a market."""
        return [p for p in self._positions.values() if p.market_id == market_id]
    
    def get_position_for_token(self, token_id: str) -> Optional[Position]:
        """Get position for a specific token (if any)."""
        for p in self._positions.values():
            if p.token_id == token_id:
                return p
        return None
    
    @property
    def open_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())
    
    @property
    def open_position_count(self) -> int:
        """Number of open positions."""
        return len(self._positions)
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L from closed positions."""
        return sum(p.realized_pnl for p in self._closed_positions)
    
    @property
    def total_exposure(self) -> float:
        """Total value of open positions."""
        return sum(p.current_value for p in self._positions.values())
    
    def get_daily_stats(self, date: Optional[str] = None) -> DailyStats:
        """
        Get statistics for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD) or None for today
            
        Returns:
            DailyStats for the date
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if date not in self._daily_stats:
            self._daily_stats[date] = DailyStats(date=date)
        
        return self._daily_stats[date]
    
    @property
    def today_stats(self) -> DailyStats:
        """Get today's statistics."""
        return self.get_daily_stats()
    
    @property
    def today_pnl(self) -> float:
        """Today's realized P&L."""
        return self.today_stats.total_pnl
    
    @property
    def today_trade_count(self) -> int:
        """Number of trades today."""
        return self.today_stats.total_trades
    
    def has_position_for_market(self, market_id: str) -> bool:
        """Check if we have an open position for a market."""
        return any(p.market_id == market_id for p in self._positions.values())
    
    def can_open_new_position(self) -> bool:
        """Check if we can open a new position (within limits)."""
        max_positions = self.settings.risk.max_open_trades
        return self.open_position_count < max_positions
    
    def _update_daily_stats(self, position: Position):
        """Update daily stats when a position is closed."""
        date = datetime.now().strftime("%Y-%m-%d")
        stats = self.get_daily_stats(date)
        
        stats.total_trades += 1
        stats.total_pnl += position.realized_pnl
        stats.total_fees += position.fees_paid
        
        if position.realized_pnl >= 0:
            stats.winning_trades += 1
        else:
            stats.losing_trades += 1
        
        # Track drawdown
        if stats.total_pnl < stats.max_drawdown:
            stats.max_drawdown = stats.total_pnl
    
    def _save_state(self):
        """Persist position state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "positions": {
                    pid: p.to_dict() for pid, p in self._positions.items()
                },
                "closed_positions": [p.to_dict() for p in self._closed_positions[-100:]],
                "daily_stats": {
                    date: asdict(stats) for date, stats in self._daily_stats.items()
                },
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save position state: {e}")
    
    def _load_state(self):
        """Load position state from disk."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            # Load positions
            for pid, pdata in state.get("positions", {}).items():
                self._positions[pid] = Position.from_dict(pdata)
            
            # Load closed positions
            for pdata in state.get("closed_positions", []):
                self._closed_positions.append(Position.from_dict(pdata))
            
            # Load daily stats
            for date, sdata in state.get("daily_stats", {}).items():
                self._daily_stats[date] = DailyStats(**sdata)
            
            logger.info(f"Loaded {len(self._positions)} open positions from state")
            
        except Exception as e:
            logger.error(f"Failed to load position state: {e}")
    
    def clear_all(self):
        """Clear all positions (for testing)."""
        self._positions.clear()
        self._closed_positions.clear()
        self._daily_stats.clear()
        
        if self.state_file.exists():
            self.state_file.unlink()


if __name__ == "__main__":
    # Test the position manager
    from src.core.detector import TradingSignal, SignalDirection
    from src.core.executor import ExecutionResult, ExecutionStatus
    from src.core.client import Order, OrderStatus, Side
    
    print("Testing Position Manager...")
    print("=" * 60)
    
    manager = PositionManager(state_file="data/test_positions.json")
    manager.clear_all()
    
    # Create a test signal
    signal = TradingSignal(
        market_id="test_market",
        token_id="test_token",
        direction=SignalDirection.BUY,
        confidence=0.8,
        entry_price=0.50,
        stop_loss=0.48,
        take_profit=0.54,
        cusum_value=0.035,
        roc_value=1.5,
        volatility=0.01,
    )
    
    # Create a test execution
    execution = ExecutionResult(
        signal=signal,
        status=ExecutionStatus.SUCCESS,
        order=Order(
            id="test_order",
            market_id="test_market",
            token_id="test_token",
            side=Side.BUY,
            price=0.505,
            size=4.0,
            status=OrderStatus.FILLED,
            filled_size=4.0,
        ),
        executed_price=0.505,
        executed_size=4.0,
        fees=0.01,
    )
    
    # Open position
    position = manager.open_position(signal, execution, "Test Market")
    print(f"\nOpened position: {position.id}")
    print(f"  Entry: {position.entry_price:.4f}")
    print(f"  Size: {position.entry_size:.2f}")
    print(f"  Stop Loss: {position.stop_loss:.4f}")
    print(f"  Take Profit: {position.take_profit:.4f}")
    
    # Update price
    print("\n[Simulating price updates...]")
    
    # Price goes up
    manager.update_price("test_token", 0.52)
    print(f"Price: 0.52 - Unrealized P&L: ${position.unrealized_pnl:.2f}")
    
    # Price hits take profit
    manager.update_price("test_token", 0.55)
    print(f"Price: 0.55 - Unrealized P&L: ${position.unrealized_pnl:.2f}")
    
    # Check exits
    exits = manager.check_exits()
    print(f"\nExits triggered: {len(exits)}")
    
    if exits:
        closed = exits[0]
        print(f"  Closed at: {closed.exit_price:.4f}")
        print(f"  Reason: {closed.exit_reason}")
        print(f"  Realized P&L: ${closed.realized_pnl:.2f}")
    
    # Show stats
    stats = manager.today_stats
    print(f"\nToday's Stats:")
    print(f"  Trades: {stats.total_trades}")
    print(f"  Win Rate: {stats.win_rate:.0f}%")
    print(f"  Total P&L: ${stats.total_pnl:.2f}")
    
    # Cleanup
    manager.clear_all()