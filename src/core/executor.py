"""
Order Executor Module
=====================

Handles order execution with support for both paper trading (dry-run)
and live trading modes.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import logging

from src.config.settings import get_settings, Settings
from src.core.client import PolymarketClient, Side, Order, OrderStatus
from src.core.detector import TradingSignal, SignalDirection


logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of trade execution."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    signal: TradingSignal
    status: ExecutionStatus
    order: Optional[Order] = None
    
    # Execution details
    executed_price: float = 0.0
    executed_size: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0  # Difference from expected price
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str = ""
    is_paper_trade: bool = True
    
    @property
    def total_cost(self) -> float:
        """Total cost including fees."""
        return (self.executed_price * self.executed_size) + self.fees
    
    @property
    def slippage_percent(self) -> float:
        """Slippage as percentage."""
        if self.signal.entry_price == 0:
            return 0.0
        return ((self.executed_price - self.signal.entry_price) / 
                self.signal.entry_price) * 100


class OrderExecutor:
    """
    Executes trading signals as orders.
    
    Supports two modes:
    1. Paper Trading (dry-run) - Simulates orders without real execution
    2. Live Trading - Places real orders on Polymarket
    
    Usage:
        executor = OrderExecutor(client)
        result = await executor.execute(signal)
    """
    
    def __init__(
        self,
        client: PolymarketClient,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the executor.
        
        Args:
            client: Polymarket API client
            settings: Optional settings object
        """
        self.client = client
        self.settings = settings or get_settings()
        
        # Execution history
        self._executions: List[ExecutionResult] = []
        
        # Callbacks for notifications
        self._on_execution: Optional[Callable[[ExecutionResult], None]] = None
    
    def set_execution_callback(self, callback: Callable[[ExecutionResult], None]):
        """Set callback for execution events."""
        self._on_execution = callback
    
    async def execute(self, signal: TradingSignal) -> ExecutionResult:
        """
        Execute a trading signal.
        
        Args:
            signal: The trading signal to execute
            
        Returns:
            ExecutionResult with execution details
        """
        is_paper = self.settings.is_paper_trading()
        
        if is_paper:
            result = await self._execute_paper(signal)
        else:
            result = await self._execute_live(signal)
        
        # Record execution
        self._executions.append(result)
        
        # Notify callback
        if self._on_execution:
            self._on_execution(result)
        
        return result
    
    async def _execute_paper(self, signal: TradingSignal) -> ExecutionResult:
        """
        Execute signal in paper trading mode.
        
        Simulates execution at the signal's entry price with
        small random slippage.
        """
        import random
        
        # Simulate small slippage (0-0.5%)
        slippage = random.uniform(0, 0.005) * signal.entry_price
        if signal.direction == SignalDirection.BUY:
            executed_price = signal.entry_price + slippage  # Pay slightly more
        else:
            executed_price = signal.entry_price - slippage  # Receive slightly less
        
        # Calculate size based on bet_size setting
        bet_size = self.settings.money.bet_size
        executed_size = bet_size / executed_price if executed_price > 0 else 0
        
        # Simulate small fees (0.1%)
        fees = bet_size * 0.001
        
        logger.info(
            f"[PAPER] Executed {signal.direction.value} "
            f"{executed_size:.2f} shares @ {executed_price:.4f} "
            f"(slippage: {slippage:.4f})"
        )
        
        # Create simulated order
        order = Order(
            id=f"paper_{datetime.now().timestamp()}",
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=Side.BUY if signal.direction == SignalDirection.BUY else Side.SELL,
            price=executed_price,
            size=executed_size,
            status=OrderStatus.FILLED,
            filled_size=executed_size,
            filled_at=datetime.now(),
        )
        
        return ExecutionResult(
            signal=signal,
            status=ExecutionStatus.SUCCESS,
            order=order,
            executed_price=executed_price,
            executed_size=executed_size,
            fees=fees,
            slippage=slippage,
            is_paper_trade=True,
        )
    
    async def _execute_live(self, signal: TradingSignal) -> ExecutionResult:
        """
        Execute signal in live trading mode.
        
        Places a real order on Polymarket.
        """
        side = Side.BUY if signal.direction == SignalDirection.BUY else Side.SELL
        bet_size = self.settings.money.bet_size
        
        try:
            # Use market order for faster execution
            order = await self.client.place_market_order(
                token_id=signal.token_id,
                side=side,
                amount=bet_size,
            )
            
            if order and order.is_filled:
                slippage = order.price - signal.entry_price
                
                logger.info(
                    f"[LIVE] Executed {signal.direction.value} "
                    f"{order.filled_size:.2f} shares @ {order.price:.4f}"
                )
                
                return ExecutionResult(
                    signal=signal,
                    status=ExecutionStatus.SUCCESS,
                    order=order,
                    executed_price=order.price,
                    executed_size=order.filled_size,
                    fees=0.0,  # Polymarket doesn't have taker fees currently
                    slippage=slippage,
                    is_paper_trade=False,
                )
            else:
                return ExecutionResult(
                    signal=signal,
                    status=ExecutionStatus.FAILED,
                    error_message="Order not filled",
                    is_paper_trade=False,
                )
                
        except Exception as e:
            logger.error(f"Live execution failed: {e}")
            return ExecutionResult(
                signal=signal,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                is_paper_trade=False,
            )
    
    async def close_position(
        self,
        token_id: str,
        size: float,
        current_price: float,
        reason: str = "manual",
    ) -> ExecutionResult:
        """
        Close an existing position.
        
        Args:
            token_id: Token to close
            size: Size to close
            current_price: Current market price
            reason: Reason for closing (stop_loss, take_profit, manual)
            
        Returns:
            ExecutionResult
        """
        is_paper = self.settings.is_paper_trading()
        
        # Create a synthetic signal for closing
        close_signal = TradingSignal(
            market_id="",
            token_id=token_id,
            direction=SignalDirection.SELL,  # Closing = selling
            confidence=1.0,
            entry_price=current_price,
            stop_loss=0,
            take_profit=0,
            cusum_value=0,
            roc_value=0,
            volatility=0,
        )
        
        if is_paper:
            # Paper close - instant at current price
            logger.info(f"[PAPER] Closing position: {size:.2f} shares @ {current_price:.4f} ({reason})")
            
            order = Order(
                id=f"paper_close_{datetime.now().timestamp()}",
                market_id="",
                token_id=token_id,
                side=Side.SELL,
                price=current_price,
                size=size,
                status=OrderStatus.FILLED,
                filled_size=size,
                filled_at=datetime.now(),
            )
            
            return ExecutionResult(
                signal=close_signal,
                status=ExecutionStatus.SUCCESS,
                order=order,
                executed_price=current_price,
                executed_size=size,
                is_paper_trade=True,
            )
        else:
            # Live close
            try:
                order = await self.client.place_market_order(
                    token_id=token_id,
                    side=Side.SELL,
                    amount=size * current_price,  # Convert to dollar amount
                )
                
                if order:
                    logger.info(f"[LIVE] Closed position: {order.filled_size:.2f} @ {order.price:.4f}")
                    return ExecutionResult(
                        signal=close_signal,
                        status=ExecutionStatus.SUCCESS,
                        order=order,
                        executed_price=order.price,
                        executed_size=order.filled_size,
                        is_paper_trade=False,
                    )
                    
            except Exception as e:
                logger.error(f"Failed to close position: {e}")
                
            return ExecutionResult(
                signal=close_signal,
                status=ExecutionStatus.FAILED,
                error_message="Failed to close position",
                is_paper_trade=False,
            )
    
    @property
    def execution_count(self) -> int:
        """Number of executions performed."""
        return len(self._executions)
    
    @property
    def successful_executions(self) -> int:
        """Number of successful executions."""
        return sum(1 for e in self._executions if e.status == ExecutionStatus.SUCCESS)
    
    def get_executions(self, limit: int = 100) -> List[ExecutionResult]:
        """Get recent executions."""
        return self._executions[-limit:]
    
    def clear_history(self):
        """Clear execution history."""
        self._executions.clear()