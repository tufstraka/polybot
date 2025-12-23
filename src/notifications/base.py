"""
Base notification system and manager.

Plain English Explanation:
==========================

This is the central hub for all notifications. Instead of calling each
notification service separately, you just tell the manager what happened
and it handles sending to all configured services (Telegram, Discord, etc.)

Think of it like a news agency: you tell the agency about an event once,
and they distribute it to all subscribed outlets.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications we can send."""
    SIGNAL = "signal"           # New trading signal detected
    TRADE_OPEN = "trade_open"   # Position opened
    TRADE_CLOSE = "trade_close" # Position closed
    PROFIT = "profit"           # Trade was profitable
    LOSS = "loss"               # Trade lost money
    WARNING = "warning"         # Risk warning (approaching limits)
    ERROR = "error"             # Something went wrong
    STATUS = "status"           # Bot status update
    DAILY_SUMMARY = "summary"   # End of day summary


@dataclass
class Notification:
    """
    A notification to be sent to all channels.
    
    This is the standard format that all notification services understand.
    """
    type: NotificationType
    title: str
    message: str
    timestamp: datetime = None
    
    # Optional fields for specific notification types
    market_name: Optional[str] = None
    market_id: Optional[str] = None
    price: Optional[float] = None
    direction: Optional[str] = None  # "up" or "down"
    pnl: Optional[float] = None
    confidence: Optional[float] = None
    
    # Rich data for detailed notifications
    extra_data: Optional[dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class BaseNotifier(ABC):
    """
    Abstract base class for notification services.
    
    Each notification service (Telegram, Discord, etc.) must implement:
    - send(): Actually send the notification
    - format_message(): Convert Notification to service-specific format
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._send_queue: List[Notification] = []
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """
        Send a notification.
        
        Returns True if sent successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def format_message(self, notification: Notification) -> str:
        """
        Convert notification to service-specific message format.
        
        For example, Telegram might use HTML formatting while
        Discord uses Markdown.
        """
        pass
    
    async def send_if_enabled(self, notification: Notification) -> bool:
        """Send notification only if service is enabled."""
        if not self.enabled:
            logger.debug(f"{self.__class__.__name__} disabled, skipping")
            return True
        return await self.send(notification)


class NotificationManager:
    """
    Central manager for all notification services.
    
    Usage:
        manager = NotificationManager()
        manager.add_notifier(TelegramNotifier(token="...", chat_id="..."))
        manager.add_notifier(DiscordNotifier(webhook_url="..."))
        
        # Send to all services at once
        await manager.notify_signal(market="Trump Win", direction="up", price=0.65)
    """
    
    def __init__(self):
        self._notifiers: List[BaseNotifier] = []
        self._enabled = True
    
    def add_notifier(self, notifier: BaseNotifier):
        """Add a notification service."""
        self._notifiers.append(notifier)
        logger.info(f"Added notifier: {notifier.__class__.__name__}")
    
    def enable(self):
        """Enable all notifications."""
        self._enabled = True
    
    def disable(self):
        """Disable all notifications."""
        self._enabled = False
    
    async def _send_to_all(self, notification: Notification):
        """Send notification to all registered services."""
        if not self._enabled:
            logger.debug("Notifications disabled globally")
            return
        
        if not self._notifiers:
            logger.debug("No notifiers registered")
            return
        
        # Send to all notifiers concurrently
        tasks = [n.send_if_enabled(notification) for n in self._notifiers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any failures
        for notifier, result in zip(self._notifiers, results):
            if isinstance(result, Exception):
                logger.error(f"{notifier.__class__.__name__} failed: {result}")
            elif not result:
                logger.warning(f"{notifier.__class__.__name__} returned False")
    
    # Convenience methods for common notification types
    
    async def notify_signal(
        self,
        market: str,
        direction: str,
        price: float,
        confidence: float,
        market_id: str = None
    ):
        """
        Send notification about a detected trading signal.
        
        Args:
            market: Market name (e.g., "Trump wins election")
            direction: "up" or "down"
            price: Current price
            confidence: Signal confidence (0-1)
        """
        emoji = "🚀" if direction == "up" else "📉"
        
        notification = Notification(
            type=NotificationType.SIGNAL,
            title=f"{emoji} Spike Detected!",
            message=f"Sharp move {direction} in {market}",
            market_name=market,
            market_id=market_id,
            price=price,
            direction=direction,
            confidence=confidence
        )
        
        await self._send_to_all(notification)
    
    async def notify_trade_open(
        self,
        market: str,
        side: str,
        price: float,
        size: float,
        market_id: str = None
    ):
        """Notify when a trade is opened."""
        notification = Notification(
            type=NotificationType.TRADE_OPEN,
            title="📊 Trade Opened",
            message=f"{'Bought' if side == 'BUY' else 'Sold'} ${size:.2f} @ ${price:.3f}",
            market_name=market,
            market_id=market_id,
            price=price,
            extra_data={"side": side, "size": size}
        )
        
        await self._send_to_all(notification)
    
    async def notify_trade_close(
        self,
        market: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
        exit_reason: str = "take_profit"
    ):
        """Notify when a trade is closed."""
        is_profit = pnl > 0
        emoji = "💰" if is_profit else "😞"
        
        notification = Notification(
            type=NotificationType.PROFIT if is_profit else NotificationType.LOSS,
            title=f"{emoji} Trade Closed",
            message=f"{'Profit' if is_profit else 'Loss'}: ${abs(pnl):.2f}",
            market_name=market,
            pnl=pnl,
            extra_data={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason
            }
        )
        
        await self._send_to_all(notification)
    
    async def notify_warning(self, title: str, message: str):
        """Send a warning notification."""
        notification = Notification(
            type=NotificationType.WARNING,
            title=f"⚠️ {title}",
            message=message
        )
        
        await self._send_to_all(notification)
    
    async def notify_error(self, error: str, details: str = None):
        """Send an error notification."""
        notification = Notification(
            type=NotificationType.ERROR,
            title="🚨 Error",
            message=error,
            extra_data={"details": details} if details else None
        )
        
        await self._send_to_all(notification)
    
    async def notify_status(self, status: str, details: dict = None):
        """Send a status update notification."""
        notification = Notification(
            type=NotificationType.STATUS,
            title="ℹ️ Status Update",
            message=status,
            extra_data=details
        )
        
        await self._send_to_all(notification)
    
    async def notify_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        pnl: float,
        capital: float
    ):
        """Send end-of-day summary notification."""
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
        
        notification = Notification(
            type=NotificationType.DAILY_SUMMARY,
            title=f"{emoji} Daily Summary",
            message=f"P&L: ${pnl:+.2f} | Win Rate: {win_rate:.0f}%",
            pnl=pnl,
            extra_data={
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "capital": capital
            }
        )
        
        await self._send_to_all(notification)