"""
Telegram Notification Service.

Plain English Explanation:
==========================

This sends notifications to your Telegram chat. You need:
1. A Telegram Bot Token (from @BotFather)
2. Your Chat ID (from @userinfobot)

Setup Steps:
1. Message @BotFather on Telegram
2. Send /newbot and follow instructions
3. Copy the bot token (looks like: 123456:ABC-DEF1234...)
4. Message @userinfobot to get your chat ID
5. Add both to your .env file

Features:
- Formatted messages with emojis
- Clickable market links
- Detailed trade information
- Rate limiting to avoid spam
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import aiohttp

from src.notifications.base import BaseNotifier, Notification, NotificationType

logger = logging.getLogger(__name__)


class TelegramNotifier(BaseNotifier):
    """
    Sends notifications via Telegram Bot API.
    
    Usage:
        notifier = TelegramNotifier(
            bot_token="123456:ABC-DEF...",
            chat_id="987654321"
        )
        
        await notifier.send(notification)
    """
    
    # Telegram API base URL
    API_BASE = "https://api.telegram.org/bot{token}/{method}"
    
    # Rate limiting: min seconds between messages
    MIN_INTERVAL = 1.0
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        parse_mode: str = "HTML",
        disable_preview: bool = True
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Your bot token from @BotFather
            chat_id: Chat ID to send messages to
            enabled: Whether to actually send messages
            parse_mode: Message format ("HTML" or "Markdown")
            disable_preview: Disable link previews
        """
        super().__init__(enabled)
        
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.disable_preview = disable_preview
        
        self._last_send_time: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(self, method: str) -> str:
        """Build Telegram API URL."""
        return self.API_BASE.format(token=self.bot_token, method=method)
    
    async def _rate_limit(self):
        """Ensure we don't send messages too quickly."""
        if self._last_send_time:
            elapsed = (datetime.utcnow() - self._last_send_time).total_seconds()
            if elapsed < self.MIN_INTERVAL:
                await asyncio.sleep(self.MIN_INTERVAL - elapsed)
        self._last_send_time = datetime.utcnow()
    
    def format_message(self, notification: Notification) -> str:
        """
        Format notification as HTML message for Telegram.
        
        Uses HTML tags like <b>bold</b>, <i>italic</i>, <code>mono</code>
        """
        lines = []
        
        # Title with bold
        lines.append(f"<b>{notification.title}</b>")
        lines.append("")
        
        # Main message
        lines.append(notification.message)
        
        # Type-specific formatting
        if notification.type == NotificationType.SIGNAL:
            lines.append("")
            lines.append(f"📌 <b>Market:</b> {notification.market_name}")
            lines.append(f"💲 <b>Price:</b> ${notification.price:.3f}")
            lines.append(f"📊 <b>Direction:</b> {notification.direction.upper()}")
            if notification.confidence:
                lines.append(f"🎯 <b>Confidence:</b> {notification.confidence:.1%}")
        
        elif notification.type in (NotificationType.PROFIT, NotificationType.LOSS):
            lines.append("")
            lines.append(f"📌 <b>Market:</b> {notification.market_name}")
            lines.append(f"💵 <b>P&L:</b> ${notification.pnl:+.2f}")
            
            if notification.extra_data:
                entry = notification.extra_data.get("entry_price", 0)
                exit_price = notification.extra_data.get("exit_price", 0)
                reason = notification.extra_data.get("exit_reason", "unknown")
                
                lines.append(f"📈 <b>Entry:</b> ${entry:.3f}")
                lines.append(f"📉 <b>Exit:</b> ${exit_price:.3f}")
                lines.append(f"🏷️ <b>Reason:</b> {reason.replace('_', ' ').title()}")
        
        elif notification.type == NotificationType.TRADE_OPEN:
            lines.append("")
            lines.append(f"📌 <b>Market:</b> {notification.market_name}")
            lines.append(f"💲 <b>Price:</b> ${notification.price:.3f}")
            
            if notification.extra_data:
                side = notification.extra_data.get("side", "BUY")
                size = notification.extra_data.get("size", 0)
                lines.append(f"📊 <b>Side:</b> {side}")
                lines.append(f"💵 <b>Size:</b> ${size:.2f}")
        
        elif notification.type == NotificationType.DAILY_SUMMARY:
            if notification.extra_data:
                data = notification.extra_data
                lines.append("")
                lines.append(f"📊 <b>Trades:</b> {data.get('total_trades', 0)}")
                lines.append(f"✅ <b>Wins:</b> {data.get('wins', 0)}")
                lines.append(f"❌ <b>Losses:</b> {data.get('losses', 0)}")
                lines.append(f"🎯 <b>Win Rate:</b> {data.get('win_rate', 0):.1f}%")
                lines.append(f"💰 <b>Capital:</b> ${data.get('capital', 0):.2f}")
        
        elif notification.type == NotificationType.WARNING:
            lines.append("")
            lines.append("⚠️ <i>Action may be required</i>")
        
        elif notification.type == NotificationType.ERROR:
            if notification.extra_data and notification.extra_data.get("details"):
                lines.append("")
                lines.append(f"<code>{notification.extra_data['details']}</code>")
        
        # Timestamp
        lines.append("")
        lines.append(f"<i>{notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")
        
        return "\n".join(lines)
    
    async def send(self, notification: Notification) -> bool:
        """
        Send notification to Telegram.
        
        Returns True if successful, False otherwise.
        """
        try:
            await self._rate_limit()
            
            message = self.format_message(notification)
            
            session = await self._get_session()
            url = self._build_url("sendMessage")
            
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": self.parse_mode,
                "disable_web_page_preview": self.disable_preview
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        logger.debug(f"Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"Telegram API error: {result.get('description')}")
                        return False
                else:
                    text = await response.text()
                    logger.error(f"Telegram HTTP error {response.status}: {text}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"Telegram connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def send_raw(self, text: str) -> bool:
        """
        Send a raw text message without formatting.
        
        Useful for quick debug messages.
        """
        notification = Notification(
            type=NotificationType.STATUS,
            title="",
            message=text
        )
        return await self.send(notification)
    
    async def test_connection(self) -> bool:
        """
        Test if the bot token and chat ID are valid.
        
        Returns True if we can send messages.
        """
        try:
            session = await self._get_session()
            
            # First, check if bot token is valid
            url = self._build_url("getMe")
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error("Invalid bot token")
                    return False
                
                result = await response.json()
                if not result.get("ok"):
                    logger.error("Bot token validation failed")
                    return False
                
                bot_name = result.get("result", {}).get("username", "unknown")
                logger.info(f"Connected to Telegram bot: @{bot_name}")
            
            # Send a test message
            return await self.send_raw("🤖 Polybot connected and ready!")
            
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False


# Quick test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        logging.basicConfig(level=logging.DEBUG)
        
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            return
        
        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        
        # Test connection
        print("Testing connection...")
        if await notifier.test_connection():
            print("✅ Connection successful!")
            
            # Send test notifications
            notification = Notification(
                type=NotificationType.SIGNAL,
                title="🚀 Test Signal",
                message="Testing spike detection",
                market_name="Trump wins election",
                price=0.652,
                direction="up",
                confidence=0.85
            )
            
            await notifier.send(notification)
            print("✅ Test notification sent!")
        else:
            print("❌ Connection failed")
        
        await notifier.close()
    
    asyncio.run(main())