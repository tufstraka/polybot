"""
Discord Notification Service.

Plain English Explanation:
==========================

This sends notifications to your Discord server using webhooks.
Webhooks are special URLs that let you post messages to a channel.

Setup Steps:
1. Open Discord and go to your server
2. Click on the channel settings (gear icon)
3. Go to "Integrations" → "Webhooks"
4. Click "New Webhook"
5. Copy the Webhook URL
6. Add it to your .env file as DISCORD_WEBHOOK_URL

Features:
- Rich embed messages with colors
- Market information
- Trade alerts
- Daily summaries
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiohttp

from src.notifications.base import BaseNotifier, Notification, NotificationType

logger = logging.getLogger(__name__)


class DiscordNotifier(BaseNotifier):
    """
    Sends notifications via Discord Webhooks.
    
    Usage:
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/..."
        )
        
        await notifier.send(notification)
    """
    
    # Rate limiting: min seconds between messages
    MIN_INTERVAL = 1.0
    
    # Discord embed color codes
    COLORS = {
        NotificationType.SIGNAL: 0x3498DB,      # Blue
        NotificationType.TRADE_OPEN: 0x9B59B6,  # Purple
        NotificationType.PROFIT: 0x2ECC71,      # Green
        NotificationType.LOSS: 0xE74C3C,        # Red
        NotificationType.TRADE_CLOSE: 0x95A5A6, # Gray
        NotificationType.WARNING: 0xF39C12,     # Orange
        NotificationType.ERROR: 0xE74C3C,       # Red
        NotificationType.STATUS: 0x3498DB,      # Blue
        NotificationType.DAILY_SUMMARY: 0x1ABC9C,  # Teal
    }
    
    def __init__(
        self,
        webhook_url: str,
        username: str = "Polybot",
        avatar_url: str = None,
        enabled: bool = True
    ):
        """
        Initialize Discord notifier.
        
        Args:
            webhook_url: Your Discord webhook URL
            username: Bot name to display
            avatar_url: Optional avatar image URL
            enabled: Whether to actually send messages
        """
        super().__init__(enabled)
        
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
        
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
    
    async def _rate_limit(self):
        """Ensure we don't send messages too quickly."""
        if self._last_send_time:
            elapsed = (datetime.utcnow() - self._last_send_time).total_seconds()
            if elapsed < self.MIN_INTERVAL:
                await asyncio.sleep(self.MIN_INTERVAL - elapsed)
        self._last_send_time = datetime.utcnow()
    
    def format_message(self, notification: Notification) -> str:
        """
        Format notification as plain text.
        
        Discord webhooks support both plain text and embeds.
        This returns plain text for simple messages.
        """
        return f"**{notification.title}**\n{notification.message}"
    
    def _build_embed(self, notification: Notification) -> Dict[str, Any]:
        """
        Build a Discord embed object.
        
        Embeds are rich message cards with colors, fields, etc.
        """
        embed = {
            "title": notification.title,
            "description": notification.message,
            "color": self.COLORS.get(notification.type, 0x95A5A6),
            "timestamp": notification.timestamp.isoformat(),
            "footer": {
                "text": "Polybot Spike Hunter"
            }
        }
        
        # Add fields based on notification type
        fields: List[Dict[str, Any]] = []
        
        if notification.type == NotificationType.SIGNAL:
            if notification.market_name:
                fields.append({
                    "name": "📌 Market",
                    "value": notification.market_name,
                    "inline": False
                })
            if notification.price:
                fields.append({
                    "name": "💲 Price",
                    "value": f"${notification.price:.3f}",
                    "inline": True
                })
            if notification.direction:
                fields.append({
                    "name": "📊 Direction",
                    "value": notification.direction.upper(),
                    "inline": True
                })
            if notification.confidence:
                fields.append({
                    "name": "🎯 Confidence",
                    "value": f"{notification.confidence:.1%}",
                    "inline": True
                })
        
        elif notification.type in (NotificationType.PROFIT, NotificationType.LOSS):
            if notification.market_name:
                fields.append({
                    "name": "📌 Market",
                    "value": notification.market_name,
                    "inline": False
                })
            if notification.pnl is not None:
                pnl_emoji = "💰" if notification.pnl > 0 else "😞"
                fields.append({
                    "name": f"{pnl_emoji} P&L",
                    "value": f"${notification.pnl:+.2f}",
                    "inline": True
                })
            
            if notification.extra_data:
                entry = notification.extra_data.get("entry_price")
                exit_price = notification.extra_data.get("exit_price")
                reason = notification.extra_data.get("exit_reason")
                
                if entry:
                    fields.append({
                        "name": "📈 Entry",
                        "value": f"${entry:.3f}",
                        "inline": True
                    })
                if exit_price:
                    fields.append({
                        "name": "📉 Exit",
                        "value": f"${exit_price:.3f}",
                        "inline": True
                    })
                if reason:
                    fields.append({
                        "name": "🏷️ Reason",
                        "value": reason.replace("_", " ").title(),
                        "inline": True
                    })
        
        elif notification.type == NotificationType.TRADE_OPEN:
            if notification.market_name:
                fields.append({
                    "name": "📌 Market",
                    "value": notification.market_name,
                    "inline": False
                })
            if notification.price:
                fields.append({
                    "name": "💲 Price",
                    "value": f"${notification.price:.3f}",
                    "inline": True
                })
            
            if notification.extra_data:
                side = notification.extra_data.get("side")
                size = notification.extra_data.get("size")
                
                if side:
                    fields.append({
                        "name": "📊 Side",
                        "value": side,
                        "inline": True
                    })
                if size:
                    fields.append({
                        "name": "💵 Size",
                        "value": f"${size:.2f}",
                        "inline": True
                    })
        
        elif notification.type == NotificationType.DAILY_SUMMARY:
            if notification.extra_data:
                data = notification.extra_data
                
                fields.append({
                    "name": "📊 Total Trades",
                    "value": str(data.get("total_trades", 0)),
                    "inline": True
                })
                fields.append({
                    "name": "✅ Wins",
                    "value": str(data.get("wins", 0)),
                    "inline": True
                })
                fields.append({
                    "name": "❌ Losses",
                    "value": str(data.get("losses", 0)),
                    "inline": True
                })
                fields.append({
                    "name": "🎯 Win Rate",
                    "value": f"{data.get('win_rate', 0):.1f}%",
                    "inline": True
                })
                fields.append({
                    "name": "💰 Capital",
                    "value": f"${data.get('capital', 0):.2f}",
                    "inline": True
                })
                fields.append({
                    "name": "📈 P&L",
                    "value": f"${notification.pnl:+.2f}" if notification.pnl else "$0.00",
                    "inline": True
                })
        
        elif notification.type == NotificationType.ERROR:
            if notification.extra_data and notification.extra_data.get("details"):
                fields.append({
                    "name": "🔍 Details",
                    "value": f"```{notification.extra_data['details']}```",
                    "inline": False
                })
        
        if fields:
            embed["fields"] = fields
        
        return embed
    
    def _build_payload(self, notification: Notification) -> Dict[str, Any]:
        """Build the complete webhook payload."""
        payload = {
            "username": self.username,
            "embeds": [self._build_embed(notification)]
        }
        
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url
        
        return payload
    
    async def send(self, notification: Notification) -> bool:
        """
        Send notification to Discord.
        
        Returns True if successful, False otherwise.
        """
        try:
            await self._rate_limit()
            
            payload = self._build_payload(notification)
            
            session = await self._get_session()
            
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status in (200, 204):
                    logger.debug("Discord message sent successfully")
                    return True
                elif response.status == 429:
                    # Rate limited - wait and retry
                    retry_after = response.headers.get("Retry-After", 5)
                    logger.warning(f"Discord rate limited, waiting {retry_after}s")
                    await asyncio.sleep(float(retry_after))
                    return await self.send(notification)  # Retry
                else:
                    text = await response.text()
                    logger.error(f"Discord HTTP error {response.status}: {text}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"Discord connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False
    
    async def send_raw(self, text: str) -> bool:
        """
        Send a raw text message without embeds.
        
        Useful for quick debug messages.
        """
        try:
            session = await self._get_session()
            
            payload = {
                "username": self.username,
                "content": text
            }
            
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            async with session.post(self.webhook_url, json=payload) as response:
                return response.status in (200, 204)
                
        except Exception as e:
            logger.error(f"Discord raw send error: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """
        Test if the webhook URL is valid.
        
        Returns True if we can send messages.
        """
        try:
            return await self.send_raw("🤖 Polybot connected and ready!")
        except Exception as e:
            logger.error(f"Discord connection test failed: {e}")
            return False


# Quick test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        logging.basicConfig(level=logging.DEBUG)
        
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        
        if not webhook_url:
            print("Set DISCORD_WEBHOOK_URL in .env")
            return
        
        notifier = DiscordNotifier(webhook_url=webhook_url)
        
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
            
            # Test daily summary
            await asyncio.sleep(1)
            summary = Notification(
                type=NotificationType.DAILY_SUMMARY,
                title="📈 Daily Summary",
                message="Trading day complete",
                pnl=1.25,
                extra_data={
                    "total_trades": 5,
                    "wins": 3,
                    "losses": 2,
                    "win_rate": 60.0,
                    "capital": 76.25
                }
            )
            await notifier.send(summary)
            print("✅ Summary notification sent!")
        else:
            print("❌ Connection failed")
        
        await notifier.close()
    
    asyncio.run(main())