"""
Trading Bot - Discord Notifier
Sends alerts and notifications to a Discord channel via webhook or bot.
"""

import asyncio
import aiohttp
from loguru import logger
from datetime import datetime

from config import settings


class DiscordNotifier:
    """Sends trading alerts and status updates to Discord."""

    def __init__(self):
        self.token = settings.discord.BOT_TOKEN
        self.channel_id = settings.discord.CHANNEL_ID
        self.enabled = settings.discord.ENABLED
        self.session = None

    async def _get_session(self):
        """Lazy initialization of aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send_message(self, content: str, title: str = None, color: int = 0x3498db):
        """
        Send a message to Discord.
        
        Args:
            content: The message body
            title: Optional embed title
            color: Hex color code for the embed
        """
        if not self.enabled or not self.token or not self.channel_id:
            logger.debug(f"Discord disabled or missing config. Message: {content}")
            return

        url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "embeds": [{
                "description": content,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        
        if title:
            payload["embeds"][0]["title"] = title

        try:
            session = await self._get_session()
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    logger.debug("Discord message sent successfully")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to send Discord message: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")

    async def notify_trade_open(self, trade_data: dict):
        """Send notification for a newly opened trade."""
        side_emoji = "🟢 BUY" if trade_data["side"] == "buy" else "🔴 SELL"
        title = f"🚀 Trade Opened: {side_emoji} {trade_data['pair']}"
        content = (
            f"**Price:** ${trade_data['entry_price']:,.2f}\n"
            f"**Size:** {trade_data['amount']:.6f} BTC\n"
            f"**Stop-Loss:** ${trade_data['stop_loss']:,.2f}\n"
            f"**Take-Profit:** ${trade_data['take_profit']:,.2f}\n"
            f"**Reason:** {trade_data['notes']}"
        )
        color = 0x2ecc71 if trade_data["side"] == "buy" else 0xe74c3c
        await self.send_message(content, title=title, color=color)

    async def notify_trade_closed(self, trade_data: dict):
        """Send notification for a closed trade."""
        pnl = trade_data["pnl"]
        pnl_pct = trade_data["pnl_pct"]
        emoji = "✅" if pnl > 0 else "❌"
        title = f"{emoji} Trade Closed: {trade_data['pair']}"
        
        content = (
            f"**PnL:** ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
            f"**Entry:** ${trade_data['entry_price']:,.2f}\n"
            f"**Exit:** ${trade_data['exit_price']:,.2f}\n"
            f"**Reason:** {trade_data['notes']}"
        )
        color = 0x2ecc71 if pnl > 0 else 0xe74c3c
        await self.send_message(content, title=title, color=color)

    async def notify_error(self, message: str):
        """Send notification for a system error."""
        await self.send_message(f"⚠️ **Error:** {message}", title="System Alert", color=0xFF0000)

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Global notifier instance
notifier = DiscordNotifier()
