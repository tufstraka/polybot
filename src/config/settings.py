"""
Configuration Settings for Polybot
===================================

Loads configuration from YAML file and environment variables.
Uses Pydantic for validation and type safety.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Models (matching config.yaml structure)
# ═══════════════════════════════════════════════════════════════════════════════


class ModeConfig(BaseModel):
    """Trading mode configuration."""
    paper_trading: bool = Field(
        default=True,
        description="If True, run in paper trading mode (no real money)"
    )


class MoneyConfig(BaseModel):
    """Money and position sizing configuration."""
    starting_balance: float = Field(
        default=75.0,
        ge=0,
        description="Total capital in USD"
    )
    bet_size: float = Field(
        default=2.0,
        ge=0.1,
        le=100,
        description="Amount per trade in USD"
    )
    max_daily_loss: float = Field(
        default=2.0,
        ge=0,
        description="Maximum loss allowed per day before stopping"
    )


class DetectionConfig(BaseModel):
    """Spike detection algorithm configuration."""
    # CUSUM parameters
    cusum_threshold: float = Field(
        default=0.03,
        ge=0.001,
        le=0.5,
        description="Cumulative deviation threshold to trigger trade"
    )
    cusum_slack: float = Field(
        default=0.005,
        ge=0,
        le=0.1,
        description="Noise filter - ignore moves smaller than this"
    )
    
    # EWMA parameters
    ewma_span: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of price points for moving average"
    )
    ewma_multiplier: float = Field(
        default=2.5,
        ge=1.0,
        le=5.0,
        description="Standard deviations for volatility bands"
    )
    
    # ROC (Rate of Change) parameters
    roc_threshold: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Minimum price velocity in percent"
    )
    roc_periods: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Lookback periods for momentum calculation"
    )


class RiskConfig(BaseModel):
    """Risk management configuration."""
    take_profit_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=20.0,
        description="Take profit percentage"
    )
    stop_loss_percent: float = Field(
        default=2.0,
        ge=0.5,
        le=20.0,
        description="Stop loss percentage"
    )
    max_open_trades: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum concurrent open positions"
    )
    cooldown_seconds: int = Field(
        default=30,
        ge=0,
        le=600,
        description="Seconds to wait before trading same market again"
    )
    atr_stop_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="ATR multiplier for adaptive stop loss"
    )
    atr_profit_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="ATR multiplier for adaptive take profit"
    )


class FiltersConfig(BaseModel):
    """Market filtering configuration."""
    min_daily_volume: float = Field(
        default=5000.0,
        ge=0,
        description="Minimum daily volume in USD"
    )
    min_liquidity_usd: float = Field(
        default=50.0,
        ge=0,
        description="Minimum orderbook depth in USD"
    )
    max_spread_percent: float = Field(
        default=3.0,
        ge=0,
        le=20.0,
        description="Maximum bid-ask spread percentage"
    )


class PollingConfig(BaseModel):
    """Price polling configuration."""
    interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="How often to check prices"
    )
    market_refresh_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="How often to refresh market list"
    )
    price_history_size: int = Field(
        default=60,
        ge=20,
        le=1000,
        description="Number of price points to keep in memory"
    )


class NotificationsConfig(BaseModel):
    """Notification configuration."""
    telegram_enabled: bool = Field(default=False)
    discord_enabled: bool = Field(default=False)
    notify_on_trade: bool = Field(default=True)
    notify_on_spike: bool = Field(default=True)
    notify_on_daily_summary: bool = Field(default=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables (secrets - loaded from .env)
# ═══════════════════════════════════════════════════════════════════════════════


class EnvSettings(BaseSettings):
    """Environment variables for secrets."""
    
    # Polymarket API credentials
    polymarket_api_key: str = Field(default="")
    polymarket_api_secret: str = Field(default="")
    polymarket_passphrase: str = Field(default="")
    polymarket_private_key: str = Field(default="")
    
    # Telegram notifications
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    
    # Discord notifications
    discord_webhook_url: str = Field(default="")
    
    # Logging
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Settings Class
# ═══════════════════════════════════════════════════════════════════════════════


class Settings(BaseModel):
    """
    Main settings class combining YAML config and environment variables.
    
    Usage:
        settings = get_settings()
        print(settings.money.bet_size)  # $2
        print(settings.env.polymarket_api_key)  # from .env
    """
    
    mode: ModeConfig = Field(default_factory=ModeConfig)
    money: MoneyConfig = Field(default_factory=MoneyConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    env: EnvSettings = Field(default_factory=EnvSettings)
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "Settings":
        """Load settings from YAML file and environment variables."""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            yaml_config = {}
        
        # Load environment variables
        env_settings = EnvSettings()
        
        # Merge YAML config with env settings
        return cls(
            mode=ModeConfig(**yaml_config.get("mode", {})),
            money=MoneyConfig(**yaml_config.get("money", {})),
            detection=DetectionConfig(**yaml_config.get("detection", {})),
            risk=RiskConfig(**yaml_config.get("risk", {})),
            filters=FiltersConfig(**yaml_config.get("filters", {})),
            polling=PollingConfig(**yaml_config.get("polling", {})),
            notifications=NotificationsConfig(**yaml_config.get("notifications", {})),
            env=env_settings,
        )
    
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.mode.paper_trading
    
    def has_valid_credentials(self) -> bool:
        """Check if Polymarket credentials are configured."""
        return bool(
            self.env.polymarket_api_key and
            self.env.polymarket_private_key
        )
    
    def get_telegram_config(self) -> Optional[tuple]:
        """Get Telegram configuration if enabled."""
        if self.notifications.telegram_enabled:
            if self.env.telegram_bot_token and self.env.telegram_chat_id:
                return (self.env.telegram_bot_token, self.env.telegram_chat_id)
        return None
    
    def get_discord_config(self) -> Optional[str]:
        """Get Discord webhook URL if enabled."""
        if self.notifications.discord_enabled:
            if self.env.discord_webhook_url:
                return self.env.discord_webhook_url
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Settings Singleton
# ═══════════════════════════════════════════════════════════════════════════════


@lru_cache()
def get_settings(config_path: str = "config/config.yaml") -> Settings:
    """
    Get settings singleton.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Settings object with all configuration
    """
    return Settings.from_yaml(config_path)


def reload_settings(config_path: str = "config/config.yaml") -> Settings:
    """
    Force reload settings (clears cache).
    
    Useful for testing or when config file changes.
    """
    get_settings.cache_clear()
    return get_settings(config_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════


def is_paper_trading() -> bool:
    """Quick check if running in paper trading mode."""
    return get_settings().is_paper_trading()


def get_bet_size() -> float:
    """Get configured bet size."""
    return get_settings().money.bet_size


def get_max_daily_loss() -> float:
    """Get maximum daily loss limit."""
    return get_settings().money.max_daily_loss


if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    print("=" * 60)
    print("POLYBOT CONFIGURATION")
    print("=" * 60)
    print(f"\n📋 Mode: {'Paper Trading' if settings.is_paper_trading() else 'LIVE TRADING'}")
    print(f"\n💰 Money Settings:")
    print(f"   Starting Balance: ${settings.money.starting_balance}")
    print(f"   Bet Size: ${settings.money.bet_size}")
    print(f"   Max Daily Loss: ${settings.money.max_daily_loss}")
    print(f"\n🎯 Detection Settings:")
    print(f"   CUSUM Threshold: {settings.detection.cusum_threshold}")
    print(f"   EWMA Span: {settings.detection.ewma_span}")
    print(f"   ROC Threshold: {settings.detection.roc_threshold}%")
    print(f"\n⚠️ Risk Settings:")
    print(f"   Take Profit: {settings.risk.take_profit_percent}%")
    print(f"   Stop Loss: {settings.risk.stop_loss_percent}%")
    print(f"   Max Open Trades: {settings.risk.max_open_trades}")
    print(f"\n🔑 Credentials: {'✅ Configured' if settings.has_valid_credentials() else '❌ Not configured'}")
    print("=" * 60)