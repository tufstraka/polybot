"""
Price Tracker Module
====================

Maintains real-time price history for each market and calculates
technical indicators like EWMA (Exponentially Weighted Moving Average).
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from src.config.settings import get_settings, Settings
from src.core.scanner import MarketSnapshot


logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price observation."""
    timestamp: datetime
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: float = 0.0


@dataclass
class MarketIndicators:
    """
    Technical indicators for a market.
    
    These are calculated from price history and used by the detector.
    """
    # Current values
    current_price: float = 0.0
    ewma_mean: float = 0.0
    ewma_std: float = 0.0
    ewma_upper_band: float = 0.0
    ewma_lower_band: float = 0.0
    
    # CUSUM values
    cusum_positive: float = 0.0  # Cumulative sum for upward deviations
    cusum_negative: float = 0.0  # Cumulative sum for downward deviations
    
    # Rate of Change
    roc: float = 0.0  # Rate of change (momentum)
    roc_short: float = 0.0  # Very short-term ROC
    
    # Volatility
    volatility: float = 0.0  # Current volatility (std dev)
    atr: float = 0.0  # Average True Range (for prediction markets, similar to volatility)
    
    # Derived signals
    is_above_upper_band: bool = False
    is_below_lower_band: bool = False
    
    @property
    def band_position(self) -> float:
        """
        Position relative to bands (-1 to +1).
        
        -1 = at or below lower band
        0 = at mean
        +1 = at or above upper band
        """
        if self.ewma_std == 0:
            return 0.0
        
        z_score = (self.current_price - self.ewma_mean) / self.ewma_std
        return max(-1, min(1, z_score / 2.5))  # Normalize to -1 to +1
    
    @property
    def momentum_direction(self) -> str:
        """Direction of momentum: 'UP', 'DOWN', or 'NEUTRAL'."""
        if self.roc > 0.5:
            return "UP"
        elif self.roc < -0.5:
            return "DOWN"
        return "NEUTRAL"


@dataclass  
class MarketHistory:
    """
    Price history and indicators for a single market.
    
    Uses a deque for efficient rolling window storage.
    """
    market_id: str
    token_id: str
    max_size: int = 60
    
    # Price history (deque for efficient append/pop)
    prices: deque = field(default_factory=lambda: deque(maxlen=60))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    
    # Current indicators
    indicators: MarketIndicators = field(default_factory=MarketIndicators)
    
    # CUSUM state (persists across updates)
    _cusum_target: Optional[float] = None
    
    def add_price(self, price: float, timestamp: Optional[datetime] = None):
        """
        Add a new price observation.
        
        Args:
            price: The new price
            timestamp: Optional timestamp (defaults to now)
        """
        self.prices.append(price)
        self.timestamps.append(timestamp or datetime.now())
    
    @property
    def price_array(self) -> np.ndarray:
        """Get prices as numpy array."""
        return np.array(self.prices)
    
    @property
    def price_series(self) -> pd.Series:
        """Get prices as pandas Series with datetime index."""
        return pd.Series(list(self.prices), index=list(self.timestamps))
    
    @property
    def has_enough_data(self) -> bool:
        """Check if we have enough data for calculations."""
        return len(self.prices) >= 10
    
    @property
    def current_price(self) -> float:
        """Get the most recent price."""
        return self.prices[-1] if self.prices else 0.0
    
    @property
    def previous_price(self) -> float:
        """Get the second most recent price."""
        return self.prices[-2] if len(self.prices) >= 2 else self.current_price
    
    @property
    def price_change(self) -> float:
        """Get price change since last observation."""
        return self.current_price - self.previous_price
    
    @property
    def price_change_percent(self) -> float:
        """Get price change as percentage."""
        if self.previous_price == 0:
            return 0.0
        return (self.price_change / self.previous_price) * 100


class PriceTracker:
    """
    Tracks prices and calculates indicators for multiple markets.
    
    This is the core component that processes market snapshots and
    calculates the technical indicators used by the spike detector.
    
    Usage:
        tracker = PriceTracker()
        tracker.update(snapshot)
        indicators = tracker.get_indicators(market_id)
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the tracker.
        
        Args:
            settings: Optional Settings object
        """
        self.settings = settings or get_settings()
        
        # Market histories keyed by market_id
        self._histories: Dict[str, MarketHistory] = {}
        
        # Configuration from settings
        self._ewma_span = self.settings.detection.ewma_span
        self._ewma_multiplier = self.settings.detection.ewma_multiplier
        self._cusum_threshold = self.settings.detection.cusum_threshold
        self._cusum_slack = self.settings.detection.cusum_slack
        self._roc_periods = self.settings.detection.roc_periods
        self._history_size = self.settings.polling.price_history_size
    
    def update(self, snapshot: MarketSnapshot) -> MarketIndicators:
        """
        Update price history and recalculate indicators for a market.
        
        Args:
            snapshot: Current market snapshot with price data
            
        Returns:
            Updated MarketIndicators
        """
        market_id = snapshot.market.id
        
        # Get or create history
        if market_id not in self._histories:
            self._histories[market_id] = MarketHistory(
                market_id=market_id,
                token_id=snapshot.market.yes_token_id or "",
                max_size=self._history_size,
            )
            self._histories[market_id].prices = deque(maxlen=self._history_size)
            self._histories[market_id].timestamps = deque(maxlen=self._history_size)
        
        history = self._histories[market_id]
        
        # Add new price
        history.add_price(snapshot.price, snapshot.timestamp)
        
        # Calculate indicators if we have enough data
        if history.has_enough_data:
            indicators = self._calculate_indicators(history)
            history.indicators = indicators
        else:
            # Set basic indicators
            history.indicators = MarketIndicators(
                current_price=snapshot.price,
            )
        
        return history.indicators
    
    def update_batch(self, snapshots: List[MarketSnapshot]) -> Dict[str, MarketIndicators]:
        """
        Update multiple markets at once.
        
        Args:
            snapshots: List of market snapshots
            
        Returns:
            Dictionary mapping market_id to indicators
        """
        results = {}
        for snapshot in snapshots:
            indicators = self.update(snapshot)
            results[snapshot.market.id] = indicators
        return results
    
    def _calculate_indicators(self, history: MarketHistory) -> MarketIndicators:
        """
        Calculate all technical indicators for a market.
        
        This is where the magic happens - we calculate:
        - EWMA mean and standard deviation
        - Bollinger-style bands
        - CUSUM for regime change detection
        - Rate of change (momentum)
        - Volatility measures
        """
        prices = history.price_series
        current_price = history.current_price
        
        # ═══════════════════════════════════════════════════════════════════════
        # EWMA Calculations
        # ═══════════════════════════════════════════════════════════════════════
        
        # Calculate EWMA mean
        ewma_mean = prices.ewm(span=self._ewma_span, adjust=False).mean().iloc[-1]
        
        # Calculate EWMA standard deviation
        ewma_std = prices.ewm(span=self._ewma_span, adjust=False).std().iloc[-1]
        
        # Ensure minimum std to avoid division by zero
        ewma_std = max(ewma_std, 0.001)
        
        # Calculate bands
        ewma_upper = ewma_mean + (self._ewma_multiplier * ewma_std)
        ewma_lower = ewma_mean - (self._ewma_multiplier * ewma_std)
        
        # ═══════════════════════════════════════════════════════════════════════
        # CUSUM Calculation
        # ═══════════════════════════════════════════════════════════════════════
        
        # Initialize target if not set
        if history._cusum_target is None:
            history._cusum_target = ewma_mean
        
        # Update CUSUM accumulators
        # CUSUM accumulates deviations from target, filtered by slack
        cusum_pos = 0.0
        cusum_neg = 0.0
        
        for price in prices:
            deviation = price - history._cusum_target
            cusum_pos = max(0, cusum_pos + deviation - self._cusum_slack)
            cusum_neg = max(0, cusum_neg - deviation - self._cusum_slack)
        
        # Update target to current mean (slowly adapts)
        history._cusum_target = ewma_mean
        
        # ═══════════════════════════════════════════════════════════════════════
        # Rate of Change (Momentum)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Calculate ROC over configured periods
        if len(prices) > self._roc_periods:
            old_price = prices.iloc[-self._roc_periods - 1]
            if old_price != 0:
                roc = ((current_price - old_price) / old_price) * 100
            else:
                roc = 0.0
        else:
            roc = 0.0
        
        # Short-term ROC (1 period)
        roc_short = ((current_price - history.previous_price) / history.previous_price * 100
                     if history.previous_price != 0 else 0.0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Volatility / ATR
        # ═══════════════════════════════════════════════════════════════════════
        
        # For prediction markets, we use standard deviation as ATR proxy
        # since there's no high/low/close like traditional markets
        volatility = ewma_std
        
        # ATR-like measure using recent price range
        if len(prices) >= 5:
            recent_prices = list(prices)[-5:]
            atr = max(recent_prices) - min(recent_prices)
        else:
            atr = volatility
        
        # ═══════════════════════════════════════════════════════════════════════
        # Assemble Indicators
        # ═══════════════════════════════════════════════════════════════════════
        
        return MarketIndicators(
            current_price=current_price,
            ewma_mean=ewma_mean,
            ewma_std=ewma_std,
            ewma_upper_band=ewma_upper,
            ewma_lower_band=ewma_lower,
            cusum_positive=cusum_pos,
            cusum_negative=cusum_neg,
            roc=roc,
            roc_short=roc_short,
            volatility=volatility,
            atr=atr,
            is_above_upper_band=current_price > ewma_upper,
            is_below_lower_band=current_price < ewma_lower,
        )
    
    def get_indicators(self, market_id: str) -> Optional[MarketIndicators]:
        """
        Get current indicators for a market.
        
        Args:
            market_id: The market's condition ID
            
        Returns:
            MarketIndicators or None if market not tracked
        """
        history = self._histories.get(market_id)
        return history.indicators if history else None
    
    def get_history(self, market_id: str) -> Optional[MarketHistory]:
        """
        Get full price history for a market.
        
        Args:
            market_id: The market's condition ID
            
        Returns:
            MarketHistory or None if market not tracked
        """
        return self._histories.get(market_id)
    
    def get_price_data(self, market_id: str) -> Tuple[List[datetime], List[float]]:
        """
        Get price data for charting.
        
        Args:
            market_id: The market's condition ID
            
        Returns:
            Tuple of (timestamps, prices)
        """
        history = self._histories.get(market_id)
        if history:
            return list(history.timestamps), list(history.prices)
        return [], []
    
    def get_band_data(self, market_id: str) -> Tuple[List[float], List[float], List[float]]:
        """
        Get EWMA band data for charting.
        
        Args:
            market_id: The market's condition ID
            
        Returns:
            Tuple of (means, upper_bands, lower_bands)
        """
        history = self._histories.get(market_id)
        if not history or not history.has_enough_data:
            return [], [], []
        
        prices = history.price_series
        
        # Calculate bands for all points
        ewma_mean = prices.ewm(span=self._ewma_span, adjust=False).mean()
        ewma_std = prices.ewm(span=self._ewma_span, adjust=False).std()
        
        upper = ewma_mean + (self._ewma_multiplier * ewma_std)
        lower = ewma_mean - (self._ewma_multiplier * ewma_std)
        
        return list(ewma_mean), list(upper), list(lower)
    
    def get_cusum_data(self, market_id: str) -> Tuple[List[float], List[float]]:
        """
        Get CUSUM data for charting.
        
        Args:
            market_id: The market's condition ID
            
        Returns:
            Tuple of (cusum_positive, cusum_negative) histories
        """
        history = self._histories.get(market_id)
        if not history or not history.has_enough_data:
            return [], []
        
        prices = history.price_series
        target = prices.ewm(span=self._ewma_span, adjust=False).mean()
        
        cusum_pos_history = []
        cusum_neg_history = []
        
        cusum_pos = 0.0
        cusum_neg = 0.0
        
        for i, (price, t) in enumerate(zip(prices, target)):
            deviation = price - t
            cusum_pos = max(0, cusum_pos + deviation - self._cusum_slack)
            cusum_neg = max(0, cusum_neg - deviation - self._cusum_slack)
            cusum_pos_history.append(cusum_pos)
            cusum_neg_history.append(cusum_neg)
        
        return cusum_pos_history, cusum_neg_history
    
    def is_tracking(self, market_id: str) -> bool:
        """Check if a market is being tracked."""
        return market_id in self._histories
    
    def remove_market(self, market_id: str):
        """Stop tracking a market."""
        self._histories.pop(market_id, None)
    
    def clear(self):
        """Clear all tracked markets."""
        self._histories.clear()
    
    @property
    def tracked_markets(self) -> List[str]:
        """Get list of tracked market IDs."""
        return list(self._histories.keys())
    
    @property
    def market_count(self) -> int:
        """Number of markets being tracked."""
        return len(self._histories)


if __name__ == "__main__":
    # Test the tracker with simulated data
    import random
    
    print("Testing Price Tracker...")
    print("=" * 60)
    
    tracker = PriceTracker()
    
    # Create a mock market snapshot
    from src.core.client import Market
    
    mock_market = Market(
        id="test_market_123",
        question="Will Bitcoin reach $100k?",
        description="Test market",
        end_date=None,
        volume_24h=100000,
        liquidity=50000,
        yes_token_id="token_yes_123",
        no_token_id="token_no_123",
        yes_price=0.5,
        no_price=0.5,
    )
    
    # Simulate price series with a spike
    print("\nSimulating price data with spike...")
    base_price = 0.5
    prices = []
    
    # Normal period
    for i in range(30):
        noise = random.gauss(0, 0.005)
        price = base_price + noise
        prices.append(price)
    
    # Spike!
    print("SPIKE INJECTION at t=30")
    spike_price = base_price + 0.05  # 10% spike
    for i in range(5):
        prices.append(spike_price - i * 0.005)  # Spike then decay
    
    # Recovery
    for i in range(25):
        noise = random.gauss(0, 0.005)
        price = base_price + 0.02 + noise  # Settle slightly higher
        prices.append(price)
    
    # Feed prices to tracker
    for i, price in enumerate(prices):
        mock_snapshot = MarketSnapshot(
            market=mock_market,
            orderbook=None,
            timestamp=datetime.now(),
        )
        # Override price
        mock_market.yes_price = price
        
        indicators = tracker.update(mock_snapshot)
        
        if i >= 10 and i % 10 == 0:
            print(f"\nt={i}: Price={price:.4f}")
            print(f"  EWMA Mean: {indicators.ewma_mean:.4f}")
            print(f"  Upper Band: {indicators.ewma_upper_band:.4f}")
            print(f"  Lower Band: {indicators.ewma_lower_band:.4f}")
            print(f"  CUSUM+: {indicators.cusum_positive:.4f}")
            print(f"  CUSUM-: {indicators.cusum_negative:.4f}")
            print(f"  ROC: {indicators.roc:+.2f}%")
            print(f"  Above Upper: {indicators.is_above_upper_band}")
    
    # Final state
    final = tracker.get_indicators("test_market_123")
    print("\n" + "=" * 60)
    print("FINAL INDICATORS:")
    print(f"  Price: {final.current_price:.4f}")
    print(f"  Band Position: {final.band_position:.2f}")
    print(f"  Momentum: {final.momentum_direction}")