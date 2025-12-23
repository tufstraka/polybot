"""
Spike Detector Module
=====================

The heart of the trading algorithm. Implements multi-layer detection:
1. CUSUM - Regime change detection
2. EWMA Bands - Adaptive volatility bands
3. ROC - Momentum confirmation
4. Liquidity - Execution feasibility check

A signal is only generated when ALL layers agree.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import logging

from src.config.settings import get_settings, Settings
from src.core.tracker import MarketIndicators, PriceTracker
from src.core.scanner import MarketSnapshot


logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Direction of trading signal."""
    BUY = "BUY"    # Price spiked down, expect recovery
    SELL = "SELL"  # Price spiked up, expect mean reversion
    NONE = "NONE"


class RegimeChange(str, Enum):
    """Type of regime change detected by CUSUM."""
    UPWARD = "UPWARD"    # Price trending up
    DOWNWARD = "DOWNWARD"  # Price trending down
    NONE = "NONE"


@dataclass
class LayerResult:
    """Result from a single detection layer."""
    name: str
    passed: bool
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    
    def __str__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.name}: {self.value:.4f} (threshold: {self.threshold:.4f})"


@dataclass
class DetectionResult:
    """
    Complete result from spike detection.
    
    Contains results from all layers and the final signal.
    """
    market_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Layer results
    cusum_result: Optional[LayerResult] = None
    ewma_result: Optional[LayerResult] = None
    roc_result: Optional[LayerResult] = None
    liquidity_result: Optional[LayerResult] = None
    
    # Final signal
    signal: SignalDirection = SignalDirection.NONE
    confidence: float = 0.0  # 0-1
    
    # Additional data for trade execution
    suggested_entry_price: float = 0.0
    suggested_stop_loss: float = 0.0
    suggested_take_profit: float = 0.0
    
    @property
    def all_layers_passed(self) -> bool:
        """Check if all layers passed."""
        layers = [
            self.cusum_result,
            self.ewma_result,
            self.roc_result,
            self.liquidity_result,
        ]
        return all(layer.passed for layer in layers if layer is not None)
    
    @property
    def layers_passed_count(self) -> int:
        """Count how many layers passed."""
        layers = [
            self.cusum_result,
            self.ewma_result,
            self.roc_result,
            self.liquidity_result,
        ]
        return sum(1 for layer in layers if layer and layer.passed)
    
    @property
    def is_signal(self) -> bool:
        """Check if this is a valid trading signal."""
        return self.signal != SignalDirection.NONE and self.all_layers_passed
    
    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [f"Detection Result for {self.market_id}:"]
        
        for layer in [self.cusum_result, self.ewma_result, self.roc_result, self.liquidity_result]:
            if layer:
                lines.append(f"  {layer}")
        
        lines.append(f"  Signal: {self.signal.value} (confidence: {self.confidence:.0%})")
        
        if self.is_signal:
            lines.append(f"  Entry: {self.suggested_entry_price:.4f}")
            lines.append(f"  Stop Loss: {self.suggested_stop_loss:.4f}")
            lines.append(f"  Take Profit: {self.suggested_take_profit:.4f}")
        
        return "\n".join(lines)


@dataclass
class TradingSignal:
    """
    A trading signal ready for execution.
    
    This is what gets passed to the order executor.
    """
    market_id: str
    token_id: str
    direction: SignalDirection
    confidence: float
    
    # Price levels
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Indicator values at signal time (for logging/analysis)
    cusum_value: float
    roc_value: float
    volatility: float
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return (
            f"Signal({self.direction.value} @ {self.entry_price:.4f}, "
            f"SL={self.stop_loss:.4f}, TP={self.take_profit:.4f})"
        )


class SpikeDetector:
    """
    Multi-layer spike detection system.
    
    Uses four layers of confirmation:
    1. CUSUM - Detects regime changes (is price drifting?)
    2. EWMA Bands - Price outside normal range? (adaptive)
    3. ROC - Is there actual momentum in this direction?
    4. Liquidity - Can we actually execute this trade?
    
    All layers must pass for a signal to be generated.
    
    Usage:
        detector = SpikeDetector(tracker)
        result = detector.check(snapshot)
        if result.is_signal:
            signal = detector.create_signal(result, snapshot)
    """
    
    def __init__(
        self,
        tracker: PriceTracker,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the detector.
        
        Args:
            tracker: PriceTracker instance for indicator access
            settings: Optional Settings object
        """
        self.tracker = tracker
        self.settings = settings or get_settings()
        
        # Detection parameters from settings
        self._cusum_threshold = self.settings.detection.cusum_threshold
        self._cusum_slack = self.settings.detection.cusum_slack
        self._ewma_multiplier = self.settings.detection.ewma_multiplier
        self._roc_threshold = self.settings.detection.roc_threshold
        self._min_liquidity = self.settings.filters.min_liquidity_usd
        self._max_spread = self.settings.filters.max_spread_percent
        
        # Risk parameters for signal generation
        self._atr_stop_mult = self.settings.risk.atr_stop_multiplier
        self._atr_profit_mult = self.settings.risk.atr_profit_multiplier
        
        # Track recent signals to avoid duplicates
        self._recent_signals: Dict[str, datetime] = {}
        self._cooldown_seconds = self.settings.risk.cooldown_seconds
    
    def check(self, snapshot: MarketSnapshot) -> DetectionResult:
        """
        Check a market snapshot for spike signals.
        
        This is the main detection method. It runs all four layers
        and returns a DetectionResult.
        
        Args:
            snapshot: Current market snapshot
            
        Returns:
            DetectionResult with all layer results and final signal
        """
        market_id = snapshot.market.id
        
        # Get indicators from tracker
        indicators = self.tracker.get_indicators(market_id)
        
        if not indicators:
            return DetectionResult(
                market_id=market_id,
                cusum_result=LayerResult("CUSUM", False, message="No indicators available"),
            )
        
        # Run all detection layers
        cusum_result = self._check_cusum(indicators)
        ewma_result = self._check_ewma_bands(indicators)
        roc_result = self._check_roc(indicators, cusum_result)
        liquidity_result = self._check_liquidity(snapshot)
        
        # Determine signal direction based on layers
        signal = self._determine_signal(
            cusum_result, ewma_result, roc_result, liquidity_result, indicators
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            cusum_result, ewma_result, roc_result, liquidity_result
        )
        
        # Calculate suggested price levels
        entry_price = indicators.current_price
        stop_loss, take_profit = self._calculate_exit_levels(
            entry_price, indicators.volatility, signal
        )
        
        return DetectionResult(
            market_id=market_id,
            cusum_result=cusum_result,
            ewma_result=ewma_result,
            roc_result=roc_result,
            liquidity_result=liquidity_result,
            signal=signal,
            confidence=confidence,
            suggested_entry_price=entry_price,
            suggested_stop_loss=stop_loss,
            suggested_take_profit=take_profit,
        )
    
    def _check_cusum(self, indicators: MarketIndicators) -> LayerResult:
        """
        Layer 1: CUSUM Regime Change Detection
        
        CUSUM accumulates deviations from the mean. When it exceeds
        the threshold, we've detected a regime change.
        """
        # Use the larger of positive or negative CUSUM
        cusum_value = max(indicators.cusum_positive, indicators.cusum_negative)
        
        passed = cusum_value >= self._cusum_threshold
        
        # Determine direction
        if passed:
            if indicators.cusum_positive > indicators.cusum_negative:
                message = "Upward regime shift detected"
            else:
                message = "Downward regime shift detected"
        else:
            message = "No regime change"
        
        return LayerResult(
            name="CUSUM",
            passed=passed,
            value=cusum_value,
            threshold=self._cusum_threshold,
            message=message,
        )
    
    def _check_ewma_bands(self, indicators: MarketIndicators) -> LayerResult:
        """
        Layer 2: EWMA Adaptive Volatility Bands
        
        Price must be outside the adaptive bands to signal
        an unusual move.
        """
        # Calculate how far outside the bands
        if indicators.is_above_upper_band:
            distance = indicators.current_price - indicators.ewma_upper_band
            passed = True
            message = f"Price above upper band by {distance:.4f}"
        elif indicators.is_below_lower_band:
            distance = indicators.ewma_lower_band - indicators.current_price
            passed = True
            message = f"Price below lower band by {distance:.4f}"
        else:
            distance = 0.0
            passed = False
            message = "Price within bands"
        
        return LayerResult(
            name="EWMA Bands",
            passed=passed,
            value=indicators.band_position,  # -1 to +1
            threshold=1.0,  # Must exceed bands
            message=message,
        )
    
    def _check_roc(
        self,
        indicators: MarketIndicators,
        cusum_result: LayerResult,
    ) -> LayerResult:
        """
        Layer 3: Rate of Change (Momentum)
        
        Momentum must confirm the direction of the detected regime change.
        This filters out false signals where price moved but has no momentum.
        """
        roc = abs(indicators.roc)
        passed = roc >= self._roc_threshold
        
        # Check if momentum direction matches CUSUM
        if passed:
            if indicators.cusum_positive > indicators.cusum_negative:
                # Expecting upward move
                if indicators.roc > 0:
                    message = f"Momentum confirms upward move (+{indicators.roc:.2f}%)"
                else:
                    message = "Momentum conflicts with CUSUM"
                    passed = False
            else:
                # Expecting downward move
                if indicators.roc < 0:
                    message = f"Momentum confirms downward move ({indicators.roc:.2f}%)"
                else:
                    message = "Momentum conflicts with CUSUM"
                    passed = False
        else:
            message = f"Insufficient momentum ({indicators.roc:.2f}%)"
        
        return LayerResult(
            name="ROC",
            passed=passed,
            value=indicators.roc,
            threshold=self._roc_threshold,
            message=message,
        )
    
    def _check_liquidity(self, snapshot: MarketSnapshot) -> LayerResult:
        """
        Layer 4: Liquidity Check
        
        Ensures we can actually execute the trade without
        excessive slippage.
        """
        if not snapshot.orderbook:
            return LayerResult(
                name="Liquidity",
                passed=False,
                value=0.0,
                threshold=self._min_liquidity,
                message="No orderbook data",
            )
        
        # Check minimum liquidity
        min_depth = min(snapshot.bid_depth, snapshot.ask_depth)
        liquidity_ok = min_depth >= self._min_liquidity
        
        # Check spread
        spread_ok = True
        if snapshot.spread_percent:
            spread_ok = snapshot.spread_percent <= self._max_spread
        
        passed = liquidity_ok and spread_ok
        
        if not liquidity_ok:
            message = f"Insufficient liquidity (${min_depth:.2f})"
        elif not spread_ok:
            message = f"Spread too wide ({snapshot.spread_percent:.2f}%)"
        else:
            message = f"Good liquidity (${min_depth:.2f}, spread {snapshot.spread_percent:.2f}%)"
        
        return LayerResult(
            name="Liquidity",
            passed=passed,
            value=min_depth,
            threshold=self._min_liquidity,
            message=message,
        )
    
    def _determine_signal(
        self,
        cusum: LayerResult,
        ewma: LayerResult,
        roc: LayerResult,
        liquidity: LayerResult,
        indicators: MarketIndicators,
    ) -> SignalDirection:
        """
        Determine the final signal direction.
        
        For mean reversion strategy:
        - If price spiked UP (above upper band), we SELL expecting it to come down
        - If price spiked DOWN (below lower band), we BUY expecting it to come up
        """
        # All layers must pass
        if not all([cusum.passed, ewma.passed, roc.passed, liquidity.passed]):
            return SignalDirection.NONE
        
        # Determine direction based on band position (mean reversion)
        if indicators.is_above_upper_band:
            # Price spiked up - sell expecting mean reversion
            return SignalDirection.SELL
        elif indicators.is_below_lower_band:
            # Price spiked down - buy expecting mean reversion
            return SignalDirection.BUY
        
        return SignalDirection.NONE
    
    def _calculate_confidence(
        self,
        cusum: LayerResult,
        ewma: LayerResult,
        roc: LayerResult,
        liquidity: LayerResult,
    ) -> float:
        """
        Calculate signal confidence (0-1).
        
        Higher confidence when:
        - CUSUM well above threshold
        - Price far outside bands
        - Strong momentum
        - Good liquidity
        """
        if not all([cusum.passed, ewma.passed, roc.passed, liquidity.passed]):
            return 0.0
        
        # Calculate confidence components
        cusum_conf = min(1.0, cusum.value / (self._cusum_threshold * 2))
        band_conf = min(1.0, abs(ewma.value))
        roc_conf = min(1.0, abs(roc.value) / (self._roc_threshold * 2))
        liq_conf = min(1.0, liquidity.value / (self._min_liquidity * 2))
        
        # Weight and combine
        confidence = (
            cusum_conf * 0.3 +
            band_conf * 0.3 +
            roc_conf * 0.2 +
            liq_conf * 0.2
        )
        
        return confidence
    
    def _calculate_exit_levels(
        self,
        entry_price: float,
        volatility: float,
        signal: SignalDirection,
    ) -> tuple:
        """
        Calculate stop loss and take profit levels.
        
        Uses ATR (volatility) based exits that adapt to market conditions.
        """
        if signal == SignalDirection.NONE:
            return entry_price, entry_price
        
        # Ensure minimum volatility for calculations
        vol = max(volatility, 0.005)
        
        if signal == SignalDirection.BUY:
            # Buying - stop below, target above
            stop_loss = entry_price - (vol * self._atr_stop_mult)
            take_profit = entry_price + (vol * self._atr_profit_mult)
        else:
            # Selling - stop above, target below
            stop_loss = entry_price + (vol * self._atr_stop_mult)
            take_profit = entry_price - (vol * self._atr_profit_mult)
        
        # Clamp to valid price range (0-1 for prediction markets)
        stop_loss = max(0.01, min(0.99, stop_loss))
        take_profit = max(0.01, min(0.99, take_profit))
        
        return stop_loss, take_profit
    
    def is_in_cooldown(self, market_id: str) -> bool:
        """
        Check if a market is in cooldown period.
        
        We don't trade the same market repeatedly in quick succession.
        """
        if market_id not in self._recent_signals:
            return False
        
        last_signal_time = self._recent_signals[market_id]
        elapsed = (datetime.now() - last_signal_time).total_seconds()
        
        return elapsed < self._cooldown_seconds
    
    def create_signal(
        self,
        result: DetectionResult,
        snapshot: MarketSnapshot,
    ) -> Optional[TradingSignal]:
        """
        Create a trading signal from a detection result.
        
        Also handles cooldown checking and signal recording.
        
        Args:
            result: DetectionResult that passed all checks
            snapshot: The market snapshot
            
        Returns:
            TradingSignal or None if in cooldown
        """
        if not result.is_signal:
            return None
        
        market_id = snapshot.market.id
        
        # Check cooldown
        if self.is_in_cooldown(market_id):
            logger.debug(f"Market {market_id} in cooldown, skipping signal")
            return None
        
        # Record signal time
        self._recent_signals[market_id] = datetime.now()
        
        # Get indicators for signal metadata
        indicators = self.tracker.get_indicators(market_id)
        
        signal = TradingSignal(
            market_id=market_id,
            token_id=snapshot.market.yes_token_id or "",
            direction=result.signal,
            confidence=result.confidence,
            entry_price=result.suggested_entry_price,
            stop_loss=result.suggested_stop_loss,
            take_profit=result.suggested_take_profit,
            cusum_value=max(indicators.cusum_positive, indicators.cusum_negative) if indicators else 0,
            roc_value=indicators.roc if indicators else 0,
            volatility=indicators.volatility if indicators else 0,
        )
        
        logger.info(f"Generated signal: {signal}")
        return signal
    
    def check_and_signal(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """
        Convenience method: check for spike and create signal if found.
        
        Args:
            snapshot: Market snapshot to check
            
        Returns:
            TradingSignal or None
        """
        result = self.check(snapshot)
        if result.is_signal:
            return self.create_signal(result, snapshot)
        return None
    
    def clear_cooldowns(self):
        """Clear all cooldown timers."""
        self._recent_signals.clear()
    
    def get_cooldown_markets(self) -> List[str]:
        """Get list of markets currently in cooldown."""
        now = datetime.now()
        return [
            market_id
            for market_id, signal_time in self._recent_signals.items()
            if (now - signal_time).total_seconds() < self._cooldown_seconds
        ]


if __name__ == "__main__":
    # Test the detector with simulated data
    import random
    from src.core.tracker import PriceTracker
    from src.core.client import Market, Orderbook, OrderbookLevel
    
    print("Testing Spike Detector...")
    print("=" * 60)
    
    # Create tracker and detector
    tracker = PriceTracker()
    detector = SpikeDetector(tracker)
    
    # Create mock market
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
    
    # Create mock orderbook
    mock_orderbook = Orderbook(
        market_id="test_market_123",
        token_id="token_yes_123",
        bids=[OrderbookLevel(0.49, 100), OrderbookLevel(0.48, 200)],
        asks=[OrderbookLevel(0.51, 100), OrderbookLevel(0.52, 200)],
    )
    
    # Simulate price series
    print("\nSimulating price data...")
    base_price = 0.5
    
    # Normal period - build up history
    print("\n[Phase 1: Normal prices - building history]")
    for i in range(30):
        noise = random.gauss(0, 0.005)
        price = base_price + noise
        mock_market.yes_price = price
        
        snapshot = MarketSnapshot(
            market=mock_market,
            orderbook=mock_orderbook,
        )
        tracker.update(snapshot)
    
    print(f"Built {tracker.market_count} market(s) with history")
    
    # Inject a spike
    print("\n[Phase 2: SPIKE - price jumps up]")
    spike_price = base_price + 0.06  # Big spike up
    mock_market.yes_price = spike_price
    
    # Update orderbook to reflect spike
    mock_orderbook.bids = [OrderbookLevel(0.55, 100), OrderbookLevel(0.54, 200)]
    mock_orderbook.asks = [OrderbookLevel(0.57, 100), OrderbookLevel(0.58, 200)]
    
    snapshot = MarketSnapshot(
        market=mock_market,
        orderbook=mock_orderbook,
    )
    tracker.update(snapshot)
    
    # Check for signal
    result = detector.check(snapshot)
    print(f"\n{result.summary()}")
    
    if result.is_signal:
        signal = detector.create_signal(result, snapshot)
        print(f"\n🚨 SIGNAL GENERATED: {signal}")
    else:
        print("\n❌ No signal generated")
    
    # Check cooldown
    print(f"\nCooldown active for market: {detector.is_in_cooldown('test_market_123')}")