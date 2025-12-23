"""
Ensemble Signal Generator
=========================

Combines multiple signal sources into unified trading recommendations:
1. Technical Indicators (CUSUM, EWMA, ROC)
2. AI Market Analysis (via Bedrock)
3. Sentiment Analysis
4. Mean Reversion Signals

Uses configurable weighting to blend signals based on autonomy level.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from src.config.settings import Settings, get_settings
from src.ai.bedrock_client import BedrockClient, ModelResponse
from src.ai.prompts import PromptTemplates, MarketContext, TradingContext


logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Trading signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalSource(str, Enum):
    """Source of the trading signal."""
    TECHNICAL = "technical"
    AI_ANALYSIS = "ai_analysis"
    SENTIMENT = "sentiment"
    MEAN_REVERSION = "mean_reversion"
    ENSEMBLE = "ensemble"


@dataclass
class ComponentSignal:
    """
    Signal from a single component/source.
    """
    source: SignalSource
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    weight: float  # Contribution weight to ensemble
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted contribution to ensemble."""
        direction_multiplier = {
            SignalDirection.BUY: 1.0,
            SignalDirection.SELL: -1.0,
            SignalDirection.HOLD: 0.0,
        }
        return self.confidence * self.weight * direction_multiplier[self.direction]


@dataclass
class EnsembleSignal:
    """
    Combined signal from all sources.
    
    Represents the final trading recommendation.
    """
    direction: SignalDirection
    confidence: float
    
    # Component signals
    components: List[ComponentSignal] = field(default_factory=list)
    
    # Trade parameters
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    recommended_size: float = 0.0
    
    # AI reasoning
    ai_reasoning: str = ""
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    market_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal should trigger a trade."""
        return self.direction != SignalDirection.HOLD and self.confidence >= 0.6
    
    @property
    def signal_strength(self) -> str:
        """Categorize signal strength."""
        if self.confidence >= 0.8:
            return "strong"
        elif self.confidence >= 0.6:
            return "moderate"
        elif self.confidence >= 0.4:
            return "weak"
        else:
            return "none"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "direction": self.direction.value,
            "confidence": self.confidence,
            "signal_strength": self.signal_strength,
            "is_actionable": self.is_actionable,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "recommended_size": self.recommended_size,
            "ai_reasoning": self.ai_reasoning,
            "components": [
                {
                    "source": c.source.value,
                    "direction": c.direction.value,
                    "confidence": c.confidence,
                    "weight": c.weight,
                    "reasoning": c.reasoning,
                }
                for c in self.components
            ],
            "market_id": self.market_id,
            "timestamp": self.timestamp.isoformat(),
        }


class EnsembleSignalGenerator:
    """
    Generates trading signals by combining multiple analysis methods.
    
    Signal Sources:
    1. Technical: CUSUM, EWMA bands, ROC momentum
    2. AI Analysis: LLM-based market understanding
    3. Sentiment: Market sentiment from description analysis
    4. Mean Reversion: Statistical extremes detection
    
    The ensemble uses configurable weights based on the autonomy_level setting:
    - autonomy_level = 0.0: Pure technical signals
    - autonomy_level = 0.5: 50/50 blend
    - autonomy_level = 1.0: AI-dominant decisions
    
    Usage:
        generator = EnsembleSignalGenerator()
        signal = await generator.generate(market_context, trading_context)
        if signal.is_actionable:
            execute_trade(signal)
    """
    
    # Base weights for different signal sources
    BASE_WEIGHTS = {
        SignalSource.TECHNICAL: 0.40,
        SignalSource.AI_ANALYSIS: 0.35,
        SignalSource.SENTIMENT: 0.10,
        SignalSource.MEAN_REVERSION: 0.15,
    }
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the signal generator.
        
        Args:
            bedrock_client: Optional pre-initialized Bedrock client
            settings: Optional settings object
        """
        self.settings = settings or get_settings()
        self.bedrock = bedrock_client or BedrockClient(settings=self.settings)
        
        # Adjust weights based on autonomy level
        self._weights = self._calculate_weights()
        
        # Cache for recent AI analyses (avoid repeated API calls)
        self._analysis_cache: Dict[str, Tuple[datetime, Dict]] = {}
        self._cache_ttl_seconds = 60  # Cache AI results for 60 seconds
    
    def _calculate_weights(self) -> Dict[SignalSource, float]:
        """
        Calculate signal source weights based on autonomy level.
        
        Higher autonomy = more weight to AI signals
        """
        autonomy = self.settings.ai.autonomy_level
        
        # Base weights
        weights = dict(self.BASE_WEIGHTS)
        
        # Adjust based on autonomy
        # At autonomy=0: technical=0.6, AI=0.1, sentiment=0.1, mean_rev=0.2
        # At autonomy=1: technical=0.2, AI=0.6, sentiment=0.1, mean_rev=0.1
        
        ai_boost = autonomy * 0.25  # +0 to +0.25
        tech_reduction = autonomy * 0.2  # -0 to -0.2
        
        weights[SignalSource.TECHNICAL] = max(0.2, weights[SignalSource.TECHNICAL] - tech_reduction)
        weights[SignalSource.AI_ANALYSIS] = min(0.6, weights[SignalSource.AI_ANALYSIS] + ai_boost)
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        logger.debug(f"Signal weights (autonomy={autonomy}): {weights}")
        return weights
    
    async def generate(
        self,
        market: MarketContext,
        trading: TradingContext,
        technical_result: Optional[Dict] = None,
    ) -> EnsembleSignal:
        """
        Generate ensemble trading signal.
        
        Args:
            market: Market context with price and indicator data
            trading: Trading context with portfolio state
            technical_result: Optional pre-computed technical analysis
            
        Returns:
            EnsembleSignal with combined recommendation
        """
        components: List[ComponentSignal] = []
        
        # 1. Technical Signal
        tech_signal = self._generate_technical_signal(market, technical_result)
        components.append(tech_signal)
        
        # 2. Mean Reversion Signal
        mr_signal = self._generate_mean_reversion_signal(market)
        components.append(mr_signal)
        
        # 3. AI Analysis Signal (if enabled)
        if self.settings.is_ai_enabled():
            try:
                ai_signal = await self._generate_ai_signal(market, trading, tech_signal)
                components.append(ai_signal)
                
                # Also get sentiment if enabled
                if self.settings.ai.sentiment_enabled:
                    sentiment_signal = await self._generate_sentiment_signal(market)
                    components.append(sentiment_signal)
            except Exception as e:
                logger.warning(f"AI signal generation failed: {e}")
                # Add placeholder AI signal with neutral stance
                components.append(ComponentSignal(
                    source=SignalSource.AI_ANALYSIS,
                    direction=SignalDirection.HOLD,
                    confidence=0.0,
                    weight=self._weights[SignalSource.AI_ANALYSIS],
                    reasoning="AI analysis unavailable",
                ))
        
        # Combine signals into ensemble
        ensemble = self._combine_signals(components, market)
        
        # Calculate trade parameters
        ensemble = self._calculate_trade_params(ensemble, market, trading)
        
        return ensemble
    
    def _generate_technical_signal(
        self,
        market: MarketContext,
        technical_result: Optional[Dict] = None,
    ) -> ComponentSignal:
        """
        Generate signal from technical indicators.
        
        Uses CUSUM, EWMA bands, and ROC.
        """
        direction = SignalDirection.HOLD
        confidence = 0.0
        reasons = []
        
        # CUSUM signal
        cusum_threshold = self.settings.detection.cusum_threshold
        if market.cusum_positive > cusum_threshold:
            cusum_signal = "bullish"
            reasons.append(f"CUSUM+ breach ({market.cusum_positive:.4f})")
        elif market.cusum_negative < -cusum_threshold:
            cusum_signal = "bearish"
            reasons.append(f"CUSUM- breach ({market.cusum_negative:.4f})")
        else:
            cusum_signal = "neutral"
        
        # EWMA band signal
        band_pos = market.band_position()
        if band_pos == "above_upper":
            band_signal = "overbought"
            reasons.append("Price above upper band (overbought)")
        elif band_pos == "below_lower":
            band_signal = "oversold"
            reasons.append("Price below lower band (oversold)")
        else:
            band_signal = "neutral"
        
        # ROC momentum
        roc_threshold = self.settings.detection.roc_threshold
        if market.roc > roc_threshold:
            roc_signal = "bullish"
            reasons.append(f"Strong momentum ({market.roc:+.2f}%)")
        elif market.roc < -roc_threshold:
            roc_signal = "bearish"
            reasons.append(f"Weak momentum ({market.roc:+.2f}%)")
        else:
            roc_signal = "neutral"
        
        # Combine technical signals
        bullish_count = sum(1 for s in [cusum_signal, roc_signal] if s == "bullish")
        bearish_count = sum(1 for s in [cusum_signal, roc_signal] if s == "bearish")
        
        # Add band signal consideration (contrarian for extreme bands)
        if band_signal == "oversold":
            bullish_count += 0.5
        elif band_signal == "overbought":
            bearish_count += 0.5
        
        if bullish_count > bearish_count and bullish_count >= 1.5:
            direction = SignalDirection.BUY
            confidence = min(0.9, 0.5 + (bullish_count * 0.15))
        elif bearish_count > bullish_count and bearish_count >= 1.5:
            direction = SignalDirection.SELL
            confidence = min(0.9, 0.5 + (bearish_count * 0.15))
        else:
            direction = SignalDirection.HOLD
            confidence = 0.3
        
        return ComponentSignal(
            source=SignalSource.TECHNICAL,
            direction=direction,
            confidence=confidence,
            weight=self._weights[SignalSource.TECHNICAL],
            reasoning="; ".join(reasons) if reasons else "No significant signals",
            metadata={
                "cusum_signal": cusum_signal,
                "band_signal": band_signal,
                "roc_signal": roc_signal,
            },
        )
    
    def _generate_mean_reversion_signal(self, market: MarketContext) -> ComponentSignal:
        """
        Generate signal based on mean reversion theory.
        
        Extreme prices tend to revert to mean.
        """
        direction = SignalDirection.HOLD
        confidence = 0.0
        reasoning = ""
        
        # Calculate deviation from EWMA
        deviation = market.current_price - market.ewma_price
        deviation_pct = abs(deviation) / market.ewma_price if market.ewma_price > 0 else 0
        
        # Strong deviation = mean reversion opportunity
        if deviation_pct > 0.10:  # >10% deviation
            confidence = min(0.8, 0.5 + deviation_pct)
            if deviation > 0:
                direction = SignalDirection.SELL  # Overbought, expect reversion down
                reasoning = f"Price {deviation_pct*100:.1f}% above EWMA - expect reversion"
            else:
                direction = SignalDirection.BUY  # Oversold, expect reversion up
                reasoning = f"Price {deviation_pct*100:.1f}% below EWMA - expect reversion"
        elif deviation_pct > 0.05:  # 5-10% deviation
            confidence = 0.4 + (deviation_pct * 2)
            if deviation > 0:
                direction = SignalDirection.SELL
                reasoning = f"Moderate overbought ({deviation_pct*100:.1f}% above EWMA)"
            else:
                direction = SignalDirection.BUY
                reasoning = f"Moderate oversold ({deviation_pct*100:.1f}% below EWMA)"
        else:
            reasoning = "Price near fair value"
            confidence = 0.2
        
        # Also consider extreme prices (near 0 or 1)
        if market.current_price > 0.90:
            confidence *= 0.7  # Reduce confidence - limited upside
            reasoning += " (Warning: price near 1.0, limited upside)"
        elif market.current_price < 0.10:
            confidence *= 0.7  # Reduce confidence - limited downside
            reasoning += " (Warning: price near 0.0, limited downside)"
        
        return ComponentSignal(
            source=SignalSource.MEAN_REVERSION,
            direction=direction,
            confidence=confidence,
            weight=self._weights[SignalSource.MEAN_REVERSION],
            reasoning=reasoning,
            metadata={
                "deviation_pct": deviation_pct,
                "ewma_price": market.ewma_price,
            },
        )
    
    async def _generate_ai_signal(
        self,
        market: MarketContext,
        trading: TradingContext,
        tech_signal: ComponentSignal,
    ) -> ComponentSignal:
        """
        Generate signal using AI market analysis.
        """
        # Check cache first
        cache_key = f"{market.market_id}_{int(datetime.utcnow().timestamp() / self._cache_ttl_seconds)}"
        if cache_key in self._analysis_cache:
            cached_time, cached_result = self._analysis_cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self._cache_ttl_seconds:
                logger.debug(f"Using cached AI analysis for {market.market_id}")
                return self._parse_ai_trading_response(cached_result, market)
        
        # Generate AI analysis
        tech_signal_str = f"{tech_signal.direction.value} (confidence: {tech_signal.confidence:.0%})"
        
        prompt = PromptTemplates.trading_decision_prompt(
            market=market,
            trading=trading,
            technical_signal=tech_signal_str,
        )
        
        response = await self.bedrock.generate_async(
            prompt=prompt,
            system_prompt=PromptTemplates.system_prompt(),
            max_tokens=self.settings.ai.max_tokens,
            temperature=self.settings.ai.temperature,
        )
        
        # Parse response
        result = PromptTemplates.parse_json_response(response.content)
        
        # Cache the result
        self._analysis_cache[cache_key] = (datetime.utcnow(), result)
        
        return self._parse_ai_trading_response(result, market)
    
    def _parse_ai_trading_response(
        self,
        result: Dict,
        market: MarketContext,
    ) -> ComponentSignal:
        """Parse AI trading decision response into ComponentSignal."""
        if "error" in result:
            return ComponentSignal(
                source=SignalSource.AI_ANALYSIS,
                direction=SignalDirection.HOLD,
                confidence=0.0,
                weight=self._weights[SignalSource.AI_ANALYSIS],
                reasoning=f"AI parse error: {result.get('error')}",
            )
        
        # Extract action
        action = result.get("action", "HOLD").upper()
        if action == "BUY":
            direction = SignalDirection.BUY
        elif action == "SELL":
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        
        confidence = float(result.get("confidence", 0.5))
        
        # Build reasoning
        reasons = result.get("key_reasons", [])
        reasoning = result.get("reasoning", "")
        if reasons:
            reasoning = f"{reasoning} | Factors: {', '.join(reasons[:3])}"
        
        return ComponentSignal(
            source=SignalSource.AI_ANALYSIS,
            direction=direction,
            confidence=confidence,
            weight=self._weights[SignalSource.AI_ANALYSIS],
            reasoning=reasoning,
            metadata={
                "edge_estimate": result.get("edge_estimate", 0),
                "risk_reward": result.get("risk_reward_ratio", 0),
                "time_horizon": result.get("time_horizon", "unknown"),
                "entry_price": result.get("entry_price", market.current_price),
                "stop_loss": result.get("stop_loss", 0),
                "take_profit": result.get("take_profit", 0),
                "recommended_size": result.get("recommended_size", 0),
            },
        )
    
    async def _generate_sentiment_signal(self, market: MarketContext) -> ComponentSignal:
        """
        Generate signal from sentiment analysis of market question.
        """
        prompt = PromptTemplates.sentiment_analysis_prompt(
            question=market.question,
            description=market.description,
        )
        
        response = await self.bedrock.generate_async(
            prompt=prompt,
            system_prompt=PromptTemplates.system_prompt(),
            max_tokens=512,
            temperature=0.2,  # More deterministic for sentiment
        )
        
        result = PromptTemplates.parse_json_response(response.content)
        
        if "error" in result:
            return ComponentSignal(
                source=SignalSource.SENTIMENT,
                direction=SignalDirection.HOLD,
                confidence=0.0,
                weight=self._weights[SignalSource.SENTIMENT],
                reasoning="Sentiment analysis failed",
            )
        
        # Convert sentiment to trading signal
        sentiment_score = float(result.get("sentiment_toward_yes", 0.5))
        base_rate = float(result.get("base_rate_estimate", 0.5))
        
        # If sentiment differs significantly from current price, signal opportunity
        price_diff = sentiment_score - market.current_price
        
        if abs(price_diff) > 0.15:  # >15% sentiment vs price difference
            if price_diff > 0:
                direction = SignalDirection.BUY
                confidence = min(0.7, 0.4 + abs(price_diff))
                reasoning = f"Public sentiment ({sentiment_score:.0%}) higher than price ({market.current_price:.0%})"
            else:
                direction = SignalDirection.SELL
                confidence = min(0.7, 0.4 + abs(price_diff))
                reasoning = f"Public sentiment ({sentiment_score:.0%}) lower than price ({market.current_price:.0%})"
        else:
            direction = SignalDirection.HOLD
            confidence = 0.3
            reasoning = "Sentiment aligned with price"
        
        return ComponentSignal(
            source=SignalSource.SENTIMENT,
            direction=direction,
            confidence=confidence,
            weight=self._weights[SignalSource.SENTIMENT],
            reasoning=reasoning,
            metadata={
                "sentiment_score": sentiment_score,
                "base_rate": base_rate,
                "category": result.get("category", "unknown"),
            },
        )
    
    def _combine_signals(
        self,
        components: List[ComponentSignal],
        market: MarketContext,
    ) -> EnsembleSignal:
        """
        Combine component signals into ensemble decision.
        
        Uses weighted voting with confidence scaling.
        """
        # Calculate weighted score
        total_score = sum(c.weighted_score for c in components)
        total_weight = sum(c.weight for c in components)
        
        if total_weight == 0:
            normalized_score = 0
        else:
            normalized_score = total_score / total_weight
        
        # Determine direction
        if normalized_score > 0.15:
            direction = SignalDirection.BUY
        elif normalized_score < -0.15:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        
        # Calculate ensemble confidence
        # Average confidence weighted by alignment with final direction
        aligned_confidence = 0.0
        aligned_weight = 0.0
        
        for c in components:
            if c.direction == direction:
                aligned_confidence += c.confidence * c.weight
                aligned_weight += c.weight
        
        if aligned_weight > 0:
            confidence = aligned_confidence / aligned_weight
        else:
            confidence = 0.0
        
        # Apply minimum confidence threshold
        min_confidence = self.settings.ai.min_confidence
        if confidence < min_confidence and direction != SignalDirection.HOLD:
            direction = SignalDirection.HOLD
            confidence = confidence  # Keep original for transparency
        
        # Build AI reasoning summary
        ai_components = [c for c in components if c.source == SignalSource.AI_ANALYSIS]
        ai_reasoning = ai_components[0].reasoning if ai_components else ""
        ai_analysis = ai_components[0].metadata if ai_components else {}
        
        return EnsembleSignal(
            direction=direction,
            confidence=confidence,
            components=components,
            market_id=market.market_id,
            ai_reasoning=ai_reasoning,
            ai_analysis=ai_analysis,
        )
    
    def _calculate_trade_params(
        self,
        signal: EnsembleSignal,
        market: MarketContext,
        trading: TradingContext,
    ) -> EnsembleSignal:
        """
        Calculate entry, stop loss, take profit, and size.
        """
        if signal.direction == SignalDirection.HOLD:
            return signal
        
        # Use AI suggestions if available
        ai_meta = signal.ai_analysis
        
        # Entry price (current or AI suggested)
        signal.entry_price = ai_meta.get("entry_price", market.current_price)
        
        # Stop loss
        stop_pct = self.settings.risk.stop_loss_percent / 100
        if signal.direction == SignalDirection.BUY:
            ai_stop = ai_meta.get("stop_loss", 0)
            if ai_stop and ai_stop < signal.entry_price:
                signal.stop_loss = ai_stop
            else:
                signal.stop_loss = signal.entry_price * (1 - stop_pct)
        else:
            ai_stop = ai_meta.get("stop_loss", 0)
            if ai_stop and ai_stop > signal.entry_price:
                signal.stop_loss = ai_stop
            else:
                signal.stop_loss = signal.entry_price * (1 + stop_pct)
        
        # Take profit
        tp_pct = self.settings.risk.take_profit_percent / 100
        if signal.direction == SignalDirection.BUY:
            ai_tp = ai_meta.get("take_profit", 0)
            if ai_tp and ai_tp > signal.entry_price:
                signal.take_profit = ai_tp
            else:
                signal.take_profit = min(0.99, signal.entry_price * (1 + tp_pct))
        else:
            ai_tp = ai_meta.get("take_profit", 0)
            if ai_tp and 0 < ai_tp < signal.entry_price:
                signal.take_profit = ai_tp
            else:
                signal.take_profit = max(0.01, signal.entry_price * (1 - tp_pct))
        
        # Position size - will be refined by Kelly Criterion module
        ai_size = ai_meta.get("recommended_size", 0)
        if ai_size and ai_size <= trading.max_position_size:
            signal.recommended_size = ai_size
        else:
            signal.recommended_size = min(
                trading.max_position_size,
                trading.available_capital * 0.1,  # Max 10% of capital
            )
        
        return signal
    
    def get_weights(self) -> Dict[str, float]:
        """Get current signal source weights."""
        return {k.value: v for k, v in self._weights.items()}


if __name__ == "__main__":
    # Test the signal generator
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        from src.ai.prompts import MarketContext, TradingContext
        
        market = MarketContext(
            market_id="test123",
            question="Will Bitcoin exceed $100,000 by December 31, 2024?",
            current_price=0.65,
            price_24h_ago=0.58,
            ewma_price=0.61,
            ewma_upper_band=0.70,
            ewma_lower_band=0.52,
            roc=3.5,
            cusum_positive=0.045,
            volume_24h=125000,
        )
        
        trading = TradingContext(
            current_capital=75.0,
            max_position_size=2.0,
            remaining_daily_risk=1.50,
        )
        
        generator = EnsembleSignalGenerator()
        
        print("Signal Generator Weights:")
        print(generator.get_weights())
        print()
        
        # Test technical signal only (without AI)
        tech_signal = generator._generate_technical_signal(market)
        print(f"Technical Signal: {tech_signal.direction.value} ({tech_signal.confidence:.0%})")
        print(f"Reasoning: {tech_signal.reasoning}")
        print()
        
        mr_signal = generator._generate_mean_reversion_signal(market)
        print(f"Mean Reversion Signal: {mr_signal.direction.value} ({mr_signal.confidence:.0%})")
        print(f"Reasoning: {mr_signal.reasoning}")
    
    asyncio.run(test())