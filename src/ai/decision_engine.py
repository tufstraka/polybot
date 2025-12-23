"""
AI Decision Engine
==================

Central orchestrator for AI-powered trading decisions.

Coordinates:
1. Bedrock client for LLM inference
2. Ensemble signal generator
3. Monte Carlo simulation
4. Kelly Criterion position sizing
5. Reasoning tracker

This is the main entry point for the bot to get AI-powered
trading recommendations.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.config.settings import Settings, get_settings
from src.ai.bedrock_client import BedrockClient, ModelResponse
from src.ai.prompts import PromptTemplates, MarketContext, TradingContext
from src.ai.signal_generator import (
    EnsembleSignalGenerator, EnsembleSignal, SignalDirection, ComponentSignal
)
from src.ai.monte_carlo import MonteCarloSimulator, SimulationResult, kelly_adjusted_by_simulation
from src.ai.reasoning_tracker import (
    ReasoningTracker, ReasoningEntry, DecisionType, DecisionOutcome, get_reasoning_tracker
)


logger = logging.getLogger(__name__)


class TradingRecommendation:
    """Action to take: BUY, SELL, or HOLD."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class AIDecision:
    """
    Complete AI trading decision with all supporting data.
    
    This is the final output from the AI decision engine,
    containing all information needed to execute a trade.
    """
    # Decision
    recommendation: str  # TradingRecommendation
    confidence: float
    
    # Trade parameters
    market_id: str
    market_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    # AI reasoning
    reasoning: str
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Component signals
    ensemble_signal: Optional[EnsembleSignal] = None
    
    # Risk assessment
    monte_carlo_result: Optional[SimulationResult] = None
    kelly_fraction: float = 0.0
    risk_assessment: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reasoning_entry_id: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    
    @property
    def is_actionable(self) -> bool:
        """Check if this decision should result in a trade."""
        return (
            self.recommendation != TradingRecommendation.HOLD and
            self.confidence >= 0.6 and
            self.position_size > 0
        )
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value of the trade."""
        if not self.monte_carlo_result:
            return 0.0
        return self.monte_carlo_result.mean_return * self.position_size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "reasoning": self.reasoning,
            "kelly_fraction": self.kelly_fraction,
            "risk_assessment": self.risk_assessment,
            "is_actionable": self.is_actionable,
            "expected_value": self.expected_value,
            "timestamp": self.timestamp.isoformat(),
            "reasoning_entry_id": self.reasoning_entry_id,
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
        }


class AIDecisionEngine:
    """
    Main AI-powered decision engine for trading.
    
    Integrates all AI components:
    - LLM analysis via Amazon Bedrock
    - Ensemble signal generation
    - Monte Carlo risk simulation
    - Kelly Criterion position sizing
    - Decision logging and tracking
    
    Usage:
        engine = AIDecisionEngine()
        decision = await engine.analyze_market(market, trading_context)
        if decision.is_actionable:
            execute_trade(decision)
    
    The engine respects the autonomy_level setting:
    - 0.0: AI provides advice but doesn't override rules
    - 0.5: Balanced AI/rules decision making
    - 1.0: Fully autonomous AI decisions
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        bedrock_client: Optional[BedrockClient] = None,
    ):
        """
        Initialize the AI decision engine.
        
        Args:
            settings: Optional settings object
            bedrock_client: Optional pre-initialized Bedrock client
        """
        self.settings = settings or get_settings()
        
        # Initialize components
        self.bedrock = bedrock_client or BedrockClient(settings=self.settings)
        self.signal_generator = EnsembleSignalGenerator(
            bedrock_client=self.bedrock,
            settings=self.settings
        )
        self.monte_carlo = MonteCarloSimulator(settings=self.settings)
        self.reasoning_tracker = get_reasoning_tracker()
        
        # Cache for recent analyses
        self._analysis_cache: Dict[str, Tuple[datetime, AIDecision]] = {}
        self._cache_ttl_seconds = 30
        
        # Performance tracking
        self._decisions_made = 0
        self._total_latency_ms = 0.0
    
    async def analyze_market(
        self,
        market: MarketContext,
        trading: TradingContext,
        technical_result: Optional[Dict] = None,
        force_refresh: bool = False,
    ) -> AIDecision:
        """
        Analyze a market and generate trading recommendation.
        
        This is the main entry point for AI-powered analysis.
        
        Args:
            market: Market context with price data
            trading: Trading context with portfolio state
            technical_result: Optional pre-computed technical indicators
            force_refresh: Skip cache and force new analysis
            
        Returns:
            AIDecision with complete trading recommendation
        """
        import time
        start_time = time.time()
        
        # Check cache
        cache_key = f"{market.market_id}_{int(datetime.utcnow().timestamp() / self._cache_ttl_seconds)}"
        if not force_refresh and cache_key in self._analysis_cache:
            cached_time, cached_decision = self._analysis_cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self._cache_ttl_seconds:
                logger.debug(f"Using cached AI decision for {market.market_id}")
                return cached_decision
        
        # Create reasoning entry
        reasoning_entry = self.reasoning_tracker.create_entry(
            decision_type=DecisionType.SIGNAL_GENERATION,
            market_id=market.market_id,
            market_name=market.question[:50] if market.question else market.market_id,
        )
        
        try:
            # Generate ensemble signal
            ensemble_signal = await self.signal_generator.generate(
                market=market,
                trading=trading,
                technical_result=technical_result,
            )
            
            # Run Monte Carlo simulation if enabled
            mc_result = None
            if self.settings.ai.monte_carlo_enabled and ensemble_signal.is_actionable:
                mc_result = self._run_monte_carlo(ensemble_signal, market)
            
            # Calculate position size using Kelly Criterion
            position_size, kelly_fraction = self._calculate_position_size(
                ensemble_signal, trading, mc_result
            )
            
            # Build final decision
            decision = self._build_decision(
                ensemble_signal=ensemble_signal,
                market=market,
                position_size=position_size,
                kelly_fraction=kelly_fraction,
                mc_result=mc_result,
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            decision.latency_ms = latency_ms
            decision.model_used = self.settings.get_bedrock_model_id()
            
            # Update reasoning entry
            self._update_reasoning_entry(
                entry=reasoning_entry,
                market=market,
                trading=trading,
                ensemble_signal=ensemble_signal,
                decision=decision,
                mc_result=mc_result,
            )
            
            # Save reasoning
            decision.reasoning_entry_id = reasoning_entry.entry_id
            self.reasoning_tracker.save_entry(reasoning_entry)
            
            # Cache the decision
            self._analysis_cache[cache_key] = (datetime.utcnow(), decision)
            
            # Update stats
            self._decisions_made += 1
            self._total_latency_ms += latency_ms
            
            logger.info(
                f"AI Decision: {decision.recommendation} {market.question[:30]}... "
                f"(conf: {decision.confidence:.0%}, size: ${decision.position_size:.2f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            
            # Return hold decision on error
            return AIDecision(
                recommendation=TradingRecommendation.HOLD,
                confidence=0.0,
                market_id=market.market_id,
                market_name=market.question[:50] if market.question else "",
                entry_price=market.current_price,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                reasoning=f"AI analysis failed: {str(e)}",
                risk_assessment="ERROR",
            )
    
    def _run_monte_carlo(
        self,
        signal: EnsembleSignal,
        market: MarketContext,
    ) -> SimulationResult:
        """Run Monte Carlo simulation for risk assessment."""
        direction = "BUY" if signal.direction == SignalDirection.BUY else "SELL"
        
        result = self.monte_carlo.simulate_trade(
            entry_price=signal.entry_price,
            position_size=signal.recommended_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volatility=market.volatility if market.volatility > 0 else 0.02,
            direction=direction,
            time_steps=100,
            mean_reversion_strength=0.1,  # Slight mean reversion for prediction markets
            fair_value=market.ewma_price,
        )
        
        return result
    
    def _calculate_position_size(
        self,
        signal: EnsembleSignal,
        trading: TradingContext,
        mc_result: Optional[SimulationResult],
    ) -> Tuple[float, float]:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly formula: f* = (p * b - q) / b
        Where:
        - p = probability of win
        - q = probability of loss (1-p)
        - b = ratio of win to loss
        
        We use fractional Kelly (typically 1/4 to 1/2) for safety.
        """
        method = self.settings.position_sizing.method
        
        if method == "fixed":
            # Fixed position sizing from config
            return self.settings.money.bet_size, 0.0
        
        # Calculate Kelly fraction
        if mc_result and mc_result.prob_profit > 0:
            p = mc_result.prob_profit
            q = mc_result.prob_loss
            
            # Estimate win/loss ratio from simulation
            avg_win = mc_result.percentile_75 if mc_result.percentile_75 > 0 else 0.01
            avg_loss = abs(mc_result.percentile_25) if mc_result.percentile_25 < 0 else 0.01
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            # Kelly formula
            kelly = (p * b - q) / b if b > 0 else 0
        else:
            # Fallback: use confidence as probability estimate
            p = signal.confidence
            q = 1 - p
            b = 1.5  # Assume 1.5:1 risk/reward
            kelly = (p * b - q) / b if b > 0 else 0
        
        # Apply fractional Kelly
        kelly_fraction = self.settings.position_sizing.kelly_fraction
        adjusted_kelly = max(0, kelly * kelly_fraction)
        
        # Further adjust based on Monte Carlo results
        if mc_result:
            adjusted_kelly = kelly_adjusted_by_simulation(
                kelly_fraction=adjusted_kelly,
                simulation_result=mc_result,
            )
        
        # Calculate position size
        max_position = min(
            trading.max_position_size,
            trading.available_capital * self.settings.position_sizing.max_position_fraction,
        )
        
        position_size = adjusted_kelly * trading.available_capital
        
        # Apply bounds
        position_size = max(
            self.settings.position_sizing.min_trade_size,
            min(position_size, max_position)
        )
        
        # Final check: ensure we have capital
        if position_size > trading.available_capital:
            position_size = 0
        
        logger.debug(
            f"Position sizing: Kelly={kelly:.3f}, adjusted={adjusted_kelly:.3f}, "
            f"size=${position_size:.2f}"
        )
        
        return position_size, adjusted_kelly
    
    def _build_decision(
        self,
        ensemble_signal: EnsembleSignal,
        market: MarketContext,
        position_size: float,
        kelly_fraction: float,
        mc_result: Optional[SimulationResult],
    ) -> AIDecision:
        """Build the final AI decision object."""
        # Map signal direction to recommendation
        if ensemble_signal.direction == SignalDirection.BUY:
            recommendation = TradingRecommendation.BUY
        elif ensemble_signal.direction == SignalDirection.SELL:
            recommendation = TradingRecommendation.SELL
        else:
            recommendation = TradingRecommendation.HOLD
        
        # Build reasoning string
        reasoning_parts = []
        
        # Add AI reasoning
        if ensemble_signal.ai_reasoning:
            reasoning_parts.append(f"AI: {ensemble_signal.ai_reasoning}")
        
        # Add component signals summary
        for comp in ensemble_signal.components:
            reasoning_parts.append(
                f"{comp.source.value}: {comp.direction.value} "
                f"({comp.confidence:.0%}) - {comp.reasoning[:50]}"
            )
        
        # Add Monte Carlo insight
        if mc_result:
            reasoning_parts.append(
                f"MC Risk: {mc_result.get_risk_assessment()}, "
                f"P(profit)={mc_result.prob_profit:.0%}"
            )
        
        reasoning = " | ".join(reasoning_parts)
        
        # Determine risk assessment
        if mc_result:
            risk_assessment = mc_result.get_risk_assessment()
        elif ensemble_signal.confidence < 0.5:
            risk_assessment = "LOW_CONFIDENCE"
        elif ensemble_signal.confidence < 0.7:
            risk_assessment = "MODERATE"
        else:
            risk_assessment = "FAVORABLE"
        
        return AIDecision(
            recommendation=recommendation,
            confidence=ensemble_signal.confidence,
            market_id=market.market_id,
            market_name=market.question[:50] if market.question else "",
            entry_price=ensemble_signal.entry_price or market.current_price,
            stop_loss=ensemble_signal.stop_loss,
            take_profit=ensemble_signal.take_profit,
            position_size=position_size,
            reasoning=reasoning,
            ai_analysis=ensemble_signal.ai_analysis,
            ensemble_signal=ensemble_signal,
            monte_carlo_result=mc_result,
            kelly_fraction=kelly_fraction,
            risk_assessment=risk_assessment,
        )
    
    def _update_reasoning_entry(
        self,
        entry: ReasoningEntry,
        market: MarketContext,
        trading: TradingContext,
        ensemble_signal: EnsembleSignal,
        decision: AIDecision,
        mc_result: Optional[SimulationResult],
    ):
        """Update reasoning entry with analysis results."""
        entry.market_context = {
            "current_price": market.current_price,
            "ewma_price": market.ewma_price,
            "roc": market.roc,
            "cusum_positive": market.cusum_positive,
            "cusum_negative": market.cusum_negative,
            "volume_24h": market.volume_24h,
            "band_position": market.band_position(),
        }
        
        entry.technical_indicators = {
            "ewma_upper": market.ewma_upper_band,
            "ewma_lower": market.ewma_lower_band,
            "volatility": market.volatility,
        }
        
        entry.portfolio_context = {
            "current_capital": trading.current_capital,
            "available_capital": trading.available_capital,
            "daily_pnl": trading.daily_pnl,
            "open_positions": trading.open_positions_count,
        }
        
        entry.ai_reasoning = ensemble_signal.ai_reasoning
        entry.ai_confidence = ensemble_signal.confidence
        entry.ai_response = ensemble_signal.ai_analysis
        
        entry.component_signals = [
            {
                "source": c.source.value,
                "direction": c.direction.value,
                "confidence": c.confidence,
                "weight": c.weight,
                "reasoning": c.reasoning,
            }
            for c in ensemble_signal.components
        ]
        
        entry.final_action = decision.recommendation
        entry.final_confidence = decision.confidence
        entry.position_size = decision.position_size
        entry.entry_price = decision.entry_price
        entry.stop_loss = decision.stop_loss
        entry.take_profit = decision.take_profit
        
        if mc_result:
            entry.monte_carlo_results = mc_result.to_dict()
    
    async def analyze_exit(
        self,
        market: MarketContext,
        position: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze whether to exit an open position.
        
        Args:
            market: Current market context
            position: Open position details
            
        Returns:
            Exit recommendation with reasoning
        """
        prompt = PromptTemplates.exit_timing_prompt(market, position)
        
        response = await self.bedrock.generate_async(
            prompt=prompt,
            system_prompt=PromptTemplates.system_prompt(),
            max_tokens=512,
        )
        
        result = PromptTemplates.parse_json_response(response.content)
        
        if "error" in result:
            return {
                "recommendation": "HOLD",
                "reasoning": "Exit analysis failed",
                "error": result.get("error"),
            }
        
        return result
    
    async def batch_analyze(
        self,
        markets: List[MarketContext],
        trading: TradingContext,
        max_opportunities: int = 3,
    ) -> List[AIDecision]:
        """
        Analyze multiple markets and return top opportunities.
        
        Efficient batch analysis for market scanning.
        """
        # First pass: quick batch analysis via AI
        prompt = PromptTemplates.batch_analysis_prompt(markets)
        
        response = await self.bedrock.generate_async(
            prompt=prompt,
            system_prompt=PromptTemplates.system_prompt(),
            max_tokens=1024,
        )
        
        batch_result = PromptTemplates.parse_json_response(response.content)
        
        opportunities = batch_result.get("opportunities", [])
        
        # Full analysis on top opportunities
        decisions = []
        for opp in opportunities[:max_opportunities]:
            market_idx = opp.get("market_index", 1) - 1
            if 0 <= market_idx < len(markets):
                decision = await self.analyze_market(markets[market_idx], trading)
                if decision.is_actionable:
                    decisions.append(decision)
        
        # Sort by expected value
        decisions.sort(key=lambda d: d.confidence, reverse=True)
        
        return decisions
    
    def record_trade_outcome(
        self,
        reasoning_entry_id: str,
        pnl: float,
        exit_price: float,
        exit_reason: str,
    ):
        """
        Record the outcome of a trade for learning.
        
        Called when a position is closed.
        """
        if pnl > 0:
            outcome = DecisionOutcome.PROFITABLE
        elif pnl < 0:
            outcome = DecisionOutcome.UNPROFITABLE
        else:
            outcome = DecisionOutcome.BREAKEVEN
        
        self.reasoning_tracker.update_outcome(
            entry_id=reasoning_entry_id,
            outcome=outcome,
            actual_pnl=pnl,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "decisions_made": self._decisions_made,
            "avg_latency_ms": (
                self._total_latency_ms / self._decisions_made 
                if self._decisions_made > 0 else 0
            ),
            "bedrock_usage": self.bedrock.get_usage_stats(),
            "reasoning_stats": self.reasoning_tracker.get_stats(),
            "signal_weights": self.signal_generator.get_weights(),
        }
    
    def get_recent_reasoning(self, limit: int = 10) -> List[Dict]:
        """Get recent AI reasoning for dashboard."""
        return self.reasoning_tracker.get_for_dashboard()[:limit]


# Convenience function for creating engine
def create_ai_engine(settings: Optional[Settings] = None) -> AIDecisionEngine:
    """Create a new AI decision engine instance."""
    return AIDecisionEngine(settings=settings)


if __name__ == "__main__":
    # Test the decision engine
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        print("=" * 60)
        print("AI DECISION ENGINE TEST")
        print("=" * 60)
        
        # Create test market context
        market = MarketContext(
            market_id="test_btc_100k",
            question="Will Bitcoin reach $100,000 by December 31, 2024?",
            current_price=0.65,
            price_24h_ago=0.58,
            price_1h_ago=0.63,
            ewma_price=0.61,
            ewma_upper_band=0.70,
            ewma_lower_band=0.52,
            roc=3.5,
            cusum_positive=0.045,
            cusum_negative=-0.01,
            volatility=0.02,
            volume_24h=125000,
            liquidity=5000,
        )
        
        trading = TradingContext(
            current_capital=75.0,
            available_capital=70.0,
            daily_pnl=0.50,
            max_position_size=2.0,
            max_daily_loss=2.0,
            remaining_daily_risk=1.50,
            open_positions_count=1,
            max_positions=2,
        )
        
        settings = get_settings()
        print(f"AI Enabled: {settings.is_ai_enabled()}")
        print(f"AWS Credentials: {settings.has_aws_credentials()}")
        
        if not settings.has_aws_credentials():
            print("\n⚠️ AWS credentials not configured - skipping live test")
            print("Configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to test")
            return
        
        engine = AIDecisionEngine()
        
        print("\nAnalyzing market...")
        decision = await engine.analyze_market(market, trading)
        
        print(f"\n{'='*60}")
        print("DECISION:")
        print(f"  Recommendation: {decision.recommendation}")
        print(f"  Confidence: {decision.confidence:.0%}")
        print(f"  Position Size: ${decision.position_size:.2f}")
        print(f"  Entry: ${decision.entry_price:.3f}")
        print(f"  Stop Loss: ${decision.stop_loss:.3f}")
        print(f"  Take Profit: ${decision.take_profit:.3f}")
        print(f"  Risk Assessment: {decision.risk_assessment}")
        print(f"\nReasoning:")
        print(f"  {decision.reasoning}")
        print(f"\nLatency: {decision.latency_ms:.0f}ms")
        print(f"Actionable: {decision.is_actionable}")
    
    asyncio.run(test())