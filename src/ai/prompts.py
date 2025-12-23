"""
AI Prompt Templates for Trading Decisions
==========================================

Structured prompts for:
1. Market Analysis - Understanding prediction market context
2. Sentiment Analysis - Extracting sentiment from market descriptions
3. Trading Decisions - Generate buy/sell/hold recommendations
4. Risk Assessment - Evaluate trade risk factors
5. Position Management - Exit timing recommendations

Each prompt is designed to elicit structured, parseable responses
from the AI model for automated decision-making.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class PromptType(str, Enum):
    """Types of prompts for different decision stages."""
    MARKET_ANALYSIS = "market_analysis"
    SENTIMENT = "sentiment"
    TRADING_DECISION = "trading_decision"
    RISK_ASSESSMENT = "risk_assessment"
    EXIT_TIMING = "exit_timing"


@dataclass
class MarketContext:
    """
    Context data for a prediction market.
    
    Provides all relevant information for AI analysis.
    """
    # Market identification
    market_id: str
    question: str  # The prediction market question
    description: str = ""  # Additional market description
    
    # Price data
    current_price: float = 0.5
    price_24h_ago: float = 0.5
    price_1h_ago: float = 0.5
    
    # Technical indicators
    ewma_price: float = 0.5
    ewma_upper_band: float = 0.55
    ewma_lower_band: float = 0.45
    roc: float = 0.0  # Rate of change
    cusum_positive: float = 0.0
    cusum_negative: float = 0.0
    volatility: float = 0.0
    
    # Market metrics
    volume_24h: float = 0.0
    liquidity: float = 0.0
    spread: float = 0.0
    
    # Resolution info
    end_date: Optional[str] = None
    resolution_source: str = ""
    
    # Recent price history (last 10 prices)
    recent_prices: List[float] = field(default_factory=list)
    
    def price_change_24h(self) -> float:
        """Calculate 24h price change percentage."""
        if self.price_24h_ago == 0:
            return 0.0
        return ((self.current_price - self.price_24h_ago) / self.price_24h_ago) * 100
    
    def price_change_1h(self) -> float:
        """Calculate 1h price change percentage."""
        if self.price_1h_ago == 0:
            return 0.0
        return ((self.current_price - self.price_1h_ago) / self.price_1h_ago) * 100
    
    def is_above_ewma(self) -> bool:
        """Check if price is above EWMA."""
        return self.current_price > self.ewma_price
    
    def band_position(self) -> str:
        """Describe price position relative to bands."""
        if self.current_price > self.ewma_upper_band:
            return "above_upper"
        elif self.current_price < self.ewma_lower_band:
            return "below_lower"
        else:
            return "within_bands"
    
    def to_summary(self) -> str:
        """Generate a text summary for AI prompt."""
        return f"""
Market: {self.question}
Current Price: ${self.current_price:.3f} (probability: {self.current_price*100:.1f}%)
24h Change: {self.price_change_24h():+.2f}%
1h Change: {self.price_change_1h():+.2f}%

Technical Indicators:
- EWMA Price: ${self.ewma_price:.3f}
- Upper Band: ${self.ewma_upper_band:.3f}
- Lower Band: ${self.ewma_lower_band:.3f}
- Position: {self.band_position().replace('_', ' ')}
- ROC (momentum): {self.roc:+.2f}%
- CUSUM+: {self.cusum_positive:.4f}
- CUSUM-: {self.cusum_negative:.4f}
- Volatility: {self.volatility:.4f}

Market Metrics:
- 24h Volume: ${self.volume_24h:,.0f}
- Liquidity: ${self.liquidity:,.0f}
- Spread: {self.spread:.2f}%

Resolution: {self.end_date or 'Unknown'}
""".strip()


@dataclass
class TradingContext:
    """
    Additional context for trading decisions.
    
    Includes portfolio state and risk parameters.
    """
    # Current capital state
    current_capital: float = 75.0
    available_capital: float = 75.0
    daily_pnl: float = 0.0
    
    # Risk parameters
    max_position_size: float = 2.0
    max_daily_loss: float = 2.0
    remaining_daily_risk: float = 2.0
    
    # Current positions
    open_positions_count: int = 0
    max_positions: int = 2
    
    # Performance context
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return (
            self.open_positions_count < self.max_positions and
            self.remaining_daily_risk > 0.5
        )
    
    def to_summary(self) -> str:
        """Generate summary for AI prompt."""
        return f"""
Portfolio State:
- Current Capital: ${self.current_capital:.2f}
- Available: ${self.available_capital:.2f}
- Daily P&L: ${self.daily_pnl:+.2f}

Risk Limits:
- Max Position Size: ${self.max_position_size:.2f}
- Max Daily Loss: ${self.max_daily_loss:.2f}
- Remaining Risk Budget: ${self.remaining_daily_risk:.2f}

Positions:
- Open: {self.open_positions_count} / {self.max_positions}
- Can Open New: {'Yes' if self.can_open_position() else 'No'}

Performance:
- Win Rate: {self.win_rate:.1f}%
- Avg Win: ${self.avg_win:.2f}
- Avg Loss: ${self.avg_loss:.2f}
""".strip()


class PromptTemplates:
    """
    Prompt templates for AI trading decisions.
    
    All prompts are designed to produce structured, parseable JSON responses.
    """
    
    @staticmethod
    def system_prompt() -> str:
        """Base system prompt for trading AI."""
        return """You are an expert quantitative trading analyst specializing in prediction markets. Your role is to analyze market data and provide precise, actionable trading recommendations.

CRITICAL RULES:
1. Always respond with valid JSON only - no markdown, no explanations outside JSON
2. Be conservative with confidence scores - only high confidence (>0.7) when evidence is strong
3. Consider both technical indicators AND fundamental market factors
4. Account for mean reversion - extreme prices often revert
5. Never recommend trades that violate risk limits
6. Provide clear, specific reasoning for every recommendation

You understand prediction markets where:
- Price represents implied probability (0.50 = 50% chance)
- Prices near 0 or 1 have high resolution risk
- Volume and liquidity indicate market confidence
- CUSUM detects regime changes (trend shifts)
- EWMA bands show statistical extremes"""
    
    @staticmethod
    def market_analysis_prompt(market: MarketContext) -> str:
        """
        Prompt for comprehensive market analysis.
        
        Returns structured analysis of market dynamics.
        """
        return f"""Analyze this prediction market and provide a comprehensive assessment.

{market.to_summary()}

Respond with JSON in this exact format:
{{
    "market_type": "political|sports|crypto|entertainment|science|other",
    "sentiment": "bullish|bearish|neutral",
    "sentiment_score": 0.0 to 1.0 (0=very bearish, 1=very bullish),
    "trend": "uptrend|downtrend|sideways|reversal_up|reversal_down",
    "momentum": "strong_positive|positive|neutral|negative|strong_negative",
    "volatility_assessment": "low|normal|high|extreme",
    "price_fair_value": estimated fair price (0.0-1.0),
    "mispricing": "overvalued|undervalued|fair",
    "mispricing_magnitude": 0.0 to 1.0 (how mispriced),
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks": ["risk1", "risk2"],
    "confidence": 0.0 to 1.0 (confidence in this analysis),
    "reasoning": "2-3 sentence explanation"
}}"""
    
    @staticmethod
    def trading_decision_prompt(
        market: MarketContext,
        trading: TradingContext,
        technical_signal: Optional[str] = None,
    ) -> str:
        """
        Prompt for trading decision.
        
        Combines market analysis with portfolio context.
        """
        signal_info = ""
        if technical_signal:
            signal_info = f"\nTechnical Signal Detected: {technical_signal}"
        
        return f"""Based on the following market and portfolio data, provide a trading recommendation.

=== MARKET DATA ===
{market.to_summary()}
{signal_info}

=== PORTFOLIO STATE ===
{trading.to_summary()}

Respond with JSON in this exact format:
{{
    "action": "BUY|SELL|HOLD",
    "confidence": 0.0 to 1.0,
    "recommended_size": dollar amount (0 if HOLD),
    "entry_price": target entry price,
    "stop_loss": recommended stop loss price,
    "take_profit": recommended take profit price,
    "time_horizon": "scalp|intraday|swing|position",
    "edge_estimate": estimated edge in percentage points,
    "risk_reward_ratio": expected R:R ratio,
    "key_reasons": ["reason1", "reason2", "reason3"],
    "risks": ["risk1", "risk2"],
    "reasoning": "2-3 sentence explanation of the decision"
}}

IMPORTANT:
- Only recommend BUY/SELL if confidence > 0.6 and clear edge exists
- Respect position limits: max ${trading.max_position_size:.2f}
- Respect risk limits: remaining risk budget ${trading.remaining_daily_risk:.2f}
- Consider current positions: {trading.open_positions_count}/{trading.max_positions}"""
    
    @staticmethod
    def sentiment_analysis_prompt(question: str, description: str = "") -> str:
        """
        Prompt for pure sentiment analysis of market question.
        
        Useful for understanding market direction.
        """
        return f"""Analyze the sentiment and probability implications of this prediction market question.

Question: {question}

{f'Description: {description}' if description else ''}

Respond with JSON in this exact format:
{{
    "category": "political|sports|crypto|entertainment|science|economic|legal|other",
    "sentiment_toward_yes": 0.0 to 1.0 (public sentiment favoring YES outcome),
    "uncertainty_level": "low|medium|high|extreme",
    "time_sensitivity": "immediate|short_term|medium_term|long_term",
    "key_factors": ["factor1", "factor2"],
    "potential_catalysts": ["catalyst1", "catalyst2"],
    "base_rate_estimate": 0.0 to 1.0 (historical base rate for similar events),
    "reasoning": "1-2 sentence explanation"
}}"""
    
    @staticmethod
    def risk_assessment_prompt(
        market: MarketContext,
        proposed_trade: Dict[str, Any],
        trading: TradingContext,
    ) -> str:
        """
        Prompt for risk assessment of a proposed trade.
        
        Evaluates whether the trade is advisable given risk parameters.
        """
        return f"""Assess the risk of this proposed trade.

=== MARKET ===
{market.to_summary()}

=== PROPOSED TRADE ===
Action: {proposed_trade.get('action', 'UNKNOWN')}
Size: ${proposed_trade.get('size', 0):.2f}
Entry Price: ${proposed_trade.get('entry_price', 0):.3f}
Stop Loss: ${proposed_trade.get('stop_loss', 0):.3f}
Take Profit: ${proposed_trade.get('take_profit', 0):.3f}

=== PORTFOLIO CONTEXT ===
{trading.to_summary()}

Respond with JSON in this exact format:
{{
    "risk_score": 0.0 to 1.0 (0=very safe, 1=very risky),
    "risk_level": "low|moderate|high|extreme",
    "max_loss_estimate": dollar amount,
    "probability_of_loss": 0.0 to 1.0,
    "probability_of_stop_hit": 0.0 to 1.0,
    "position_size_appropriate": true|false,
    "recommended_adjustments": {{
        "size": adjusted size or null,
        "stop_loss": adjusted stop or null,
        "take_profit": adjusted TP or null
    }},
    "approval": "APPROVED|ADJUST|REJECT",
    "concerns": ["concern1", "concern2"],
    "reasoning": "1-2 sentence explanation"
}}"""
    
    @staticmethod
    def exit_timing_prompt(
        market: MarketContext,
        position: Dict[str, Any],
    ) -> str:
        """
        Prompt for exit timing recommendation on open position.
        
        Helps decide when to close profitable or losing positions.
        """
        pnl = position.get('unrealized_pnl', 0)
        pnl_pct = position.get('pnl_percent', 0)
        
        return f"""Evaluate whether to exit this open position.

=== MARKET STATE ===
{market.to_summary()}

=== POSITION ===
Side: {position.get('side', 'UNKNOWN')}
Entry Price: ${position.get('entry_price', 0):.3f}
Current Price: ${market.current_price:.3f}
Size: ${position.get('size', 0):.2f}
Unrealized P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)
Stop Loss: ${position.get('stop_loss', 0):.3f}
Take Profit: ${position.get('take_profit', 0):.3f}
Time Held: {position.get('time_held', 'unknown')}

Respond with JSON in this exact format:
{{
    "recommendation": "HOLD|CLOSE|PARTIAL_CLOSE|ADJUST_STOPS",
    "urgency": "immediate|soon|patient",
    "close_percentage": 0 to 100 (if PARTIAL_CLOSE),
    "new_stop_loss": price or null,
    "new_take_profit": price or null,
    "expected_outcome": "profit|loss|breakeven",
    "confidence": 0.0 to 1.0,
    "reasoning": "2-3 sentence explanation"
}}"""
    
    @staticmethod
    def batch_analysis_prompt(markets: List[MarketContext]) -> str:
        """
        Prompt for analyzing multiple markets at once.
        
        Efficient for scanning many markets for opportunities.
        """
        market_summaries = []
        for i, m in enumerate(markets[:10]):  # Limit to 10 markets
            market_summaries.append(f"""
Market {i+1}: {m.question[:100]}
Price: ${m.current_price:.3f} | 24h: {m.price_change_24h():+.1f}% | Vol: ${m.volume_24h:,.0f}
EWMA: ${m.ewma_price:.3f} | Band: {m.band_position()} | ROC: {m.roc:+.2f}%
""")
        
        markets_text = "\n".join(market_summaries)
        
        return f"""Analyze these prediction markets and identify trading opportunities.

{markets_text}

Respond with JSON in this exact format:
{{
    "opportunities": [
        {{
            "market_index": 1-10,
            "action": "BUY|SELL",
            "confidence": 0.0 to 1.0,
            "edge_estimate": percentage,
            "brief_reason": "one sentence"
        }}
    ],
    "best_opportunity": market index (1-10) or null if none,
    "market_conditions": "favorable|neutral|unfavorable",
    "reasoning": "brief overall assessment"
}}

Only include markets with confidence > 0.6 and clear edge in opportunities array."""
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """
        Parse JSON from AI response, handling common issues.
        
        Args:
            response: Raw AI response text
            
        Returns:
            Parsed JSON dictionary
        """
        # Clean up common issues
        text = response.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to find JSON object in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return error dict
            return {
                "error": "Failed to parse JSON",
                "raw_response": response[:500],
                "parse_error": str(e),
            }


# Example usage and testing
if __name__ == "__main__":
    # Create sample market context
    market = MarketContext(
        market_id="test123",
        question="Will Bitcoin reach $100,000 by end of 2024?",
        description="Market resolves YES if BTC/USD reaches $100,000 on any major exchange",
        current_price=0.65,
        price_24h_ago=0.58,
        price_1h_ago=0.63,
        ewma_price=0.61,
        ewma_upper_band=0.68,
        ewma_lower_band=0.54,
        roc=3.2,
        cusum_positive=0.04,
        cusum_negative=-0.01,
        volatility=0.02,
        volume_24h=125000,
        liquidity=5000,
        spread=1.5,
        end_date="2024-12-31",
    )
    
    trading = TradingContext(
        current_capital=75.0,
        available_capital=73.0,
        daily_pnl=0.50,
        max_position_size=2.0,
        max_daily_loss=2.0,
        remaining_daily_risk=1.50,
        open_positions_count=1,
        max_positions=2,
        win_rate=60.0,
        avg_win=0.45,
        avg_loss=0.30,
    )
    
    print("=" * 60)
    print("PROMPT TEMPLATES TEST")
    print("=" * 60)
    
    print("\n### Market Analysis Prompt ###")
    print(PromptTemplates.market_analysis_prompt(market))
    
    print("\n### Trading Decision Prompt ###")
    print(PromptTemplates.trading_decision_prompt(market, trading, "CUSUM breakout detected"))
    
    print("\n### Sentiment Analysis Prompt ###")
    print(PromptTemplates.sentiment_analysis_prompt(market.question, market.description))