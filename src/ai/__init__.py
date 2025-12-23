"""
AI Decision Engine Module
=========================

This module contains the AI-powered decision engine using Amazon Bedrock
for autonomous trading decisions.

Components:
- BedrockClient: AWS Bedrock API wrapper for multiple foundation models
- PromptTemplates: Structured prompts for market analysis and trading
- AIDecisionEngine: Main decision coordinator
- SignalGenerator: Ensemble signal generation combining AI with technicals
- MonteCarloSimulator: Risk assessment through simulation
- ReasoningTracker: Logs and stores AI decision rationale

Usage:
    from src.ai import AIDecisionEngine, MarketContext, TradingContext
    
    engine = AIDecisionEngine()
    decision = await engine.analyze_market(market_context, trading_context)
    
    if decision.is_actionable:
        execute_trade(decision)
"""

from src.ai.bedrock_client import BedrockClient, ModelResponse, ModelFamily
from src.ai.prompts import PromptTemplates, MarketContext, TradingContext
from src.ai.decision_engine import (
    AIDecisionEngine,
    AIDecision,
    TradingRecommendation,
    create_ai_engine,
)
from src.ai.signal_generator import (
    EnsembleSignalGenerator,
    EnsembleSignal,
    SignalDirection,
    SignalSource,
    ComponentSignal,
)
from src.ai.monte_carlo import (
    MonteCarloSimulator,
    SimulationResult,
    kelly_adjusted_by_simulation,
)
from src.ai.reasoning_tracker import (
    ReasoningTracker,
    ReasoningEntry,
    DecisionType,
    DecisionOutcome,
    get_reasoning_tracker,
)

__all__ = [
    # Bedrock Client
    "BedrockClient",
    "ModelResponse",
    "ModelFamily",
    
    # Prompts
    "PromptTemplates",
    "MarketContext",
    "TradingContext",
    
    # Decision Engine
    "AIDecisionEngine",
    "AIDecision",
    "TradingRecommendation",
    "create_ai_engine",
    
    # Signal Generator
    "EnsembleSignalGenerator",
    "EnsembleSignal",
    "SignalDirection",
    "SignalSource",
    "ComponentSignal",
    
    # Monte Carlo
    "MonteCarloSimulator",
    "SimulationResult",
    "kelly_adjusted_by_simulation",
    
    # Reasoning Tracker
    "ReasoningTracker",
    "ReasoningEntry",
    "DecisionType",
    "DecisionOutcome",
    "get_reasoning_tracker",
]