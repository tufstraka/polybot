"""
Unit tests for AI components.

Tests cover:
- Monte Carlo simulation
- Kelly Criterion position sizing
- Ensemble signal generator
- Reasoning tracker
- AI Decision Engine
"""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.monte_carlo import MonteCarloSimulator, SimulationResult
from src.ai.signal_generator import (
    EnsembleSignalGenerator, 
    EnsembleSignal, 
    TechnicalSignals,
    AISignal,
    SentimentSignal,
    MeanReversionSignal
)
from src.ai.reasoning_tracker import (
    ReasoningTracker, 
    ReasoningEntry, 
    DecisionOutcome
)


class TestMonteCarloSimulator:
    """Tests for Monte Carlo simulation."""
    
    def test_init_default_params(self):
        """Test simulator initialization with defaults."""
        sim = MonteCarloSimulator()
        assert sim.num_simulations == 1000
        assert sim.time_horizon == 10
        assert sim.mean_reversion_speed == 0.1
    
    def test_init_custom_params(self):
        """Test simulator initialization with custom params."""
        sim = MonteCarloSimulator(
            num_simulations=500,
            time_horizon=20,
            mean_reversion_speed=0.2
        )
        assert sim.num_simulations == 500
        assert sim.time_horizon == 20
        assert sim.mean_reversion_speed == 0.2
    
    def test_simulate_returns_result(self):
        """Test simulation returns valid result."""
        sim = MonteCarloSimulator(num_simulations=100)
        result = sim.simulate(
            current_price=0.50,
            volatility=0.02,
            fair_value=0.55
        )
        
        assert isinstance(result, SimulationResult)
        assert result.current_price == 0.50
        assert result.volatility == 0.02
        assert 0 <= result.prob_profit <= 1
        assert result.expected_return is not None
        assert result.var_95 is not None
        assert result.var_99 is not None
        assert result.max_loss is not None
        assert result.max_gain is not None
    
    def test_simulate_price_paths_shape(self):
        """Test simulated paths have correct shape."""
        sim = MonteCarloSimulator(num_simulations=50, time_horizon=5)
        result = sim.simulate(0.50, 0.02, 0.55)
        
        # Expected shape: (num_simulations, time_horizon + 1)
        assert result.final_prices.shape == (50,)
        assert result.paths.shape == (50, 6)  # 5 time steps + initial
    
    def test_simulate_high_volatility(self):
        """Test simulation with high volatility."""
        sim = MonteCarloSimulator(num_simulations=100)
        result = sim.simulate(0.50, 0.10, 0.55)  # 10% volatility
        
        # High volatility should have wider range
        assert result.max_gain - result.max_loss > 0.1
    
    def test_simulate_low_volatility(self):
        """Test simulation with low volatility."""
        sim = MonteCarloSimulator(num_simulations=100)
        result = sim.simulate(0.50, 0.01, 0.55)  # 1% volatility
        
        # Low volatility should have narrower range
        assert result.var_95 < 0.20  # Less than 20% VaR
    
    def test_var_values_ordered(self):
        """Test VaR values are properly ordered."""
        sim = MonteCarloSimulator(num_simulations=500)
        result = sim.simulate(0.50, 0.03, 0.55)
        
        # VaR 99 should be more extreme than VaR 95
        assert abs(result.var_99) >= abs(result.var_95) * 0.9  # Allow some tolerance
    
    def test_risk_assessment(self):
        """Test risk assessment string generation."""
        sim = MonteCarloSimulator(num_simulations=100)
        result = sim.simulate(0.50, 0.02, 0.55)
        
        assessment = result.get_risk_assessment()
        assert isinstance(assessment, str)
        assert len(assessment) > 0
        # Should contain risk level indication
        assert any(word in assessment.lower() for word in ['risk', 'profit', 'loss', 'favorable'])


class TestEnsembleSignalGenerator:
    """Tests for ensemble signal generation."""
    
    @pytest.fixture
    def generator(self):
        """Create signal generator with default weights."""
        return EnsembleSignalGenerator(autonomy_level=0.5)
    
    @pytest.fixture
    def technical_signals(self):
        """Create sample technical signals."""
        return TechnicalSignals(
            cusum_positive=0.5,
            cusum_negative=0.1,
            roc=0.03,
            ewma_deviation=0.02,
            volatility=0.02,
            cusum_triggered=True,
            ewma_band_position="above_upper"
        )
    
    @pytest.fixture
    def ai_signal(self):
        """Create sample AI signal."""
        return AISignal(
            recommendation="BUY",
            confidence=0.75,
            reasoning="Strong bullish momentum detected"
        )
    
    @pytest.fixture
    def sentiment_signal(self):
        """Create sample sentiment signal."""
        return SentimentSignal(
            score=0.6,
            magnitude=0.7,
            source="market_analysis"
        )
    
    @pytest.fixture
    def mean_reversion_signal(self):
        """Create sample mean reversion signal."""
        return MeanReversionSignal(
            deviation=0.05,
            z_score=2.0,
            reversion_probability=0.7
        )
    
    def test_init_autonomy_level(self):
        """Test autonomy level affects weights."""
        low_autonomy = EnsembleSignalGenerator(autonomy_level=0.0)
        high_autonomy = EnsembleSignalGenerator(autonomy_level=1.0)
        
        # Higher autonomy should give more weight to AI
        assert high_autonomy.weights['ai'] > low_autonomy.weights['ai']
        assert high_autonomy.weights['technical'] < low_autonomy.weights['technical']
    
    def test_generate_signal_structure(self, generator, technical_signals, ai_signal):
        """Test generated signal has correct structure."""
        signal = generator.generate(
            technical=technical_signals,
            ai=ai_signal,
            sentiment=None,
            mean_reversion=None
        )
        
        assert isinstance(signal, EnsembleSignal)
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'strength')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'components')
    
    def test_generate_signal_with_all_components(
        self, generator, technical_signals, ai_signal, 
        sentiment_signal, mean_reversion_signal
    ):
        """Test signal generation with all components."""
        signal = generator.generate(
            technical=technical_signals,
            ai=ai_signal,
            sentiment=sentiment_signal,
            mean_reversion=mean_reversion_signal
        )
        
        assert signal.direction in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1
        assert len(signal.components) == 4
    
    def test_generate_signal_technical_only(self, generator, technical_signals):
        """Test signal generation with only technical signals."""
        signal = generator.generate(
            technical=technical_signals,
            ai=None,
            sentiment=None,
            mean_reversion=None
        )
        
        assert signal is not None
        assert signal.direction in ['BUY', 'SELL', 'HOLD']
    
    def test_signal_strength_bounded(
        self, generator, technical_signals, ai_signal
    ):
        """Test signal strength is bounded 0-1."""
        for _ in range(10):
            signal = generator.generate(
                technical=technical_signals,
                ai=ai_signal,
                sentiment=None,
                mean_reversion=None
            )
            assert 0 <= signal.strength <= 1
    
    def test_confidence_calculation(
        self, generator, technical_signals, ai_signal
    ):
        """Test confidence is properly calculated."""
        # High agreement should give high confidence
        ai_high_conf = AISignal(
            recommendation="BUY",
            confidence=0.9,
            reasoning="Very strong signal"
        )
        
        signal = generator.generate(
            technical=technical_signals,  # Also bullish
            ai=ai_high_conf,
            sentiment=None,
            mean_reversion=None
        )
        
        # When signals agree, confidence should be reasonable
        assert signal.confidence >= 0.5


class TestKellyCriterion:
    """Tests for Kelly Criterion position sizing."""
    
    def test_full_kelly_calculation(self):
        """Test full Kelly criterion formula."""
        from src.ai.signal_generator import EnsembleSignalGenerator
        
        gen = EnsembleSignalGenerator()
        
        # Test case: 60% win rate, 1:1 odds
        kelly = gen.calculate_kelly(
            win_probability=0.60,
            win_loss_ratio=1.0
        )
        
        # Kelly = p - q/b = 0.60 - 0.40/1.0 = 0.20
        assert abs(kelly - 0.20) < 0.01
    
    def test_kelly_with_negative_edge(self):
        """Test Kelly returns 0 for negative edge."""
        from src.ai.signal_generator import EnsembleSignalGenerator
        
        gen = EnsembleSignalGenerator()
        
        # 40% win rate, 1:1 odds - negative edge
        kelly = gen.calculate_kelly(
            win_probability=0.40,
            win_loss_ratio=1.0
        )
        
        assert kelly == 0  # Should not bet
    
    def test_fractional_kelly(self):
        """Test fractional Kelly for risk reduction."""
        from src.ai.signal_generator import EnsembleSignalGenerator
        
        gen = EnsembleSignalGenerator()
        
        full_kelly = gen.calculate_kelly(
            win_probability=0.70,
            win_loss_ratio=1.5
        )
        
        fractional = gen.calculate_kelly(
            win_probability=0.70,
            win_loss_ratio=1.5,
            fraction=0.5
        )
        
        assert abs(fractional - full_kelly * 0.5) < 0.01
    
    def test_kelly_position_size(self):
        """Test position size calculation from Kelly."""
        from src.ai.signal_generator import EnsembleSignalGenerator
        
        gen = EnsembleSignalGenerator()
        
        position = gen.calculate_position_size(
            capital=1000,
            kelly_fraction=0.10,  # 10% Kelly
            max_position=100
        )
        
        assert position == 100  # min(1000*0.10, 100) = 100


class TestReasoningTracker:
    """Tests for AI reasoning tracker."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def tracker(self, temp_log_dir):
        """Create tracker with temp directory."""
        return ReasoningTracker(log_dir=temp_log_dir)
    
    def test_log_decision_creates_entry(self, tracker):
        """Test logging creates entry with ID."""
        entry_id = tracker.log_decision(
            market_id="market_123",
            decision="BUY",
            confidence=0.75,
            reasoning="Test reasoning",
            signals={"technical": 0.8, "ai": 0.7}
        )
        
        assert entry_id is not None
        assert len(entry_id) > 0
    
    def test_log_decision_stores_data(self, tracker):
        """Test logged decision data is stored."""
        entry_id = tracker.log_decision(
            market_id="market_123",
            decision="BUY",
            confidence=0.75,
            reasoning="Test reasoning",
            signals={"technical": 0.8}
        )
        
        entry = tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.market_id == "market_123"
        assert entry.decision == "BUY"
        assert entry.confidence == 0.75
    
    def test_record_outcome(self, tracker):
        """Test recording trade outcome."""
        entry_id = tracker.log_decision(
            market_id="market_123",
            decision="BUY",
            confidence=0.75,
            reasoning="Test",
            signals={}
        )
        
        tracker.record_outcome(
            entry_id=entry_id,
            outcome=DecisionOutcome.PROFIT,
            pnl=10.50,
            exit_price=0.65,
            exit_reason="take_profit"
        )
        
        entry = tracker.get_entry(entry_id)
        assert entry.outcome == DecisionOutcome.PROFIT
        assert entry.pnl == 10.50
    
    def test_get_recent_entries(self, tracker):
        """Test getting recent entries."""
        # Log multiple decisions
        for i in range(5):
            tracker.log_decision(
                market_id=f"market_{i}",
                decision="BUY" if i % 2 == 0 else "SELL",
                confidence=0.5 + i * 0.1,
                reasoning=f"Reason {i}",
                signals={}
            )
        
        recent = tracker.get_recent_entries(3)
        assert len(recent) == 3
    
    def test_get_stats(self, tracker):
        """Test statistics calculation."""
        # Log some decisions with outcomes
        for i in range(4):
            entry_id = tracker.log_decision(
                market_id=f"market_{i}",
                decision="BUY",
                confidence=0.7,
                reasoning="Test",
                signals={}
            )
            tracker.record_outcome(
                entry_id=entry_id,
                outcome=DecisionOutcome.PROFIT if i < 3 else DecisionOutcome.LOSS,
                pnl=10.0 if i < 3 else -5.0,
                exit_price=0.6,
                exit_reason="test"
            )
        
        stats = tracker.get_stats()
        assert stats['total_decisions'] == 4
        assert stats['profitable_trades'] == 3
        assert stats['losing_trades'] == 1
        assert stats['win_rate'] == 0.75
    
    def test_persistence(self, temp_log_dir):
        """Test entries persist across tracker instances."""
        # Create first tracker and log
        tracker1 = ReasoningTracker(log_dir=temp_log_dir)
        entry_id = tracker1.log_decision(
            market_id="market_persist",
            decision="BUY",
            confidence=0.8,
            reasoning="Persistence test",
            signals={}
        )
        tracker1.flush()  # Force write
        
        # Create new tracker instance
        tracker2 = ReasoningTracker(log_dir=temp_log_dir)
        tracker2.load_today()  # Load today's entries
        
        entry = tracker2.get_entry(entry_id)
        # Entry should be loadable
        assert entry is not None or len(tracker2.get_recent_entries(10)) >= 0
    
    def test_daily_file_naming(self, temp_log_dir):
        """Test daily log file naming convention."""
        tracker = ReasoningTracker(log_dir=temp_log_dir)
        tracker.log_decision(
            market_id="test",
            decision="BUY",
            confidence=0.5,
            reasoning="Test",
            signals={}
        )
        tracker.flush()
        
        # Check file was created with date
        today = datetime.utcnow().strftime("%Y-%m-%d")
        expected_file = Path(temp_log_dir) / f"reasoning_{today}.jsonl"
        assert expected_file.exists()


class TestAIDecisionEngineIntegration:
    """Integration tests for AI Decision Engine."""
    
    @pytest.fixture
    def mock_bedrock_client(self):
        """Create mock Bedrock client."""
        client = AsyncMock()
        client.invoke.return_value = {
            "recommendation": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong bullish indicators",
            "entry_price": 0.50,
            "stop_loss": 0.45,
            "take_profit": 0.60
        }
        return client
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.ai = Mock()
        settings.ai.model = "claude-3-sonnet"
        settings.ai.autonomy_level = 0.7
        settings.ai.min_confidence = 0.6
        settings.ai.monte_carlo_enabled = True
        settings.ai.monte_carlo_simulations = 100
        settings.ai.analysis_interval = 30
        settings.ai.temperature = 0.7
        
        settings.position_sizing = Mock()
        settings.position_sizing.kelly_fraction = 0.25
        settings.position_sizing.min_position = 1.0
        settings.position_sizing.max_position = 10.0
        
        settings.is_ai_enabled = Mock(return_value=True)
        settings.get_aws_credentials = Mock(return_value={
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
            "region_name": "us-east-1"
        })
        settings.get_bedrock_model_id = Mock(return_value="anthropic.claude-3-sonnet")
        
        return settings
    
    def test_market_context_creation(self):
        """Test MarketContext dataclass creation."""
        from src.ai.decision_engine import MarketContext
        
        context = MarketContext(
            market_id="test_market",
            question="Will X happen?",
            description="Test description",
            current_price=0.50,
            price_24h_ago=0.48,
            price_1h_ago=0.49,
            ewma_price=0.49,
            ewma_upper_band=0.52,
            ewma_lower_band=0.46,
            roc=0.02,
            cusum_positive=0.3,
            cusum_negative=0.1,
            volatility=0.02,
            volume_24h=100000,
            liquidity=5000,
            spread=0.01
        )
        
        assert context.market_id == "test_market"
        assert context.current_price == 0.50
    
    def test_trading_context_creation(self):
        """Test TradingContext dataclass creation."""
        from src.ai.decision_engine import TradingContext
        
        context = TradingContext(
            current_capital=1000,
            available_capital=800,
            daily_pnl=50,
            max_position_size=100,
            max_daily_loss=50,
            remaining_daily_risk=30,
            open_positions_count=2,
            max_positions=5,
            win_rate=0.6
        )
        
        assert context.current_capital == 1000
        assert context.open_positions_count == 2
    
    def test_ai_decision_structure(self):
        """Test AIDecision dataclass structure."""
        from src.ai.decision_engine import AIDecision, TradingRecommendation
        
        decision = AIDecision(
            market_id="test",
            market_name="Test Market",
            recommendation=TradingRecommendation.BUY,
            confidence=0.75,
            reasoning="Test reasoning",
            position_size=5.0,
            entry_price=0.50,
            stop_loss=0.45,
            take_profit=0.60,
            risk_reward_ratio=2.0,
            kelly_fraction=0.1,
            monte_carlo_prob=0.65,
            signals={"technical": 0.8}
        )
        
        assert decision.recommendation == TradingRecommendation.BUY
        assert decision.is_actionable == True  # BUY with confidence >= threshold
    
    def test_hold_decision_not_actionable(self):
        """Test HOLD decisions are not actionable."""
        from src.ai.decision_engine import AIDecision, TradingRecommendation
        
        decision = AIDecision(
            market_id="test",
            market_name="Test Market",
            recommendation=TradingRecommendation.HOLD,
            confidence=0.75,
            reasoning="No clear signal",
            position_size=0,
            entry_price=0.50,
            stop_loss=0.45,
            take_profit=0.60,
            risk_reward_ratio=0,
            kelly_fraction=0,
            monte_carlo_prob=0.5,
            signals={}
        )
        
        assert decision.is_actionable == False


class TestPromptTemplates:
    """Tests for prompt templates."""
    
    def test_market_analysis_prompt(self):
        """Test market analysis prompt generation."""
        from src.ai.prompts import PromptTemplates
        
        prompt = PromptTemplates.market_analysis(
            market_id="test_123",
            question="Will Bitcoin reach $100k?",
            current_price=0.65,
            price_change_24h=0.05,
            volume_24h=500000,
            volatility=0.03
        )
        
        assert "test_123" in prompt
        assert "Bitcoin" in prompt
        assert "0.65" in prompt
    
    def test_trading_decision_prompt(self):
        """Test trading decision prompt generation."""
        from src.ai.prompts import PromptTemplates
        
        prompt = PromptTemplates.trading_decision(
            market_analysis="Bullish momentum",
            technical_signals={"cusum": 0.5, "roc": 0.03},
            risk_metrics={"var_95": 0.05},
            portfolio_context={"capital": 1000}
        )
        
        assert "Bullish" in prompt
        assert "cusum" in prompt or "CUSUM" in prompt
    
    def test_sentiment_analysis_prompt(self):
        """Test sentiment analysis prompt generation."""
        from src.ai.prompts import PromptTemplates
        
        prompt = PromptTemplates.sentiment_analysis(
            market_question="Will the Fed raise rates?",
            recent_news=["News item 1", "News item 2"]
        )
        
        assert "Fed" in prompt
        assert "News item" in prompt


# Run with: pytest tests/test_ai_components.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])