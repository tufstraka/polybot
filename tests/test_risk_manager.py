"""
Unit tests for the Risk Manager.

Tests the risk management system including:
- Daily loss limits
- Position sizing
- Circuit breaker
- Capital protection
"""

import pytest
import os
import json
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk.risk_manager import RiskManager, RiskStatus, DailyStats


class TestRiskManager:
    """Tests for the RiskManager class."""
    
    @pytest.fixture
    def risk_manager(self, tmp_path):
        """Create a risk manager with temporary state file."""
        state_file = tmp_path / "risk_state.json"
        return RiskManager(
            initial_capital=75.0,
            max_daily_loss=2.0,
            max_position_size=2.0,
            min_position_size=0.5,
            max_consecutive_losses=3,
            circuit_breaker_minutes=30,
            min_capital_ratio=0.1,
            state_file=str(state_file)
        )
    
    def test_initialization(self, risk_manager):
        """Test proper initialization."""
        assert risk_manager.initial_capital == 75.0
        assert risk_manager.max_daily_loss == 2.0
        assert risk_manager.max_position_size == 2.0
        assert risk_manager.state.current_capital == 75.0
    
    def test_initial_status_is_ok(self, risk_manager):
        """Test that initial status is OK."""
        status = risk_manager.get_status()
        assert status == RiskStatus.OK
    
    def test_can_trade_initially(self, risk_manager):
        """Test that trading is allowed initially."""
        assert risk_manager.can_open_trade() == True
    
    def test_daily_loss_limit(self, risk_manager):
        """Test that daily loss limit stops trading."""
        # Record losses up to the limit
        risk_manager.record_trade_result(-1.0)  # Lost $1
        assert risk_manager.can_open_trade() == True
        
        risk_manager.record_trade_result(-1.0)  # Lost another $1, total -$2
        assert risk_manager.get_status() == RiskStatus.DAILY_LIMIT_HIT
        assert risk_manager.can_open_trade() == False
    
    def test_circuit_breaker_after_consecutive_losses(self, risk_manager):
        """Test circuit breaker triggers after consecutive losses."""
        # Record 3 consecutive losses
        risk_manager.record_trade_result(-0.30)
        risk_manager.record_trade_result(-0.30)
        risk_manager.record_trade_result(-0.30)
        
        status = risk_manager.get_status()
        assert status == RiskStatus.CIRCUIT_BREAKER
        assert risk_manager.can_open_trade() == False
    
    def test_consecutive_losses_reset_on_win(self, risk_manager):
        """Test that consecutive losses counter resets on a win."""
        # 2 losses
        risk_manager.record_trade_result(-0.30)
        risk_manager.record_trade_result(-0.30)
        assert risk_manager.state.daily_stats.consecutive_losses == 2
        
        # 1 win - should reset
        risk_manager.record_trade_result(0.50)
        assert risk_manager.state.daily_stats.consecutive_losses == 0
    
    def test_position_sizing_default(self, risk_manager):
        """Test default position sizing."""
        size = risk_manager.calculate_position_size()
        assert size == risk_manager.max_position_size  # 2.0
    
    def test_position_sizing_with_confidence(self, risk_manager):
        """Test position sizing scales with confidence."""
        # Low confidence = smaller position
        low_conf_size = risk_manager.calculate_position_size(confidence=0.5)
        high_conf_size = risk_manager.calculate_position_size(confidence=1.0)
        
        assert low_conf_size < high_conf_size
    
    def test_position_sizing_with_volatility(self, risk_manager):
        """Test position sizing adjusts for volatility."""
        # High volatility = smaller position
        low_vol_size = risk_manager.calculate_position_size(volatility_factor=0.5)
        high_vol_size = risk_manager.calculate_position_size(volatility_factor=2.0)
        
        assert high_vol_size < low_vol_size
    
    def test_position_sizing_minimum(self, risk_manager):
        """Test position size doesn't go below minimum."""
        # Very low confidence + high volatility
        size = risk_manager.calculate_position_size(
            confidence=0.1,
            volatility_factor=3.0
        )
        
        assert size >= risk_manager.min_position_size
    
    def test_position_sizing_near_daily_limit(self, risk_manager):
        """Test position size reduces when approaching daily limit."""
        # Use up most of daily limit
        risk_manager.record_trade_result(-1.5)  # Lost $1.50, only $0.50 remaining
        
        # Position size should be capped at remaining risk
        size = risk_manager.calculate_position_size()
        assert size <= 0.50
    
    def test_capital_tracking(self, risk_manager):
        """Test capital is properly tracked."""
        initial = risk_manager.state.current_capital
        
        risk_manager.record_trade_result(0.50)  # Win $0.50
        assert risk_manager.state.current_capital == initial + 0.50
        
        risk_manager.record_trade_result(-0.30)  # Lose $0.30
        assert risk_manager.state.current_capital == initial + 0.50 - 0.30
    
    def test_win_loss_counting(self, risk_manager):
        """Test wins and losses are counted correctly."""
        risk_manager.record_trade_result(0.50)  # Win
        risk_manager.record_trade_result(-0.30)  # Loss
        risk_manager.record_trade_result(0.20)  # Win
        
        stats = risk_manager.get_daily_stats()
        assert stats.wins == 2
        assert stats.losses == 1
    
    def test_trade_count(self, risk_manager):
        """Test total trade count."""
        risk_manager.record_trade_opened()
        risk_manager.record_trade_opened()
        risk_manager.record_trade_opened()
        
        stats = risk_manager.get_daily_stats()
        assert stats.trades_opened == 3
    
    def test_warning_status(self, risk_manager):
        """Test warning status near daily limit."""
        # Lose 70% of daily limit
        risk_manager.record_trade_result(-1.4)  # $1.40 / $2.00 = 70%
        
        status = risk_manager.get_status()
        assert status == RiskStatus.WARNING
        assert risk_manager.can_open_trade() == True  # Still allowed
    
    def test_low_capital_status(self, risk_manager):
        """Test low capital protection."""
        # Manually reduce capital to below threshold
        # Initial capital is $75, min ratio is 0.1, so min capital is $7.50
        risk_manager.state.current_capital = 5.0  # Below $7.50
        
        status = risk_manager.get_status()
        assert status == RiskStatus.LOW_CAPITAL
        assert risk_manager.can_open_trade() == False
    
    def test_summary(self, risk_manager):
        """Test summary generation."""
        risk_manager.record_trade_result(0.50)
        risk_manager.record_trade_result(-0.30)
        
        summary = risk_manager.get_summary()
        
        assert "status" in summary
        assert "can_trade" in summary
        assert "daily_pnl" in summary
        assert "current_capital" in summary
        assert "daily_win_rate" in summary
        
        # Check values
        assert summary["daily_pnl"] == 0.20  # 0.50 - 0.30
        assert summary["can_trade"] == True
    
    def test_state_persistence(self, tmp_path):
        """Test that state is persisted to disk."""
        state_file = tmp_path / "risk_state.json"
        
        # Create first instance and make some trades
        rm1 = RiskManager(
            initial_capital=75.0,
            max_daily_loss=2.0,
            state_file=str(state_file)
        )
        rm1.record_trade_result(0.50)
        rm1.record_trade_result(-0.20)
        
        # Create second instance - should load state
        rm2 = RiskManager(
            initial_capital=75.0,
            max_daily_loss=2.0,
            state_file=str(state_file)
        )
        
        # State should be preserved
        assert rm2.state.current_capital == rm1.state.current_capital
        assert rm2.state.daily_stats.realized_pnl == 0.30  # 0.50 - 0.20
    
    def test_reset_daily_stats(self, risk_manager):
        """Test manual reset of daily stats."""
        risk_manager.record_trade_result(0.50)
        risk_manager.record_trade_result(-0.30)
        
        risk_manager.reset_daily_stats()
        
        stats = risk_manager.get_daily_stats()
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.realized_pnl == 0.0
    
    def test_update_capital(self, risk_manager):
        """Test manual capital update."""
        risk_manager.update_capital(100.0)
        assert risk_manager.state.current_capital == 100.0


class TestRiskManagerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_profit_trade(self, tmp_path):
        """Test handling of break-even trades."""
        state_file = tmp_path / "risk_state.json"
        rm = RiskManager(state_file=str(state_file))
        
        rm.record_trade_result(0.0)  # Break even
        
        stats = rm.get_daily_stats()
        # Should count as... depends on implementation
        # Typically break-even is neither win nor loss
        assert stats.realized_pnl == 0.0
    
    def test_very_small_trade(self, tmp_path):
        """Test handling of very small P&L."""
        state_file = tmp_path / "risk_state.json"
        rm = RiskManager(state_file=str(state_file))
        
        rm.record_trade_result(0.001)  # Tiny win
        rm.record_trade_result(-0.001)  # Tiny loss
        
        # Should handle precision correctly
        assert abs(rm.state.daily_stats.realized_pnl) < 0.01
    
    def test_large_win(self, tmp_path):
        """Test handling of unexpectedly large win."""
        state_file = tmp_path / "risk_state.json"
        rm = RiskManager(
            initial_capital=75.0,
            state_file=str(state_file)
        )
        
        rm.record_trade_result(10.0)  # Big win
        
        assert rm.state.current_capital == 85.0
        assert rm.state.daily_stats.realized_pnl == 10.0
    
    def test_exact_daily_limit(self, tmp_path):
        """Test hitting exactly the daily limit."""
        state_file = tmp_path / "risk_state.json"
        rm = RiskManager(
            max_daily_loss=2.0,
            state_file=str(state_file)
        )
        
        rm.record_trade_result(-2.0)  # Exactly at limit
        
        assert rm.get_status() == RiskStatus.DAILY_LIMIT_HIT
    
    def test_position_sizing_after_wins(self, tmp_path):
        """Test position sizing doesn't increase unsafely after wins."""
        state_file = tmp_path / "risk_state.json"
        rm = RiskManager(
            max_position_size=2.0,
            state_file=str(state_file)
        )
        
        # Record big win
        rm.record_trade_result(5.0)
        
        # Position size should still be capped at max
        size = rm.calculate_position_size()
        assert size <= rm.max_position_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])