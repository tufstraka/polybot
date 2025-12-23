"""
Monte Carlo Simulation for Risk Assessment
===========================================

Implements Monte Carlo simulation to:
1. Estimate probability distributions of trade outcomes
2. Calculate Value at Risk (VaR) and Expected Shortfall
3. Stress test position sizing
4. Validate AI confidence scores

The simulation uses historical volatility and price patterns
to generate thousands of potential future price paths.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random

import numpy as np

from src.config.settings import Settings, get_settings


logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Results from Monte Carlo simulation.
    
    Provides statistical measures of potential trade outcomes.
    """
    # Basic statistics
    mean_return: float = 0.0
    median_return: float = 0.0
    std_dev: float = 0.0
    
    # Return distribution
    min_return: float = 0.0
    max_return: float = 0.0
    percentile_5: float = 0.0   # 5th percentile (95% VaR)
    percentile_25: float = 0.0  # 25th percentile
    percentile_75: float = 0.0  # 75th percentile
    percentile_95: float = 0.0  # 95th percentile
    
    # Risk metrics
    var_95: float = 0.0        # Value at Risk (95%)
    var_99: float = 0.0        # Value at Risk (99%)
    expected_shortfall: float = 0.0  # CVaR / Expected Shortfall
    max_drawdown: float = 0.0
    
    # Probability estimates
    prob_profit: float = 0.0   # Probability of positive return
    prob_loss: float = 0.0     # Probability of negative return
    prob_stop_hit: float = 0.0 # Probability of hitting stop loss
    prob_tp_hit: float = 0.0   # Probability of hitting take profit
    
    # Simulation metadata
    num_simulations: int = 0
    num_steps: int = 0
    volatility_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio from simulation."""
        if self.var_95 == 0:
            return 0.0
        return abs(self.mean_return / self.var_95) if self.var_95 < 0 else 0.0
    
    @property
    def sharpe_estimate(self) -> float:
        """Estimate Sharpe-like ratio from simulation."""
        if self.std_dev == 0:
            return 0.0
        return self.mean_return / self.std_dev
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "std_dev": self.std_dev,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "percentile_5": self.percentile_5,
            "percentile_95": self.percentile_95,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "expected_shortfall": self.expected_shortfall,
            "prob_profit": self.prob_profit,
            "prob_loss": self.prob_loss,
            "prob_stop_hit": self.prob_stop_hit,
            "prob_tp_hit": self.prob_tp_hit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "sharpe_estimate": self.sharpe_estimate,
            "num_simulations": self.num_simulations,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def get_risk_assessment(self) -> str:
        """Get human-readable risk assessment."""
        if self.prob_loss > 0.6:
            return "HIGH_RISK"
        elif self.prob_loss > 0.4:
            return "MODERATE_RISK"
        elif self.prob_profit > 0.6:
            return "FAVORABLE"
        else:
            return "NEUTRAL"


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for trade risk assessment.
    
    Uses Geometric Brownian Motion (GBM) to simulate price paths:
    dS = μSdt + σSdW
    
    Where:
    - S = price
    - μ = drift (expected return)
    - σ = volatility
    - dW = Wiener process (random walk)
    
    Features:
    - Configurable number of simulations
    - Support for mean-reverting dynamics (Ornstein-Uhlenbeck)
    - Stop loss and take profit boundary conditions
    - Position sizing validation
    
    Usage:
        simulator = MonteCarloSimulator()
        result = simulator.simulate_trade(
            entry_price=0.65,
            position_size=2.0,
            stop_loss=0.60,
            take_profit=0.75,
            volatility=0.02,
            time_steps=100,
        )
        print(f"Probability of profit: {result.prob_profit:.1%}")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        num_simulations: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            settings: Optional settings object
            num_simulations: Override number of simulations
            seed: Random seed for reproducibility
        """
        self.settings = settings or get_settings()
        self.num_simulations = num_simulations or self.settings.ai.monte_carlo_simulations
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def simulate_trade(
        self,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        volatility: float,
        direction: str = "BUY",
        time_steps: int = 100,
        mean_reversion_strength: float = 0.0,
        fair_value: Optional[float] = None,
    ) -> SimulationResult:
        """
        Simulate potential outcomes for a trade.
        
        Args:
            entry_price: Entry price (0-1 for prediction markets)
            position_size: Position size in dollars
            stop_loss: Stop loss price
            take_profit: Take profit price
            volatility: Historical or implied volatility
            direction: "BUY" or "SELL"
            time_steps: Number of time steps per simulation
            mean_reversion_strength: Strength of mean reversion (0-1)
            fair_value: Fair value for mean reversion (default: entry_price)
            
        Returns:
            SimulationResult with comprehensive risk metrics
        """
        if self.num_simulations <= 0:
            logger.warning("Monte Carlo disabled (num_simulations=0)")
            return SimulationResult()
        
        is_buy = direction.upper() == "BUY"
        fair_value = fair_value or entry_price
        
        # Ensure volatility is reasonable
        volatility = max(0.001, min(0.5, volatility))
        
        # Run simulations
        final_returns = []
        stop_hits = 0
        tp_hits = 0
        max_drawdowns = []
        
        for _ in range(self.num_simulations):
            path, hit_stop, hit_tp, max_dd = self._simulate_path(
                start_price=entry_price,
                volatility=volatility,
                time_steps=time_steps,
                stop_loss=stop_loss,
                take_profit=take_profit,
                is_buy=is_buy,
                mean_reversion_strength=mean_reversion_strength,
                fair_value=fair_value,
            )
            
            # Calculate return
            final_price = path[-1]
            if is_buy:
                pnl = (final_price - entry_price) * position_size / entry_price
            else:
                pnl = (entry_price - final_price) * position_size / entry_price
            
            final_returns.append(pnl)
            max_drawdowns.append(max_dd)
            
            if hit_stop:
                stop_hits += 1
            if hit_tp:
                tp_hits += 1
        
        # Calculate statistics
        returns_array = np.array(final_returns)
        
        result = SimulationResult(
            mean_return=float(np.mean(returns_array)),
            median_return=float(np.median(returns_array)),
            std_dev=float(np.std(returns_array)),
            min_return=float(np.min(returns_array)),
            max_return=float(np.max(returns_array)),
            percentile_5=float(np.percentile(returns_array, 5)),
            percentile_25=float(np.percentile(returns_array, 25)),
            percentile_75=float(np.percentile(returns_array, 75)),
            percentile_95=float(np.percentile(returns_array, 95)),
            var_95=float(np.percentile(returns_array, 5)),  # 5th percentile = 95% VaR
            var_99=float(np.percentile(returns_array, 1)),  # 1st percentile = 99% VaR
            expected_shortfall=float(np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)])),
            max_drawdown=float(np.mean(max_drawdowns)),
            prob_profit=float(np.sum(returns_array > 0) / len(returns_array)),
            prob_loss=float(np.sum(returns_array < 0) / len(returns_array)),
            prob_stop_hit=stop_hits / self.num_simulations,
            prob_tp_hit=tp_hits / self.num_simulations,
            num_simulations=self.num_simulations,
            num_steps=time_steps,
            volatility_used=volatility,
        )
        
        logger.debug(
            f"MC Simulation: {self.num_simulations} runs, "
            f"mean={result.mean_return:.4f}, "
            f"prob_profit={result.prob_profit:.1%}"
        )
        
        return result
    
    def _simulate_path(
        self,
        start_price: float,
        volatility: float,
        time_steps: int,
        stop_loss: float,
        take_profit: float,
        is_buy: bool,
        mean_reversion_strength: float,
        fair_value: float,
    ) -> Tuple[List[float], bool, bool, float]:
        """
        Simulate a single price path using GBM with optional mean reversion.
        
        Returns:
            Tuple of (price_path, hit_stop, hit_tp, max_drawdown)
        """
        path = [start_price]
        current_price = start_price
        hit_stop = False
        hit_tp = False
        
        # For drawdown calculation
        peak_pnl = 0.0
        max_drawdown = 0.0
        
        # Time step size (normalized)
        dt = 1.0 / time_steps
        sqrt_dt = math.sqrt(dt)
        
        for _ in range(time_steps):
            # Random shock (Wiener process)
            dW = np.random.normal(0, 1) * sqrt_dt
            
            # Mean reversion component (Ornstein-Uhlenbeck)
            if mean_reversion_strength > 0:
                mr_drift = mean_reversion_strength * (fair_value - current_price) * dt
            else:
                mr_drift = 0
            
            # GBM price change
            # Using log-normal to prevent negative prices
            drift = 0  # Zero drift assumption
            price_change = current_price * (drift * dt + volatility * dW) + mr_drift
            
            current_price = current_price + price_change
            
            # Bound price between 0.001 and 0.999 (prediction market constraints)
            current_price = max(0.001, min(0.999, current_price))
            
            path.append(current_price)
            
            # Check stop loss and take profit
            if is_buy:
                if current_price <= stop_loss:
                    hit_stop = True
                    current_price = stop_loss
                    break
                if current_price >= take_profit:
                    hit_tp = True
                    current_price = take_profit
                    break
                
                # Track drawdown
                current_pnl = current_price - start_price
            else:
                if current_price >= stop_loss:
                    hit_stop = True
                    current_price = stop_loss
                    break
                if current_price <= take_profit:
                    hit_tp = True
                    current_price = take_profit
                    break
                
                current_pnl = start_price - current_price
            
            if current_pnl > peak_pnl:
                peak_pnl = current_pnl
            drawdown = peak_pnl - current_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return path, hit_stop, hit_tp, max_drawdown
    
    def validate_position_size(
        self,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        volatility: float,
        max_acceptable_loss: float,
        confidence_level: float = 0.95,
    ) -> Tuple[bool, float, str]:
        """
        Validate if position size is appropriate given risk parameters.
        
        Uses Monte Carlo to estimate if position size could exceed
        acceptable loss limits.
        
        Args:
            position_size: Proposed position size
            entry_price: Entry price
            stop_loss: Stop loss price
            volatility: Historical volatility
            max_acceptable_loss: Maximum acceptable dollar loss
            confidence_level: Confidence level for VaR (default 95%)
            
        Returns:
            Tuple of (is_valid, recommended_size, reason)
        """
        result = self.simulate_trade(
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=entry_price * 1.1,  # Dummy TP
            volatility=volatility,
        )
        
        # Check VaR at confidence level
        var = result.var_95 if confidence_level == 0.95 else result.var_99
        
        if abs(var) <= max_acceptable_loss:
            return True, position_size, "Position size within risk limits"
        
        # Calculate recommended size
        if var != 0:
            size_multiplier = max_acceptable_loss / abs(var)
            recommended = position_size * size_multiplier * 0.9  # 10% buffer
        else:
            recommended = position_size
        
        return False, recommended, f"VaR ${abs(var):.2f} exceeds limit ${max_acceptable_loss:.2f}"
    
    def estimate_optimal_holding_period(
        self,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        volatility: float,
        max_periods: int = 200,
    ) -> Dict[str, float]:
        """
        Estimate optimal holding period for a trade.
        
        Runs simulations at different time horizons to find
        the period with best risk-adjusted returns.
        
        Returns:
            Dictionary with optimal period and expected metrics
        """
        best_period = 0
        best_sharpe = -999
        results = {}
        
        for periods in [10, 25, 50, 100, max_periods]:
            result = self.simulate_trade(
                entry_price=entry_price,
                position_size=1.0,  # Normalized
                stop_loss=stop_loss,
                take_profit=take_profit,
                volatility=volatility,
                time_steps=periods,
            )
            
            if result.sharpe_estimate > best_sharpe:
                best_sharpe = result.sharpe_estimate
                best_period = periods
            
            results[periods] = result.sharpe_estimate
        
        return {
            "optimal_periods": best_period,
            "optimal_sharpe": best_sharpe,
            "period_sharpes": results,
        }


# Kelly Criterion integration
def kelly_adjusted_by_simulation(
    kelly_fraction: float,
    simulation_result: SimulationResult,
    max_adjustment: float = 0.5,
) -> float:
    """
    Adjust Kelly Criterion fraction based on Monte Carlo results.
    
    Reduces Kelly sizing if simulation shows high risk.
    
    Args:
        kelly_fraction: Raw Kelly fraction
        simulation_result: Monte Carlo simulation result
        max_adjustment: Maximum adjustment factor
        
    Returns:
        Adjusted Kelly fraction
    """
    # Reduce Kelly if probability of loss is high
    risk_adjustment = 1.0 - (simulation_result.prob_loss * max_adjustment)
    
    # Further reduce if VaR is concerning
    var_ratio = abs(simulation_result.var_95) / max(0.01, simulation_result.mean_return)
    if var_ratio > 3:  # VaR more than 3x mean return
        risk_adjustment *= 0.8
    
    # Apply risk assessment
    if simulation_result.get_risk_assessment() == "HIGH_RISK":
        risk_adjustment *= 0.7
    
    adjusted_kelly = kelly_fraction * max(0.2, risk_adjustment)
    
    logger.debug(
        f"Kelly adjustment: {kelly_fraction:.3f} -> {adjusted_kelly:.3f} "
        f"(risk_adj={risk_adjustment:.2f})"
    )
    
    return adjusted_kelly


if __name__ == "__main__":
    # Test Monte Carlo simulation
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("MONTE CARLO SIMULATION TEST")
    print("=" * 60)
    
    simulator = MonteCarloSimulator(num_simulations=1000)
    
    # Test trade simulation
    result = simulator.simulate_trade(
        entry_price=0.65,
        position_size=2.0,
        stop_loss=0.58,
        take_profit=0.75,
        volatility=0.02,
        direction="BUY",
        time_steps=100,
    )
    
    print("\nSimulation Results:")
    print(f"  Simulations: {result.num_simulations}")
    print(f"  Mean Return: ${result.mean_return:.4f}")
    print(f"  Std Dev: ${result.std_dev:.4f}")
    print(f"  Min/Max: ${result.min_return:.4f} / ${result.max_return:.4f}")
    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): ${result.var_95:.4f}")
    print(f"  VaR (99%): ${result.var_99:.4f}")
    print(f"  Expected Shortfall: ${result.expected_shortfall:.4f}")
    print(f"\nProbabilities:")
    print(f"  Profit: {result.prob_profit:.1%}")
    print(f"  Loss: {result.prob_loss:.1%}")
    print(f"  Stop Hit: {result.prob_stop_hit:.1%}")
    print(f"  TP Hit: {result.prob_tp_hit:.1%}")
    print(f"\nAssessment: {result.get_risk_assessment()}")
    
    # Test position validation
    print("\n" + "=" * 60)
    print("POSITION SIZE VALIDATION")
    print("=" * 60)
    
    is_valid, recommended, reason = simulator.validate_position_size(
        position_size=5.0,
        entry_price=0.65,
        stop_loss=0.58,
        volatility=0.02,
        max_acceptable_loss=0.50,
    )
    
    print(f"Position $5.00: {'✅ Valid' if is_valid else '❌ Invalid'}")
    print(f"Recommended: ${recommended:.2f}")
    print(f"Reason: {reason}")