
"""
Polybot - AI-Powered Autonomous Trading Bot

Plain English Explanation:
==========================

This is the "brain" of the bot. It coordinates all the other parts:

1. STARTUP:
   - Loads your configuration
   - Connects to Polymarket
   - Initializes AI Decision Engine (Amazon Bedrock)
   - Sets up notifications (Telegram/Discord)
   - Initializes risk management

2. MAIN LOOP (runs every few seconds):
   - Scans for active markets
   - Gets latest prices
   - AI analyzes markets and generates signals
   - If signal found and risk allows → opens trade
   - Monitors open positions for exit conditions
   - Updates dashboard state file

3. SHUTDOWN:
   - Saves state to disk
   - Sends final notification
   - Closes connections

Run with:
    python -m src.bot
    
Or for dry-run testing:
    python -m src.bot --dry-run
"""

import asyncio
import logging
import signal
import sys
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings, Settings
from src.core.client import PolymarketClient, Market
from src.core.scanner import MarketScanner, MarketSnapshot
from src.core.tracker import PriceTracker
from src.core.detector import SpikeDetector, DetectionResult, TradingSignal, SignalDirection
from src.core.executor import OrderExecutor, ExecutionStatus
from src.core.position_manager import PositionManager
from src.core.state_writer import (
    StateWriter, MarketState, SignalState, PositionState,
    RiskState, PerformanceState, AIStatsState, AIReasoningEntry, MonteCarloState
)
from src.risk.risk_manager import RiskManager, RiskStatus
from src.notifications.base import NotificationManager
from src.notifications.telegram import TelegramNotifier
from src.notifications.discord import DiscordNotifier

# AI Decision Engine (optional - graceful degradation if not available)
try:
    from src.ai import (
        AIDecisionEngine,
        AIDecision,
        TradingRecommendation,
        MarketContext,
        TradingContext,
    )
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    AIDecisionEngine = None
    AIDecision = None
    TradingRecommendation = None
    MarketContext = None
    TradingContext = None
    logging.warning(f"AI module not available: {e}")


class Polybot:
    """
    Main bot orchestrator with AI-powered decision making.
    
    Coordinates all modules to run the AI trading strategy:
    - Technical analysis (CUSUM, EWMA, ROC)
    - AI market analysis (Amazon Bedrock)
    - Ensemble signal generation
    - Risk management (Kelly Criterion, Monte Carlo)
    
    Usage:
        bot = Polybot()
        await bot.start()
    """
    
    def __init__(self, config_path: str = "config/config.yaml", dry_run: bool = None):
        """
        Initialize the bot.
        
        Args:
            config_path: Path to configuration file
            dry_run: Override dry_run setting (None = use config)
        """
        # Load configuration
        self.config: Settings = get_settings(config_path)
        
        # Override dry run if specified
        if dry_run is not None:
            self.dry_run = dry_run
        else:
            self.dry_run = self.config.mode.paper_trading
        
        # Check AI availability
        self.ai_enabled = AI_AVAILABLE and self.config.is_ai_enabled()
        
        # Initialize logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Polybot (dry_run={self.dry_run}, ai_enabled={self.ai_enabled})")
        
        # Core components (initialized in start())
        self.client: Optional[PolymarketClient] = None
        self.scanner: Optional[MarketScanner] = None
        self.tracker: Optional[PriceTracker] = None
        self.detector: Optional[SpikeDetector] = None
        self.executor: Optional[OrderExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.state_writer: Optional[StateWriter] = None
        self.notifier: Optional[NotificationManager] = None
        
        # AI components (optional)
        self.ai_engine: Optional[AIDecisionEngine] = None
        self._ai_analysis_count = 0
        self._last_ai_analysis: Optional[datetime] = None
        
        # Track AI decision IDs for outcome recording
        self._ai_decision_map: Dict[str, str] = {}  # position_id -> reasoning_entry_id
        
        # State tracking
        self._running = False
        self._start_time: Optional[datetime] = None
        self._tracked_markets: Dict[str, dict] = {}  # market_id -> market data
        self._today: Optional[str] = None
        
    def _setup_logging(self):
        """Configure logging based on settings."""
        log_level = getattr(logging, self.config.env.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(level=log_level, handlers=[console])
    
    async def _initialize_components(self):
        """Initialize all bot components."""
        self.logger.info("Initializing components...")
        
        # Polymarket client
        self.client = PolymarketClient(settings=self.config)
        await self.client.initialize()
        
        # Market scanner - takes client and optional settings
        self.scanner = MarketScanner(
            client=self.client,
            settings=self.config
        )
        
        # Price tracker - takes optional settings
        self.tracker = PriceTracker(settings=self.config)
        
        # Spike detector - takes tracker and optional settings
        self.detector = SpikeDetector(
            tracker=self.tracker,
            settings=self.config
        )
        
        # Order executor - takes client and optional settings
        self.executor = OrderExecutor(
            client=self.client,
            settings=self.config
        )
        
        # Position manager - takes optional settings and state_file
        self.position_manager = PositionManager(
            settings=self.config,
            state_file="data/positions.json"
        )
        
        # Risk manager - has explicit parameters
        self.risk_manager = RiskManager(
            initial_capital=self.config.money.starting_balance,
            max_daily_loss=self.config.money.max_daily_loss,
            max_position_size=self.config.money.bet_size,
            min_position_size=0.5,  # Minimum $0.50
            max_consecutive_losses=3,
            circuit_breaker_minutes=30,
            state_file="data/risk_state.json"
        )
        
        # State writer for dashboard
        self.state_writer = StateWriter(
            state_file="data/bot_state.json"
        )
        
        # Notification manager
        self.notifier = NotificationManager()
        
        # Add Telegram notifier if configured
        telegram_config = self.config.get_telegram_config()
        if telegram_config:
            telegram = TelegramNotifier(
                bot_token=telegram_config[0],
                chat_id=telegram_config[1],
                enabled=True
            )
            self.notifier.add_notifier(telegram)
        
        # Add Discord notifier if configured
        discord_url = self.config.get_discord_config()
        if discord_url:
            discord = DiscordNotifier(
                webhook_url=discord_url,
                enabled=True
            )
            self.notifier.add_notifier(discord)
        
        # Initialize AI Decision Engine if enabled
        if self.ai_enabled:
            try:
                self.ai_engine = AIDecisionEngine(settings=self.config)
                self.logger.info(f"AI Decision Engine initialized (model: {self.config.get_bedrock_model_id()})")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI engine: {e}")
                self.ai_engine = None
                self.ai_enabled = False
        
        self.logger.info("All components initialized")
    
    async def start(self):
        """Start the bot."""
        self.logger.info("Starting Polybot...")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Mark as running
            self._running = True
            self._start_time = datetime.utcnow()
            self._today = date.today().isoformat()
            
            # Update state
            self.state_writer.update_bot_status(
                "running",
                "dry_run" if self.dry_run else "live"
            )
            self.state_writer.update_ai_enabled(self.ai_enabled)
            ai_status = "enabled" if self.ai_enabled else "disabled"
            model_info = self.config.get_bedrock_model_id() if self.ai_enabled else "N/A"
            self.state_writer.add_event("system", f"Bot started (AI: {ai_status}, Model: {model_info})")
            
            self.state_writer.flush(force=True)
            
            # Send startup notification
            ai_status = "🤖 AI Enabled" if self.ai_enabled else "📊 Technical Only"
            await self.notifier.notify_status(
                f"🚀 Polybot started ({'DRY RUN' if self.dry_run else 'LIVE'}) - {ai_status}",
                {
                    "capital": self.config.money.starting_balance,
                    "daily_limit": self.config.money.max_daily_loss,
                    "ai_enabled": self.ai_enabled,
                    "model": self.config.get_bedrock_model_id() if self.ai_enabled else "N/A"
                }
            )
            
            self.logger.info("Bot started successfully")
            
            # Run main loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.logger.error(traceback.format_exc())
            if self.state_writer:
                self.state_writer.record_error(str(e))
            if self.notifier:
                await self.notifier.notify_error(str(e), traceback.format_exc())
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the bot gracefully."""
        self.logger.info("Stopping Polybot...")
        self._running = False
        
        # Update state
        if self.state_writer:
            self.state_writer.update_bot_status("stopped")
            self.state_writer.add_event("system", "Bot stopped")
            self.state_writer.flush(force=True)
        
        # Send shutdown notification
        if self.notifier:
            try:
                await self.notifier.notify_status("🛑 Polybot stopped")
            except:
                pass
        
        self.logger.info("Bot stopped")
    
    async def _main_loop(self):
        """Main trading loop."""
        self.logger.info("Entering main loop...")
        
        poll_interval = self.config.polling.interval_seconds
        market_refresh_interval = self.config.polling.market_refresh_seconds
        
        loop_count = 0
        
        while self._running:
            try:
                loop_start = datetime.utcnow()
                loop_count += 1
                
                # Check for new day
                self._check_new_day()
                
                # Refresh market list periodically
                if loop_count % market_refresh_interval == 1 or not self._tracked_markets:
                    await self._refresh_markets()
                
                # Update prices for all tracked markets
                await self._update_prices()
                
                # Check for spike signals
                await self._check_signals()
                
                # Monitor open positions
                await self._monitor_positions()
                
                # Update state file
                self._update_state()
                
                # Wait for next iteration
                elapsed = (datetime.utcnow() - loop_start).total_seconds()
                sleep_time = max(0, poll_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                self.logger.info("Main loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.logger.error(traceback.format_exc())
                if self.state_writer:
                    self.state_writer.record_error(str(e))
                await asyncio.sleep(poll_interval)
    
    def _check_new_day(self):
        """Check if it's a new trading day."""
        today = date.today().isoformat()
        if today != self._today:
            self.logger.info(f"New trading day: {today}")
            self._today = today
            self._ai_analysis_count = 0  # Reset AI counter
            self.state_writer.reset_daily_counters()
            self.state_writer.add_event("system", f"New trading day: {today}")
    
    async def _refresh_markets(self):
        """Refresh the list of markets to track."""
        self.logger.debug("Refreshing market list...")
        
        try:
            # Use scanner to refresh and get ALL tradable markets
            await self.scanner.refresh_markets(max_markets=None)  # No limit
            markets = await self.scanner.get_tradable_markets(limit=None)  # No limit
            
            # Update tracked markets
            self._tracked_markets.clear()
            for market in markets:
                self._tracked_markets[market.id] = market
            
            self.logger.info(f"Tracking {len(self._tracked_markets)} markets (ALL)")
            self.state_writer.add_event(
                "system",
                f"Refreshed markets: {len(self._tracked_markets)} active (ALL available)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to refresh markets: {e}")
    
    async def _update_prices(self):
        """Update prices for all tracked markets with rate limiting."""
        market_list = list(self._tracked_markets.items())
        
        # Get rate limiting settings
        batch_size = self.config.polling.batch_size
        batch_delay = self.config.polling.batch_delay
        request_delay = self.config.rate_limits.request_delay
        
        # Process in batches to avoid overwhelming the API
        total_markets = len(market_list)
        batches_processed = 0
        
        for i in range(0, total_markets, batch_size):
            batch = market_list[i:i + batch_size]
            batches_processed += 1
            
            for market_id, market in batch:
                try:
                    # Get snapshot with orderbook from scanner
                    snapshot = await self.scanner.get_market_snapshot(market_id)
                    
                    if not snapshot:
                        continue
                    
                    # Update tracker with snapshot
                    indicators = self.tracker.update(snapshot)
                    
                    # Update state for dashboard
                    if indicators:
                        market_state = MarketState(
                            market_id=market_id,
                            name=market.question[:50] if market.question else market_id,
                            current_price=indicators.current_price,
                            ewma_price=indicators.ewma_mean,
                            ewma_upper=indicators.ewma_upper_band,
                            ewma_lower=indicators.ewma_lower_band,
                            roc=indicators.roc,
                            cusum_pos=indicators.cusum_positive,
                            cusum_neg=indicators.cusum_negative,
                            volume_24h=market.volume_24h,
                            last_updated=datetime.utcnow().isoformat()
                        )
                        self.state_writer.update_market(market_state)
                    
                    # Rate limit: delay between individual requests
                    await asyncio.sleep(request_delay)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to update price for {market_id}: {e}")
            
            # Delay between batches (if more batches remaining)
            if i + batch_size < total_markets:
                self.logger.debug(f"Processed batch {batches_processed}/{(total_markets + batch_size - 1) // batch_size}")
                await asyncio.sleep(batch_delay)
    
    async def _check_signals(self):
        """Check for signals using AI or technical analysis."""
        # Skip if we can't trade
        if not self.risk_manager.can_open_trade():
            self.logger.debug("Cannot open trade - risk limit or circuit breaker")
            return
        
        # Determine if we should run AI analysis this cycle
        should_run_ai = self._should_run_ai_analysis()
        
        markets_with_data = 0
        markets_analyzed = 0
        
        # Check each tracked market
        for market_id, market in list(self._tracked_markets.items()):
            try:
                # Skip if we already have a position
                if self.position_manager.has_position_for_market(market_id):
                    continue
                
                # Get history to check if we have enough data
                history = self.tracker.get_history(market_id)
                if not history:
                    continue
                
                # Check minimum data requirement (reduced for faster startup)
                min_data_points = 5  # Only need 5 data points to start
                if len(history.prices) < min_data_points:
                    continue
                
                markets_with_data += 1
                
                # Get snapshot with orderbook
                snapshot = await self.scanner.get_market_snapshot(market_id)
                if not snapshot or not snapshot.is_tradable:
                    continue
                
                markets_analyzed += 1
                
                # Use AI analysis if available and enabled
                if self.ai_enabled and self.ai_engine and should_run_ai:
                    self.logger.info(f"🤖 Running AI analysis on: {market.question[:40]}...")
                    await self._check_ai_signal(snapshot, market)
                else:
                    # Fallback to pure technical analysis
                    result = self.detector.check(snapshot)
                    if result.is_signal:
                        await self._handle_signal(snapshot, result)
                    
            except Exception as e:
                self.logger.warning(f"Signal check failed for {market_id}: {e}")
        
        # Log progress periodically
        if markets_with_data > 0 or markets_analyzed > 0:
            self.logger.debug(f"Signal check: {markets_with_data} markets have data, {markets_analyzed} analyzed")
        
        # Log AI status
        if should_run_ai and markets_analyzed > 0:
            self.logger.info(f"🤖 AI analysis completed on {markets_analyzed} markets (Total AI calls: {self._ai_analysis_count})")
    
    def _should_run_ai_analysis(self) -> bool:
        """Determine if we should run AI analysis this cycle."""
        if not self.ai_enabled or not self.ai_engine:
            return False
        
        # Check analysis interval
        analysis_interval = self.config.ai.analysis_interval
        
        if self._last_ai_analysis is None:
            return True
        
        elapsed = (datetime.utcnow() - self._last_ai_analysis).total_seconds()
        return elapsed >= analysis_interval
    
    async def _check_ai_signal(self, snapshot: MarketSnapshot, market):
        """Check for trading signals using AI decision engine."""
        try:
            # Build market context for AI
            history = self.tracker.get_history(snapshot.market.id)
            indicators = history.indicators if history else None
            
            # Calculate liquidity from bid/ask depth
            liquidity = (snapshot.bid_depth + snapshot.ask_depth) if snapshot.orderbook else 0
            spread = snapshot.spread if snapshot.spread else 0
            
            market_context = MarketContext(
                market_id=snapshot.market.id,
                question=snapshot.market.question or "",
                description=getattr(snapshot.market, 'description', ''),
                current_price=snapshot.price,
                price_24h_ago=indicators.current_price if indicators else snapshot.price,
                price_1h_ago=indicators.current_price if indicators else snapshot.price,
                ewma_price=indicators.ewma_mean if indicators else snapshot.price,
                ewma_upper_band=indicators.ewma_upper_band if indicators else snapshot.price * 1.05,
                ewma_lower_band=indicators.ewma_lower_band if indicators else snapshot.price * 0.95,
                roc=indicators.roc if indicators else 0.0,
                cusum_positive=indicators.cusum_positive if indicators else 0.0,
                cusum_negative=indicators.cusum_negative if indicators else 0.0,
                volatility=indicators.volatility if indicators else 0.02,
                volume_24h=market.volume_24h,
                liquidity=liquidity,
                spread=spread,
            )
            
            # Build trading context
            risk_summary = self.risk_manager.get_summary()
            trading_context = TradingContext(
                current_capital=self.risk_manager.state.current_capital,
                available_capital=risk_summary["remaining_daily_risk"],
                daily_pnl=risk_summary["daily_pnl"],
                max_position_size=self.config.money.bet_size,
                max_daily_loss=self.config.money.max_daily_loss,
                remaining_daily_risk=risk_summary["remaining_daily_risk"],
                open_positions_count=len(self.position_manager.open_positions),
                max_positions=self.config.risk.max_open_trades,
                win_rate=risk_summary["daily_win_rate"],
            )
            
            # Get AI decision
            decision = await self.ai_engine.analyze_market(market_context, trading_context)
            
            self._last_ai_analysis = datetime.utcnow()
            self._ai_analysis_count += 1
            
            # Handle actionable decision - ALWAYS execute in simulation mode
            if decision.is_actionable or (self.dry_run and decision.recommendation != "HOLD"):
                await self._handle_ai_decision(snapshot, decision)
                
        except Exception as e:
            self.logger.error(f"AI analysis failed for {snapshot.market.id}: {e}")
            # Fall back to technical analysis
            result = self.detector.check(snapshot)
            if result.is_signal:
                await self._handle_signal(snapshot, result)
    
    async def _handle_ai_decision(self, snapshot: MarketSnapshot, decision):
        """Handle an AI trading decision."""
        market_name = decision.market_name or snapshot.market.question[:50]
        market_id = snapshot.market.id
        
        self.logger.info(
            f"🤖 AI DECISION: {decision.recommendation} {market_name} | "
            f"Confidence: {decision.confidence:.0%} | "
            f"Size: ${decision.position_size:.2f}"
        )
        
        # Create a TradingSignal compatible with executor
        if decision.recommendation == TradingRecommendation.BUY:
            direction = SignalDirection.BUY
        elif decision.recommendation == TradingRecommendation.SELL:
            direction = SignalDirection.SELL
        else:
            return  # HOLD - do nothing
        
        trading_signal = TradingSignal(
            market_id=market_id,
            token_id=snapshot.market.tokens[0].token_id if snapshot.market.tokens else market_id,
            direction=direction,
            confidence=decision.confidence,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            cusum_value=0,
            roc_value=0,
            volatility=0,
        )
        
        # Create signal state for dashboard
        signal_id = f"ai_{datetime.utcnow().timestamp():.0f}"
        signal_state = SignalState(
            signal_id=signal_id,
            market_id=market_id,
            market_name=market_name,
            direction=decision.recommendation,
            price=snapshot.price,
            confidence=decision.confidence,
            detected_at=datetime.utcnow().isoformat(),
            trigger_reason=f"AI: {decision.reasoning[:100]}..."
        )
        self.state_writer.add_signal(signal_state)
        
        # Send notification
        await self.notifier.notify_signal(
            market=market_name,
            direction=decision.recommendation,
            price=snapshot.price,
            confidence=decision.confidence,
            market_id=market_id
        )
        
        # Execute trade
        try:
            execution_result = await self.executor.execute(trading_signal)
            
            if execution_result.status == ExecutionStatus.SUCCESS:
                # Record position
                position = self.position_manager.open_position(
                    signal=trading_signal,
                    execution=execution_result,
                    market_name=market_name
                )
                
                # Track AI decision for outcome recording
                if decision.reasoning_entry_id:
                    self._ai_decision_map[position.id] = decision.reasoning_entry_id
                
                # Record in risk manager
                self.risk_manager.record_trade_opened()
                
                # Update signal status
                self.state_writer.update_signal_status(signal_id, "traded")
                
                # Update position state
                pos_state = PositionState(
                    position_id=position.id,
                    market_id=market_id,
                    market_name=market_name,
                    side=decision.recommendation,
                    entry_price=execution_result.executed_price,
                    current_price=snapshot.price,
                    size=execution_result.executed_size,
                    unrealized_pnl=0.0,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    opened_at=datetime.utcnow().isoformat()
                )
                self.state_writer.update_position(pos_state)
                
                # Notify
                await self.notifier.notify_trade_open(
                    market=market_name,
                    side=decision.recommendation,
                    price=execution_result.executed_price,
                    size=execution_result.executed_size,
                    market_id=market_id
                )
                
                self.logger.info(
                    f"✅ AI Trade opened: {decision.recommendation} "
                    f"${execution_result.executed_size:.2f} @ ${execution_result.executed_price:.3f}"
                )
            else:
                self.logger.warning(f"AI trade failed: {execution_result.error_message}")
                self.state_writer.update_signal_status(signal_id, "failed")
                
        except Exception as e:
            self.logger.error(f"Failed to execute AI trade: {e}")
            self.state_writer.update_signal_status(signal_id, "error")
    
    async def _handle_signal(self, snapshot: MarketSnapshot, result: DetectionResult):
        """Handle a detected spike signal (technical analysis)."""
        market_name = snapshot.market.question[:50] if snapshot.market.question else snapshot.market.id
        market_id = snapshot.market.id
        
        self.logger.info(
            f"🎯 SIGNAL: {market_name} | "
            f"Direction: {result.signal.value} | "
            f"Confidence: {result.confidence:.0%}"
        )
        
        # Create trading signal from detection result
        trading_signal = self.detector.create_signal(result, snapshot)
        if not trading_signal:
            self.logger.debug("Signal creation failed (possibly in cooldown)")
            return
        
        # Create signal state for dashboard
        signal_id = f"sig_{datetime.utcnow().timestamp():.0f}"
        signal_state = SignalState(
            signal_id=signal_id,
            market_id=market_id,
            market_name=market_name,
            direction=result.signal.value,
            price=snapshot.price,
            confidence=result.confidence,
            detected_at=datetime.utcnow().isoformat(),
            trigger_reason=f"CUSUM: {result.cusum_result.message if result.cusum_result else 'N/A'}"
        )
        self.state_writer.add_signal(signal_state)
        
        # Send notification
        await self.notifier.notify_signal(
            market=market_name,
            direction=result.signal.value,
            price=snapshot.price,
            confidence=result.confidence,
            market_id=market_id
        )
        
        # Calculate position size from risk manager
        position_size = self.risk_manager.calculate_position_size(
            confidence=result.confidence
        )
        
        # Open the trade using executor
        try:
            execution_result = await self.executor.execute(trading_signal)
            
            if execution_result.status == ExecutionStatus.SUCCESS:
                # Record in position manager
                position = self.position_manager.open_position(
                    signal=trading_signal,
                    execution=execution_result,
                    market_name=market_name
                )
                
                # Record in risk manager
                self.risk_manager.record_trade_opened()
                
                # Update signal status
                self.state_writer.update_signal_status(signal_id, "traded")
                
                # Update state for dashboard
                pos_state = PositionState(
                    position_id=position.id,
                    market_id=market_id,
                    market_name=market_name,
                    side=trading_signal.direction.value,
                    entry_price=execution_result.executed_price,
                    current_price=snapshot.price,
                    size=execution_result.executed_size,
                    unrealized_pnl=0.0,
                    stop_loss=trading_signal.stop_loss,
                    take_profit=trading_signal.take_profit,
                    opened_at=datetime.utcnow().isoformat()
                )
                self.state_writer.update_position(pos_state)
                
                # Notify
                await self.notifier.notify_trade_open(
                    market=market_name,
                    side=trading_signal.direction.value,
                    price=execution_result.executed_price,
                    size=execution_result.executed_size,
                    market_id=market_id
                )
                
                self.logger.info(
                    f"✅ Trade opened: {trading_signal.direction.value} "
                    f"${execution_result.executed_size:.2f} @ ${execution_result.executed_price:.3f}"
                )
            else:
                self.logger.warning(f"Trade failed: {execution_result.error_message}")
                self.state_writer.update_signal_status(signal_id, "failed")
                
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")
            self.state_writer.update_signal_status(signal_id, "error")
    
    async def _monitor_positions(self):
        """Monitor open positions for exit conditions.
        
        This method continuously monitors all open positions:
        1. Fetches current market price
        2. Updates position's current_price for P&L calculation
        3. Checks stop-loss and take-profit conditions
        4. EXECUTES exit orders through the executor (crucial for live trading!)
        5. Updates dashboard state
        """
        positions = self.position_manager.open_positions
        
        for position in positions:
            try:
                # Get current price from scanner snapshot
                snapshot = await self.scanner.get_market_snapshot(position.market_id)
                if not snapshot:
                    continue
                
                current_price = snapshot.price
                
                # Update price in position manager (updates unrealized P&L)
                self.position_manager.update_price(position.token_id, current_price)
                
                # Check exit conditions MANUALLY and execute through executor
                # (Don't use check_exits() which auto-closes without executing orders!)
                exit_reason = None
                
                if position.should_stop_loss():
                    exit_reason = "stop_loss"
                    self.logger.info(
                        f"🛑 STOP LOSS triggered for {position.market_name}: "
                        f"Price {current_price:.4f} <= Stop {position.stop_loss:.4f}"
                    )
                elif position.should_take_profit():
                    exit_reason = "take_profit"
                    self.logger.info(
                        f"🎯 TAKE PROFIT triggered for {position.market_name}: "
                        f"Price {current_price:.4f} >= Target {position.take_profit:.4f}"
                    )
                
                # Execute exit if triggered
                if exit_reason:
                    # IMPORTANT: Use _close_position which executes through the executor!
                    # This places actual sell orders for live trading
                    await self._close_position(position, current_price, exit_reason)
                    continue  # Position closed, move to next
                
                # Position still open - update dashboard state
                pos_state = PositionState(
                    position_id=position.id,
                    market_id=position.market_id,
                    market_name=position.market_name,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    size=position.entry_size,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_percent,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    opened_at=position.entry_time.isoformat()
                )
                self.state_writer.update_position(pos_state)
                
                # Log position status periodically (every ~60 updates)
                if hasattr(self, '_position_log_counter'):
                    self._position_log_counter += 1
                else:
                    self._position_log_counter = 0
                
                if self._position_log_counter % 60 == 0:
                    pnl_sign = "+" if position.unrealized_pnl >= 0 else ""
                    self.logger.info(
                        f"📊 Position {position.market_name[:30]}: "
                        f"Entry ${position.entry_price:.4f} → Current ${current_price:.4f} | "
                        f"P&L: {pnl_sign}${position.unrealized_pnl:.2f} ({pnl_sign}{position.unrealized_pnl_percent:.1f}%)"
                    )
                    
            except Exception as e:
                self.logger.warning(f"Failed to monitor position {position.id}: {e}")
    
    async def _handle_closed_position(self, position):
        """Handle a position that was closed (stop-loss, take-profit, or manual).
        
        This is called AFTER the executor has successfully placed a sell order.
        Records the trade result in all systems and updates the dashboard.
        """
        pnl_sign = "+" if position.realized_pnl >= 0 else ""
        exit_emoji = "🟢" if position.realized_pnl >= 0 else "🔴"
        
        self.logger.info(
            f"{exit_emoji} Position {position.id} closed: {position.exit_reason} | "
            f"Entry ${position.entry_price:.4f} → Exit ${position.exit_price:.4f} | "
            f"P&L: {pnl_sign}${position.realized_pnl:.2f}"
        )
        
        # Record in risk manager (updates daily P&L, capital, etc.)
        self.risk_manager.record_trade_result(position.realized_pnl)
        
        # Record outcome in AI reasoning tracker (links trade result to AI decision)
        if self.ai_engine and position.id in self._ai_decision_map:
            reasoning_entry_id = self._ai_decision_map.pop(position.id)
            self.ai_engine.record_trade_outcome(
                reasoning_entry_id=reasoning_entry_id,
                pnl=position.realized_pnl,
                exit_price=position.exit_price,
                exit_reason=position.exit_reason
            )
        
        # Update state with full exit details for dashboard
        self.state_writer.close_position(
            position_id=position.id,
            exit_price=position.exit_price,
            exit_reason=position.exit_reason,
            realized_pnl=position.realized_pnl
        )
        
        # Add event to activity feed
        self.state_writer.add_event(
            "trade",
            f"{exit_emoji} Closed {position.market_name}: {pnl_sign}${position.realized_pnl:.2f} ({position.exit_reason})"
        )
        
        # Send notification (Telegram/Discord)
        await self.notifier.notify_trade_close(
            market=position.market_name,
            pnl=position.realized_pnl,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            exit_reason=position.exit_reason
        )
        
        self.logger.info(f"✅ Trade recorded: {pnl_sign}${position.realized_pnl:.2f}")
    
    async def _close_position(self, position, current_price: float, reason: str):
        """Manually close a position."""
        self.logger.info(f"Closing position {position.id}: {reason}")
        
        try:
            # Close position using executor
            execution_result = await self.executor.close_position(
                token_id=position.token_id,
                size=position.entry_size,
                current_price=current_price,
                reason=reason
            )
            
            if execution_result.status == ExecutionStatus.SUCCESS:
                # Close in position manager
                closed_position = self.position_manager.close_position(
                    position.id,
                    execution_result.executed_price,
                    reason
                )
                
                if closed_position:
                    await self._handle_closed_position(closed_position)
            else:
                self.logger.warning(f"Close order failed: {execution_result.error_message}")
                
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
    
    def _update_state(self):
        """Update the shared state file for the dashboard."""
        # Update risk state
        risk_summary = self.risk_manager.get_summary()
        risk_state = RiskState(
            status=risk_summary["status"],
            status_message=risk_summary["status_description"],
            can_trade=risk_summary["can_trade"],
            daily_pnl=risk_summary["daily_pnl"],
            daily_limit=risk_summary["daily_limit"],
            remaining_risk=risk_summary["remaining_daily_risk"],
            consecutive_losses=risk_summary["consecutive_losses"],
            circuit_breaker_active=risk_summary["circuit_breaker_active"],
            circuit_breaker_remaining=risk_summary["circuit_breaker_minutes_remaining"]
        )
        self.state_writer.update_risk(risk_state)
        
        # Update performance state
        daily = self.risk_manager.get_daily_stats()
        perf_state = PerformanceState(
            total_trades=self.risk_manager.state.total_trades,
            winning_trades=daily.wins,
            losing_trades=daily.losses,
            win_rate=risk_summary["daily_win_rate"],
            total_pnl=self.risk_manager.state.total_realized_pnl,
            daily_pnl=daily.realized_pnl,
            current_capital=self.risk_manager.state.current_capital,
            initial_capital=self.risk_manager.state.initial_capital,
            max_drawdown=daily.max_drawdown,
            best_trade=0.0,  # TODO: track this
            worst_trade=0.0  # TODO: track this
        )
        self.state_writer.update_performance(perf_state)
        
        # Update AI stats for dashboard
        if self.ai_enabled and self.ai_engine:
            self._update_ai_state()
        
        # Flush to disk
        self.state_writer.flush()
    
    def _update_ai_state(self):
        """Update AI-related state for the dashboard."""
        try:
            ai_stats_dict = self.ai_engine.get_stats()
            reasoning_stats = ai_stats_dict.get("reasoning_stats", {})
            bedrock_stats = ai_stats_dict.get("bedrock_stats", {})
            
            # Build AI stats state
            ai_stats = AIStatsState(
                model=self.config.ai.model,
                decisions_today=self._ai_analysis_count,
                avg_latency_ms=bedrock_stats.get("avg_tokens_per_request", 0) * 10,  # Rough estimate
                total_tokens=bedrock_stats.get("total_tokens", 0),
                avg_confidence=reasoning_stats.get("avg_confidence", 0),
                win_rate=reasoning_stats.get("win_rate", 0),
                profitable_trades=reasoning_stats.get("profitable_count", 0)
            )
            self.state_writer.update_ai_stats(ai_stats)
            
            # Get recent reasoning entries for dashboard - FULL reasoning
            # Use the AI engine's reasoning tracker if available (more reliable)
            try:
                if hasattr(self.ai_engine, 'reasoning_tracker') and self.ai_engine.reasoning_tracker:
                    # Use the engine's tracker directly (attribute name is reasoning_tracker, not _reasoning_tracker)
                    tracker = self.ai_engine.reasoning_tracker
                    self.logger.debug(f"Using AI engine's reasoning tracker: {tracker.log_dir}")
                else:
                    # Fallback to singleton
                    from src.ai.reasoning_tracker import get_reasoning_tracker
                    tracker = get_reasoning_tracker()
                    self.logger.debug(f"Using singleton reasoning tracker: {tracker.log_dir}")
                
                recent_entries = tracker.get_for_dashboard(limit=None)  # ALL entries
                self.logger.debug(f"Found {len(recent_entries)} AI reasoning entries")
                
                for entry in recent_entries:
                    # Calculate total tokens from input + output
                    input_tokens = entry.get("input_tokens", 0)
                    output_tokens = entry.get("output_tokens", 0)
                    total_tokens = input_tokens + output_tokens
                    
                    reasoning_entry = AIReasoningEntry(
                        action=entry.get("action", "HOLD"),
                        confidence=entry.get("confidence", "0%"),
                        market=entry.get("market", "Unknown"),
                        reasoning=entry.get("reasoning", ""),  # FULL reasoning
                        time=entry.get("time", ""),
                        outcome=entry.get("outcome", "pending"),
                        pnl=entry.get("pnl", "-"),
                        # Technical details from reasoning tracker
                        entry_price=entry.get("entry_price", 0.0),
                        stop_loss=entry.get("stop_loss", 0.0),
                        take_profit=entry.get("take_profit", 0.0),
                        position_size=entry.get("position_size", 0.0),
                        tokens_used=total_tokens,
                        latency_ms=entry.get("latency_ms", 0.0)
                    )
                    self.state_writer.add_ai_reasoning(reasoning_entry)
                    
            except Exception as e:
                self.logger.warning(f"Failed to update AI reasoning entries: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
            
            # Update Monte Carlo results if available
            try:
                mc_results = ai_stats_dict.get("monte_carlo", {})
                if mc_results:
                    mc_state = MonteCarloState(
                        prob_profit=mc_results.get("prob_profit", 0),
                        var_95=mc_results.get("var_95", 0),
                        risk_assessment=mc_results.get("risk_assessment", "Unknown"),
                        distribution=mc_results.get("distribution", [])[:100]  # Limit distribution size
                    )
                    self.state_writer.update_monte_carlo(mc_state)
            except Exception as e:
                self.logger.debug(f"Failed to update Monte Carlo: {e}")
                
        except Exception as e:
            self.logger.warning(f"Failed to update AI state: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logging.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


async def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Polybot AI Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in paper trading mode (no real trades)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI analysis (use technical only)"
    )
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Create and start bot
    bot = Polybot(
        config_path=args.config,
        dry_run=args.dry_run if args.dry_run else None
    )
    
    # Override AI if --no-ai flag
    if args.no_ai:
        bot.ai_enabled = False
    
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())