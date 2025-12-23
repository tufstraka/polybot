"""
Polybot - Main Bot Orchestrator.

Plain English Explanation:
==========================

This is the "brain" of the bot. It coordinates all the other parts:

1. STARTUP:
   - Loads your configuration
   - Connects to Polymarket
   - Sets up notifications (Telegram/Discord)
   - Initializes risk management

2. MAIN LOOP (runs every second):
   - Scans for active markets
   - Gets latest prices
   - Checks for spike signals
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
from typing import Optional, Dict
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
    RiskState, PerformanceState
)
from src.risk.risk_manager import RiskManager, RiskStatus
from src.notifications.base import NotificationManager
from src.notifications.telegram import TelegramNotifier
from src.notifications.discord import DiscordNotifier


class Polybot:
    """
    Main bot orchestrator.
    
    Coordinates all modules to run the spike hunting strategy.
    
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
        
        # Initialize logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Polybot (dry_run={self.dry_run})")
        
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
            self.state_writer.add_event("system", "Bot started")
            self.state_writer.flush(force=True)
            
            # Send startup notification
            await self.notifier.notify_status(
                f"🚀 Polybot started ({'DRY RUN' if self.dry_run else 'LIVE'})",
                {
                    "capital": self.config.money.starting_balance,
                    "daily_limit": self.config.money.max_daily_loss
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
            self.state_writer.reset_daily_counters()
            self.state_writer.add_event("system", f"New trading day: {today}")
    
    async def _refresh_markets(self):
        """Refresh the list of markets to track."""
        self.logger.debug("Refreshing market list...")
        
        try:
            # Use scanner to refresh and get tradable markets
            await self.scanner.refresh_markets()
            markets = await self.scanner.get_tradable_markets(limit=50)
            
            # Update tracked markets
            self._tracked_markets.clear()
            for market in markets:
                self._tracked_markets[market.id] = market
            
            self.logger.info(f"Tracking {len(self._tracked_markets)} markets")
            self.state_writer.add_event(
                "system",
                f"Refreshed markets: {len(self._tracked_markets)} active"
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
        """Check for spike signals and potentially open trades."""
        # Skip if we can't trade
        if not self.risk_manager.can_open_trade():
            return
        
        # Check each tracked market
        for market_id, market in list(self._tracked_markets.items()):
            try:
                # Skip if we already have a position
                if self.position_manager.has_position_for_market(market_id):
                    continue
                
                # Get history to check if we have enough data
                history = self.tracker.get_history(market_id)
                if not history or not history.has_enough_data:
                    continue
                
                # Get snapshot with orderbook
                snapshot = await self.scanner.get_market_snapshot(market_id)
                if not snapshot or not snapshot.is_tradable:
                    continue
                
                # Check for spike using detector
                result = self.detector.check(snapshot)
                
                # Check if signal is valid (all layers passed)
                if result.is_signal:
                    await self._handle_signal(snapshot, result)
                    
            except Exception as e:
                self.logger.warning(f"Signal check failed for {market_id}: {e}")
    
    async def _handle_signal(self, snapshot: MarketSnapshot, result: DetectionResult):
        """Handle a detected spike signal."""
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
        """Monitor open positions for exit conditions."""
        positions = self.position_manager.open_positions
        
        for position in positions:
            try:
                # Get current price from scanner snapshot
                snapshot = await self.scanner.get_market_snapshot(position.market_id)
                if not snapshot:
                    continue
                
                current_price = snapshot.price
                
                # Update price in position manager
                self.position_manager.update_price(position.token_id, current_price)
                
                # Check for exit conditions (stop loss / take profit)
                exits = self.position_manager.check_exits()
                
                # Handle any exits that were triggered
                for closed_position in exits:
                    if closed_position.id == position.id:
                        await self._handle_closed_position(closed_position)
                
                # If position still open, update dashboard state
                if position.is_open:
                    pos_state = PositionState(
                        position_id=position.id,
                        market_id=position.market_id,
                        market_name=position.market_name,
                        side=position.side.value,
                        entry_price=position.entry_price,
                        current_price=current_price,
                        size=position.entry_size,
                        unrealized_pnl=position.unrealized_pnl,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        opened_at=position.entry_time.isoformat()
                    )
                    self.state_writer.update_position(pos_state)
                    
            except Exception as e:
                self.logger.warning(f"Failed to monitor position {position.id}: {e}")
    
    async def _handle_closed_position(self, position):
        """Handle a position that was closed by check_exits."""
        self.logger.info(f"Position {position.id} closed: {position.exit_reason}")
        
        # Record in risk manager
        self.risk_manager.record_trade_result(position.realized_pnl)
        
        # Update state
        self.state_writer.close_position(position.id)
        self.state_writer.add_event(
            "trade",
            f"Closed {position.market_name}: ${position.realized_pnl:+.2f} ({position.exit_reason})"
        )
        
        # Notify
        await self.notifier.notify_trade_close(
            market=position.market_name,
            pnl=position.realized_pnl,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            exit_reason=position.exit_reason
        )
        
        self.logger.info(f"✅ Position closed: ${position.realized_pnl:+.2f}")
    
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
        
        # Flush to disk
        self.state_writer.flush()


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logging.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


async def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Polybot Spike Hunter")
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
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Create and start bot
    bot = Polybot(
        config_path=args.config,
        dry_run=args.dry_run if args.dry_run else None
    )
    
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())