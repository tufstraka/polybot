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

from src.config.settings import load_config
from src.core.client import PolymarketClient
from src.core.scanner import MarketScanner
from src.core.tracker import PriceTracker, MarketSnapshot
from src.core.detector import SpikeDetector, DetectionResult
from src.core.executor import OrderExecutor
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
        self.config = load_config(config_path)
        
        # Override dry run if specified
        if dry_run is not None:
            self.config.trading.dry_run = dry_run
        
        self.dry_run = self.config.trading.dry_run
        
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
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        
        # File handler (if configured)
        handlers = [console]
        
        log_file = self.config.logging.file
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(level=log_level, handlers=handlers)
    
    async def _initialize_components(self):
        """Initialize all bot components."""
        self.logger.info("Initializing components...")
        
        # Polymarket client
        self.client = PolymarketClient(
            host=self.config.polymarket.api_url,
            key=self.config.polymarket.api_key,
            secret=self.config.polymarket.api_secret,
            passphrase=self.config.polymarket.passphrase,
            chain_id=self.config.polymarket.chain_id
        )
        
        # Market scanner
        self.scanner = MarketScanner(
            client=self.client,
            min_volume=self.config.detection.min_volume,
            max_markets=50  # Track up to 50 markets
        )
        
        # Price tracker
        self.tracker = PriceTracker(
            window_size=self.config.detection.lookback_window,
            ewma_span=self.config.detection.ewma_span
        )
        
        # Spike detector
        self.detector = SpikeDetector(
            cusum_threshold=self.config.detection.cusum_threshold,
            cusum_drift=self.config.detection.cusum_drift,
            ewma_span=self.config.detection.ewma_span,
            ewma_band_width=self.config.detection.ewma_band_width,
            roc_threshold=self.config.detection.roc_threshold,
            min_liquidity=self.config.detection.min_liquidity
        )
        
        # Order executor
        self.executor = OrderExecutor(
            client=self.client,
            dry_run=self.dry_run,
            default_slippage=0.02  # 2% slippage tolerance
        )
        
        # Position manager
        self.position_manager = PositionManager(
            state_file="data/positions.json"
        )
        
        # Risk manager
        self.risk_manager = RiskManager(
            initial_capital=self.config.money.initial_capital,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_position_size=self.config.risk.max_position_size,
            min_position_size=self.config.risk.min_position_size,
            max_consecutive_losses=self.config.risk.max_consecutive_losses,
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
        if self.config.notifications.telegram_enabled:
            telegram = TelegramNotifier(
                bot_token=self.config.notifications.telegram_token,
                chat_id=self.config.notifications.telegram_chat_id,
                enabled=self.config.notifications.telegram_enabled
            )
            self.notifier.add_notifier(telegram)
        
        # Add Discord notifier if configured
        if self.config.notifications.discord_enabled:
            discord = DiscordNotifier(
                webhook_url=self.config.notifications.discord_webhook,
                enabled=self.config.notifications.discord_enabled
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
                    "capital": self.config.money.initial_capital,
                    "daily_limit": self.config.risk.max_daily_loss
                }
            )
            
            self.logger.info("Bot started successfully")
            
            # Run main loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.logger.error(traceback.format_exc())
            self.state_writer.record_error(str(e))
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
        
        poll_interval = self.config.polling.interval
        market_refresh_interval = self.config.polling.market_refresh
        
        last_market_refresh = 0
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
            markets = await self.scanner.get_active_markets()
            
            # Update tracked markets
            for market in markets:
                market_id = market.get("condition_id") or market.get("id")
                if market_id:
                    self._tracked_markets[market_id] = market
            
            self.logger.info(f"Tracking {len(self._tracked_markets)} markets")
            self.state_writer.add_event(
                "system", 
                f"Refreshed markets: {len(self._tracked_markets)} active"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to refresh markets: {e}")
    
    async def _update_prices(self):
        """Update prices for all tracked markets."""
        for market_id, market in list(self._tracked_markets.items()):
            try:
                # Get latest price
                price_data = await self.client.get_market_price(market_id)
                
                if not price_data:
                    continue
                
                current_price = price_data.get("price", 0.5)
                volume = price_data.get("volume", 0)
                
                # Update tracker
                indicators = self.tracker.update(market_id, current_price)
                
                # Get orderbook for liquidity check
                orderbook = await self.client.get_orderbook(market_id)
                
                # Create snapshot
                snapshot = MarketSnapshot(
                    market_id=market_id,
                    name=market.get("question", market_id)[:50],
                    price=current_price,
                    volume_24h=volume,
                    bid_price=orderbook.get("best_bid", current_price - 0.01),
                    ask_price=orderbook.get("best_ask", current_price + 0.01),
                    bid_size=orderbook.get("bid_size", 100),
                    ask_size=orderbook.get("ask_size", 100),
                    timestamp=datetime.utcnow()
                )
                
                # Update state
                if indicators:
                    market_state = MarketState(
                        market_id=market_id,
                        name=snapshot.name,
                        current_price=current_price,
                        ewma_price=indicators.ewma_price,
                        ewma_upper=indicators.ewma_upper,
                        ewma_lower=indicators.ewma_lower,
                        roc=indicators.roc,
                        cusum_pos=indicators.cusum_pos,
                        cusum_neg=indicators.cusum_neg,
                        volume_24h=volume,
                        last_updated=datetime.utcnow().isoformat()
                    )
                    self.state_writer.update_market(market_state)
                
            except Exception as e:
                self.logger.warning(f"Failed to update price for {market_id}: {e}")
    
    async def _check_signals(self):
        """Check for spike signals and potentially open trades."""
        # Skip if we can't trade
        if not self.risk_manager.can_open_trade():
            return
        
        # Check each tracked market
        for market_id, market in list(self._tracked_markets.items()):
            try:
                # Skip if we already have a position
                if self.position_manager.has_position(market_id):
                    continue
                
                # Get latest snapshot
                history = self.tracker.get_history(market_id)
                if not history or len(history) < 10:
                    continue
                
                latest = history[-1]
                
                # Get orderbook for the check
                orderbook = await self.client.get_orderbook(market_id)
                
                snapshot = MarketSnapshot(
                    market_id=market_id,
                    name=market.get("question", market_id)[:50],
                    price=latest.price,
                    volume_24h=market.get("volume", 0),
                    bid_price=orderbook.get("best_bid", latest.price - 0.01),
                    ask_price=orderbook.get("best_ask", latest.price + 0.01),
                    bid_size=orderbook.get("bid_size", 100),
                    ask_size=orderbook.get("ask_size", 100),
                    timestamp=datetime.utcnow()
                )
                
                # Check for spike
                result = self.detector.check(snapshot)
                
                if result.is_spike:
                    await self._handle_signal(snapshot, result)
                    
            except Exception as e:
                self.logger.warning(f"Signal check failed for {market_id}: {e}")
    
    async def _handle_signal(self, snapshot: MarketSnapshot, result: DetectionResult):
        """Handle a detected spike signal."""
        self.logger.info(
            f"🎯 SIGNAL: {snapshot.name} | "
            f"Direction: {result.direction} | "
            f"Confidence: {result.confidence:.0%}"
        )
        
        # Create signal state
        signal_id = f"sig_{datetime.utcnow().timestamp():.0f}"
        signal = SignalState(
            signal_id=signal_id,
            market_id=snapshot.market_id,
            market_name=snapshot.name,
            direction=result.direction,
            price=snapshot.price,
            confidence=result.confidence,
            detected_at=datetime.utcnow().isoformat(),
            trigger_reason=result.trigger_reason
        )
        self.state_writer.add_signal(signal)
        
        # Send notification
        await self.notifier.notify_signal(
            market=snapshot.name,
            direction=result.direction,
            price=snapshot.price,
            confidence=result.confidence,
            market_id=snapshot.market_id
        )
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            confidence=result.confidence
        )
        
        # Determine trade side (mean reversion: opposite of spike)
        # If price spiked UP, we SELL (expecting it to come down)
        # If price spiked DOWN, we BUY (expecting it to come up)
        trade_side = "SELL" if result.direction == "up" else "BUY"
        
        # Open the trade
        try:
            order_result = await self.executor.execute(
                market_id=snapshot.market_id,
                side=trade_side,
                size=position_size,
                price=snapshot.price
            )
            
            if order_result.success:
                # Record in position manager
                position = self.position_manager.open_position(
                    market_id=snapshot.market_id,
                    market_name=snapshot.name,
                    side=trade_side,
                    entry_price=order_result.executed_price,
                    size=position_size,
                    stop_loss=result.recommended_stop,
                    take_profit=result.recommended_target
                )
                
                # Record in risk manager
                self.risk_manager.record_trade_opened()
                
                # Update signal status
                self.state_writer.update_signal_status(signal_id, "traded")
                
                # Update state
                pos_state = PositionState(
                    position_id=position.position_id,
                    market_id=snapshot.market_id,
                    market_name=snapshot.name,
                    side=trade_side,
                    entry_price=order_result.executed_price,
                    current_price=snapshot.price,
                    size=position_size,
                    unrealized_pnl=0.0,
                    stop_loss=result.recommended_stop,
                    take_profit=result.recommended_target,
                    opened_at=datetime.utcnow().isoformat()
                )
                self.state_writer.update_position(pos_state)
                
                # Notify
                await self.notifier.notify_trade_open(
                    market=snapshot.name,
                    side=trade_side,
                    price=order_result.executed_price,
                    size=position_size,
                    market_id=snapshot.market_id
                )
                
                self.logger.info(
                    f"✅ Trade opened: {trade_side} ${position_size:.2f} @ ${order_result.executed_price:.3f}"
                )
            else:
                self.logger.warning(f"Trade failed: {order_result.error}")
                self.state_writer.update_signal_status(signal_id, "failed")
                
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")
            self.state_writer.update_signal_status(signal_id, "error")
    
    async def _monitor_positions(self):
        """Monitor open positions for exit conditions."""
        positions = self.position_manager.get_open_positions()
        
        for position in positions:
            try:
                # Get current price
                price_data = await self.client.get_market_price(position.market_id)
                if not price_data:
                    continue
                
                current_price = price_data.get("price", position.entry_price)
                
                # Check for exit
                exit_result = self.position_manager.check_exit(
                    position.position_id,
                    current_price
                )
                
                if exit_result.should_exit:
                    await self._close_position(position, current_price, exit_result.reason)
                else:
                    # Update position state
                    pnl = self.position_manager.calculate_pnl(
                        position.position_id, 
                        current_price
                    )
                    
                    pos_state = PositionState(
                        position_id=position.position_id,
                        market_id=position.market_id,
                        market_name=position.market_name,
                        side=position.side,
                        entry_price=position.entry_price,
                        current_price=current_price,
                        size=position.size,
                        unrealized_pnl=pnl,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        opened_at=position.opened_at.isoformat()
                    )
                    self.state_writer.update_position(pos_state)
                    
            except Exception as e:
                self.logger.warning(f"Failed to monitor position {position.position_id}: {e}")
    
    async def _close_position(self, position, current_price: float, reason: str):
        """Close a position."""
        self.logger.info(f"Closing position {position.position_id}: {reason}")
        
        try:
            # Execute closing order
            close_side = "SELL" if position.side == "BUY" else "BUY"
            
            order_result = await self.executor.execute(
                market_id=position.market_id,
                side=close_side,
                size=position.size,
                price=current_price
            )
            
            if order_result.success:
                # Calculate P&L
                pnl = self.position_manager.close_position(
                    position.position_id,
                    order_result.executed_price,
                    reason
                )
                
                # Record in risk manager
                self.risk_manager.record_trade_result(pnl)
                
                # Update state
                self.state_writer.close_position(position.position_id)
                self.state_writer.add_event(
                    "trade",
                    f"Closed {position.market_name}: ${pnl:+.2f} ({reason})"
                )
                
                # Notify
                await self.notifier.notify_trade_close(
                    market=position.market_name,
                    pnl=pnl,
                    entry_price=position.entry_price,
                    exit_price=order_result.executed_price,
                    exit_reason=reason
                )
                
                self.logger.info(f"✅ Position closed: ${pnl:+.2f}")
            else:
                self.logger.warning(f"Close order failed: {order_result.error}")
                
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