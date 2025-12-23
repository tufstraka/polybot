"""
Market Scanner Module
=====================

Discovers and filters active Polymarket markets based on
volume, liquidity, and other criteria.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging

from src.config.settings import get_settings, Settings
from src.core.client import PolymarketClient, Market, Orderbook


logger = logging.getLogger(__name__)


@dataclass
class MarketScore:
    """Scoring for market quality/tradability."""
    market: Market
    volume_score: float = 0.0      # Based on 24h volume
    liquidity_score: float = 0.0   # Based on orderbook depth
    spread_score: float = 0.0      # Lower spread = higher score
    activity_score: float = 0.0    # Recent trading activity
    
    @property
    def total_score(self) -> float:
        """Combined score for ranking."""
        return (
            self.volume_score * 0.3 +
            self.liquidity_score * 0.3 +
            self.spread_score * 0.2 +
            self.activity_score * 0.2
        )


@dataclass
class MarketSnapshot:
    """
    Point-in-time snapshot of a market with orderbook data.
    
    This is what we pass to the price tracker and detector.
    """
    market: Market
    orderbook: Optional[Orderbook]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def price(self) -> float:
        """Get current price (midpoint or YES price)."""
        if self.orderbook and self.orderbook.midpoint:
            return self.orderbook.midpoint
        return self.market.yes_price
    
    @property
    def spread(self) -> Optional[float]:
        """Get current spread."""
        return self.orderbook.spread if self.orderbook else None
    
    @property
    def spread_percent(self) -> Optional[float]:
        """Get spread as percentage."""
        return self.orderbook.spread_percent if self.orderbook else None
    
    @property
    def bid_depth(self) -> float:
        """Get total bid depth."""
        return self.orderbook.bid_depth if self.orderbook else 0.0
    
    @property
    def ask_depth(self) -> float:
        """Get total ask depth."""
        return self.orderbook.ask_depth if self.orderbook else 0.0
    
    @property
    def is_tradable(self) -> bool:
        """Check if market meets minimum requirements for trading."""
        settings = get_settings()
        
        # Must have orderbook data
        if not self.orderbook:
            return False
        
        # Must have reasonable spread
        if self.spread_percent and self.spread_percent > settings.filters.max_spread_percent:
            return False
        
        # Must have minimum liquidity
        min_depth = min(self.bid_depth, self.ask_depth)
        if min_depth < settings.filters.min_liquidity_usd:
            return False
        
        return True


class MarketScanner:
    """
    Scans and filters Polymarket markets.
    
    Responsibilities:
    - Fetching active markets from Polymarket
    - Filtering by volume, liquidity, spread
    - Ranking markets by trading potential
    - Caching market data to reduce API calls
    
    Usage:
        scanner = MarketScanner(client)
        await scanner.refresh_markets()
        tradable_markets = await scanner.get_tradable_markets()
    """
    
    def __init__(
        self,
        client: PolymarketClient,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the scanner.
        
        Args:
            client: Initialized PolymarketClient
            settings: Optional Settings object
        """
        self.client = client
        self.settings = settings or get_settings()
        
        # Market cache
        self._markets: Dict[str, Market] = {}
        self._market_scores: Dict[str, MarketScore] = {}
        self._last_refresh: Optional[datetime] = None
        
        # Blacklist markets we don't want to trade
        self._blacklist: Set[str] = set()
    
    async def refresh_markets(self, force: bool = False) -> int:
        """
        Refresh the list of markets from Polymarket.
        
        Args:
            force: If True, refresh even if cache is fresh
            
        Returns:
            Number of markets found
        """
        # Check if cache is still fresh
        if not force and self._last_refresh:
            age = datetime.now() - self._last_refresh
            if age.total_seconds() < self.settings.polling.market_refresh_seconds:
                return len(self._markets)
        
        try:
            # Fetch markets from API
            markets = await self.client.get_markets(limit=200)
            
            # Update cache
            self._markets = {m.id: m for m in markets}
            self._last_refresh = datetime.now()
            
            # Score markets
            await self._score_markets()
            
            logger.info(f"Refreshed {len(self._markets)} markets")
            return len(self._markets)
            
        except Exception as e:
            logger.error(f"Failed to refresh markets: {e}")
            return len(self._markets)
    
    async def _score_markets(self):
        """Score all markets for tradability ranking."""
        # Find max values for normalization
        max_volume = max((m.volume_24h for m in self._markets.values()), default=1)
        max_liquidity = max((m.liquidity for m in self._markets.values()), default=1)
        
        for market_id, market in self._markets.items():
            # Skip blacklisted markets
            if market_id in self._blacklist:
                continue
            
            # Calculate scores (0-1)
            volume_score = market.volume_24h / max_volume if max_volume > 0 else 0
            liquidity_score = market.liquidity / max_liquidity if max_liquidity > 0 else 0
            
            # Spread score will be updated when we get orderbook
            self._market_scores[market_id] = MarketScore(
                market=market,
                volume_score=volume_score,
                liquidity_score=liquidity_score,
                spread_score=0.5,  # Default until we get orderbook
                activity_score=volume_score,  # Use volume as proxy for activity
            )
    
    def filter_markets(self, markets: List[Market]) -> List[Market]:
        """
        Filter markets based on configured criteria.
        
        Args:
            markets: List of markets to filter
            
        Returns:
            Filtered list of markets
        """
        filtered = []
        
        for market in markets:
            # Skip blacklisted
            if market.id in self._blacklist:
                continue
            
            # Check minimum volume
            if market.volume_24h < self.settings.filters.min_daily_volume:
                continue
            
            # Check if market is active
            if not market.is_active:
                continue
            
            # Check if we have token IDs
            if not market.yes_token_id:
                continue
            
            filtered.append(market)
        
        return filtered
    
    async def get_tradable_markets(self, limit: int = 20) -> List[Market]:
        """
        Get top tradable markets sorted by score.
        
        Args:
            limit: Maximum number of markets to return
            
        Returns:
            List of Market objects, sorted by quality score
        """
        # Ensure markets are loaded
        if not self._markets:
            await self.refresh_markets()
        
        # Filter markets
        filtered = self.filter_markets(list(self._markets.values()))
        
        # Sort by score
        scored = [
            (m, self._market_scores.get(m.id, MarketScore(market=m)))
            for m in filtered
        ]
        scored.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Return top markets
        return [m for m, _ in scored[:limit]]
    
    async def get_market_snapshot(self, market_id: str) -> Optional[MarketSnapshot]:
        """
        Get a full snapshot of a market including orderbook.
        
        Args:
            market_id: The market's condition ID
            
        Returns:
            MarketSnapshot or None if market not found
        """
        market = self._markets.get(market_id)
        if not market:
            return None
        
        # Fetch orderbook
        orderbook = None
        if market.yes_token_id:
            orderbook = await self.client.get_orderbook(market.yes_token_id)
            
            # Update spread score
            if orderbook and orderbook.spread_percent:
                # Lower spread = higher score (invert and normalize)
                spread_score = max(0, 1 - (orderbook.spread_percent / 10))
                if market_id in self._market_scores:
                    self._market_scores[market_id].spread_score = spread_score
        
        return MarketSnapshot(
            market=market,
            orderbook=orderbook,
        )
    
    async def get_snapshots(self, market_ids: List[str]) -> Dict[str, MarketSnapshot]:
        """
        Get snapshots for multiple markets.
        
        Args:
            market_ids: List of market condition IDs
            
        Returns:
            Dictionary mapping market_id to MarketSnapshot
        """
        snapshots = {}
        for market_id in market_ids:
            snapshot = await self.get_market_snapshot(market_id)
            if snapshot:
                snapshots[market_id] = snapshot
        return snapshots
    
    async def get_all_tradable_snapshots(self, limit: int = 20) -> List[MarketSnapshot]:
        """
        Get snapshots for all tradable markets.
        
        This is the main method called by the main loop to get
        current state of all markets we're watching.
        
        Args:
            limit: Maximum number of markets
            
        Returns:
            List of MarketSnapshot objects
        """
        # Get tradable markets
        markets = await self.get_tradable_markets(limit)
        
        # Get snapshots for each
        snapshots = []
        for market in markets:
            snapshot = await self.get_market_snapshot(market.id)
            if snapshot and snapshot.is_tradable:
                snapshots.append(snapshot)
        
        return snapshots
    
    def blacklist_market(self, market_id: str):
        """
        Add a market to the blacklist (won't be traded).
        
        Args:
            market_id: The market to blacklist
        """
        self._blacklist.add(market_id)
        logger.info(f"Blacklisted market: {market_id}")
    
    def unblacklist_market(self, market_id: str):
        """Remove a market from the blacklist."""
        self._blacklist.discard(market_id)
    
    def get_market(self, market_id: str) -> Optional[Market]:
        """Get a market by ID from cache."""
        return self._markets.get(market_id)
    
    def get_all_markets(self) -> List[Market]:
        """Get all cached markets."""
        return list(self._markets.values())
    
    @property
    def market_count(self) -> int:
        """Number of markets in cache."""
        return len(self._markets)
    
    @property
    def last_refresh_age(self) -> Optional[timedelta]:
        """Time since last market refresh."""
        if self._last_refresh:
            return datetime.now() - self._last_refresh
        return None


if __name__ == "__main__":
    # Test the scanner
    async def test():
        from src.core.client import create_client
        
        client = await create_client()
        scanner = MarketScanner(client)
        
        print("Testing Market Scanner...")
        print("=" * 60)
        
        # Refresh markets
        count = await scanner.refresh_markets()
        print(f"\nFound {count} markets")
        
        # Get tradable markets
        tradable = await scanner.get_tradable_markets(limit=10)
        print(f"\nTop {len(tradable)} tradable markets:")
        
        for i, market in enumerate(tradable, 1):
            score = scanner._market_scores.get(market.id)
            print(f"\n{i}. {market.question[:60]}...")
            print(f"   Volume: ${market.volume_24h:,.0f}")
            print(f"   YES Price: {market.yes_price:.2f}")
            if score:
                print(f"   Score: {score.total_score:.2f}")
        
        # Get a snapshot
        if tradable:
            print(f"\n\nGetting snapshot for first market...")
            snapshot = await scanner.get_market_snapshot(tradable[0].id)
            if snapshot:
                print(f"  Price: {snapshot.price:.3f}")
                print(f"  Spread: {snapshot.spread_percent:.2f}%" if snapshot.spread_percent else "  Spread: N/A")
                print(f"  Bid Depth: ${snapshot.bid_depth:,.2f}")
                print(f"  Ask Depth: ${snapshot.ask_depth:,.2f}")
                print(f"  Tradable: {snapshot.is_tradable}")
        
        client.close()
    
    asyncio.run(test())