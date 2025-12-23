"""
Polymarket API Client Wrapper
=============================

Wraps the py-clob-client library to provide a clean interface
for interacting with Polymarket's CLOB (Central Limit Order Book).
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    OrderType,
    MarketOrderArgs,
    BalanceAllowanceParams,
    OpenOrderParams,
)
from py_clob_client.constants import POLYGON

from src.config.settings import get_settings


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


class Side(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Market:
    """Represents a Polymarket prediction market."""
    id: str  # condition_id
    question: str
    description: str
    end_date: Optional[datetime]
    volume_24h: float
    liquidity: float
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    
    # Token details (usually YES and NO)
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    yes_price: float = 0.5
    no_price: float = 0.5
    
    # Trading status from API
    accepting_orders: bool = False
    closed: bool = True
    
    @property
    def is_active(self) -> bool:
        """Check if market is still active and tradable."""
        # Use the accepting_orders flag from API - this is the definitive indicator
        return self.accepting_orders and not self.closed
    
    def __str__(self) -> str:
        return f"Market({self.question[:50]}... @ YES={self.yes_price:.2f})"


@dataclass
class OrderbookLevel:
    """Single level in the orderbook."""
    price: float
    size: float


@dataclass
class Orderbook:
    """Market orderbook with bids and asks."""
    market_id: str
    token_id: str
    bids: List[OrderbookLevel] = field(default_factory=list)  # Buy orders
    asks: List[OrderbookLevel] = field(default_factory=list)  # Sell orders
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid(self) -> Optional[float]:
        """Highest buy price."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Lowest sell price."""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Spread between best bid and ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_percent(self) -> Optional[float]:
        """Spread as percentage of midpoint."""
        if self.spread and self.midpoint:
            return (self.spread / self.midpoint) * 100
        return None
    
    @property
    def midpoint(self) -> Optional[float]:
        """Midpoint price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def bid_depth(self) -> float:
        """Total value of bids."""
        return sum(level.price * level.size for level in self.bids)
    
    @property
    def ask_depth(self) -> float:
        """Total value of asks."""
        return sum(level.price * level.size for level in self.asks)


@dataclass
class Order:
    """Represents an order on Polymarket."""
    id: str
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    status: OrderStatus
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_size: float = 0.0
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_open(self) -> bool:
        return self.status == OrderStatus.OPEN


@dataclass
class Position:
    """Represents a position in a market."""
    market_id: str
    token_id: str
    size: float
    avg_price: float
    side: Side
    current_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        if self.side == Side.BUY:
            return (self.current_price - self.avg_price) * self.size
        else:
            return (self.avg_price - self.current_price) * self.size
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P/L as percentage."""
        cost = self.avg_price * self.size
        if cost > 0:
            return (self.unrealized_pnl / cost) * 100
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket Client
# ═══════════════════════════════════════════════════════════════════════════════


class PolymarketClient:
    """
    Wrapper around py-clob-client for Polymarket interactions.
    
    Handles:
    - Authentication with API keys
    - Fetching markets and orderbooks
    - Placing and cancelling orders
    - Getting account balances
    
    Usage:
        client = PolymarketClient()
        await client.initialize()
        markets = await client.get_markets()
    """
    
    def __init__(self, settings=None):
        """
        Initialize the client.
        
        Args:
            settings: Optional Settings object. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self._client: Optional[ClobClient] = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the CLOB client with credentials.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Get credentials from settings
            api_key = self.settings.env.polymarket_api_key
            api_secret = self.settings.env.polymarket_api_secret
            passphrase = self.settings.env.polymarket_passphrase
            private_key = self.settings.env.polymarket_private_key
            
            if not private_key:
                logger.warning("No private key configured - running in read-only mode")
                # Create client without authentication for read-only access
                self._client = ClobClient(
                    host="https://clob.polymarket.com",
                    chain_id=POLYGON,
                )
            else:
                # Create account from private key
                # Ensure private key has 0x prefix
                if not private_key.startswith("0x"):
                    private_key = f"0x{private_key}"
                
                self._client = ClobClient(
                    host="https://clob.polymarket.com",
                    chain_id=POLYGON,
                    key=private_key,
                    creds={
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "api_passphrase": passphrase,
                    } if api_key else None,
                )
            
            self._initialized = True
            logger.info("Polymarket client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Polymarket client: {e}")
            return False
    
    def _ensure_initialized(self):
        """Ensure client is initialized before use."""
        if not self._initialized or not self._client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Market Data
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_markets(self, limit: int = 100, active_only: bool = True) -> List[Market]:
        """
        Fetch markets from Polymarket using Gamma API for better market discovery.
        
        The Gamma API provides volume, liquidity, and other trading data that
        the CLOB API doesn't include.
        
        Args:
            limit: Maximum number of markets to fetch
            active_only: If True, only return markets that are accepting orders
            
        Returns:
            List of Market objects
        """
        self._ensure_initialized()
        
        try:
            import httpx
            
            # Use Gamma API for market discovery (has volume, liquidity, etc.)
            params = {'limit': limit}
            if active_only:
                params['active'] = 'true'
                params['closed'] = 'false'
            
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(
                    'https://gamma-api.polymarket.com/markets',
                    params=params
                )
                
                if response.status_code != 200:
                    logger.error(f"Gamma API error: {response.status_code}")
                    return []
                
                market_list = response.json()
            
            # Parse markets
            markets = []
            for market_data in market_list:
                market = self._parse_gamma_market(market_data)
                if market:
                    markets.append(market)
            
            logger.info(f"Fetched {len(markets)} markets from Gamma API")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []
    
    def _parse_gamma_market(self, data: Dict[str, Any]) -> Optional[Market]:
        """Parse market data from Gamma API response."""
        try:
            # Get token IDs from clobTokenIds
            clob_token_ids = data.get("clobTokenIds", [])
            
            # Handle case where clobTokenIds might be a string
            if isinstance(clob_token_ids, str):
                import json
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except:
                    clob_token_ids = []
            
            yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
            no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None
            
            # Skip markets without token IDs
            if not yes_token_id:
                return None
            
            # Parse outcome prices - can be array or use bestBid/bestAsk
            yes_price = 0.5
            no_price = 0.5
            
            # Try bestBid/bestAsk first (more accurate)
            if data.get("bestBid") is not None and data.get("bestAsk") is not None:
                try:
                    bid = float(data["bestBid"])
                    ask = float(data["bestAsk"])
                    yes_price = (bid + ask) / 2
                    no_price = 1 - yes_price
                except:
                    pass
            else:
                # Fall back to outcomePrices
                outcome_prices = data.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    import json
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except:
                        outcome_prices = []
                
                if outcome_prices and len(outcome_prices) > 0:
                    try:
                        yes_price = float(outcome_prices[0])
                        no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else (1 - yes_price)
                    except:
                        pass
            
            # Parse end date
            end_date = None
            if data.get("endDate"):
                try:
                    end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
                except:
                    pass
            
            return Market(
                id=data.get("conditionId", ""),
                question=data.get("question", "Unknown"),
                description=data.get("description", "")[:500] if data.get("description") else "",
                end_date=end_date,
                volume_24h=float(data.get("volume24hr", 0) or 0),
                liquidity=float(data.get("liquidityNum", 0) or data.get("liquidity", 0) or 0),
                tokens=[],  # Gamma API doesn't include full token data
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_price=yes_price,
                no_price=no_price,
                accepting_orders=bool(data.get("acceptingOrders", False)),
                closed=bool(data.get("closed", True)),
            )
        except Exception as e:
            logger.error(f"Failed to parse Gamma market data: {e}")
            return None
    
    async def get_market(self, condition_id: str) -> Optional[Market]:
        """
        Fetch a specific market by condition ID.
        
        Args:
            condition_id: The market's condition ID
            
        Returns:
            Market object or None if not found
        """
        self._ensure_initialized()
        
        try:
            response = self._client.get_market(condition_id)
            return self._parse_market(response) if response else None
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None
    
    def _parse_market(self, data: Dict[str, Any]) -> Optional[Market]:
        """Parse market data from API response."""
        try:
            tokens = data.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
            no_token = next((t for t in tokens if t.get("outcome") == "No"), None)
            
            # Parse end date
            end_date = None
            if data.get("end_date_iso"):
                try:
                    end_date = datetime.fromisoformat(data["end_date_iso"].replace("Z", "+00:00"))
                except:
                    pass
            
            return Market(
                id=data.get("condition_id", ""),
                question=data.get("question", "Unknown"),
                description=data.get("description", ""),
                end_date=end_date,
                volume_24h=float(data.get("volume_num_24hr", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                tokens=tokens,
                yes_token_id=yes_token.get("token_id") if yes_token else None,
                no_token_id=no_token.get("token_id") if no_token else None,
                yes_price=float(yes_token.get("price", 0.5)) if yes_token else 0.5,
                no_price=float(no_token.get("price", 0.5)) if no_token else 0.5,
                accepting_orders=bool(data.get("accepting_orders", False)),
                closed=bool(data.get("closed", True)),
            )
        except Exception as e:
            logger.error(f"Failed to parse market data: {e}")
            return None
    
    async def get_orderbook(self, token_id: str) -> Optional[Orderbook]:
        """
        Fetch orderbook for a specific token.
        
        Args:
            token_id: The token ID to get orderbook for
            
        Returns:
            Orderbook object or None if failed
        """
        self._ensure_initialized()
        
        try:
            response = self._client.get_order_book(token_id)
            
            # Handle OrderBookSummary object (py-clob-client returns objects, not dicts)
            if hasattr(response, 'bids') and hasattr(response, 'asks'):
                # It's an OrderBookSummary object
                bids = [
                    OrderbookLevel(
                        price=float(b.price) if hasattr(b, 'price') else float(b.get("price", 0)),
                        size=float(b.size) if hasattr(b, 'size') else float(b.get("size", 0))
                    )
                    for b in (response.bids or [])
                ]
                asks = [
                    OrderbookLevel(
                        price=float(a.price) if hasattr(a, 'price') else float(a.get("price", 0)),
                        size=float(a.size) if hasattr(a, 'size') else float(a.get("size", 0))
                    )
                    for a in (response.asks or [])
                ]
                market_id = response.market if hasattr(response, 'market') else ""
            else:
                # Fall back to dict handling
                bids = [
                    OrderbookLevel(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
                    for b in response.get("bids", [])
                ]
                asks = [
                    OrderbookLevel(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
                    for a in response.get("asks", [])
                ]
                market_id = response.get("market", "")
            
            # Sort bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            
            return Orderbook(
                market_id=market_id,
                token_id=token_id,
                bids=bids,
                asks=asks,
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return None
    
    async def get_price(self, token_id: str) -> Optional[float]:
        """
        Get current midpoint price for a token.
        
        Args:
            token_id: The token ID
            
        Returns:
            Midpoint price or None if failed
        """
        orderbook = await self.get_orderbook(token_id)
        if orderbook:
            return orderbook.midpoint
        return None
    
    async def get_prices(self, token_ids: List[str]) -> Dict[str, float]:
        """
        Get prices for multiple tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Dictionary mapping token_id to price
        """
        prices = {}
        for token_id in token_ids:
            price = await self.get_price(token_id)
            if price is not None:
                prices[token_id] = price
        return prices
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Trading
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def place_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        order_type: str = "GTC",  # Good Till Cancelled
    ) -> Optional[Order]:
        """
        Place a limit order.
        
        Args:
            token_id: Token ID to trade
            side: BUY or SELL
            price: Limit price (0-1 for prediction markets)
            size: Number of shares
            order_type: Order type (GTC, FOK, etc.)
            
        Returns:
            Order object or None if failed
        """
        self._ensure_initialized()
        
        if self.settings.is_paper_trading():
            logger.info(f"[PAPER] Would place {side.value} order: {size} @ {price}")
            # Return a simulated order
            return Order(
                id=f"paper_{datetime.now().timestamp()}",
                market_id="",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.FILLED,  # Assume instant fill in paper trading
                filled_size=size,
                filled_at=datetime.now(),
            )
        
        try:
            # Create order args
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.value,
            )
            
            # Place the order
            response = self._client.create_order(order_args)
            
            if response and response.get("orderID"):
                return Order(
                    id=response["orderID"],
                    market_id=response.get("market", ""),
                    token_id=token_id,
                    side=side,
                    price=price,
                    size=size,
                    status=OrderStatus.OPEN,
                )
            else:
                logger.error(f"Order placement failed: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def place_market_order(
        self,
        token_id: str,
        side: Side,
        amount: float,  # USD amount to spend
    ) -> Optional[Order]:
        """
        Place a market order (takes best available price).
        
        Args:
            token_id: Token ID to trade
            side: BUY or SELL
            amount: USD amount to spend/receive
            
        Returns:
            Order object or None if failed
        """
        self._ensure_initialized()
        
        if self.settings.is_paper_trading():
            # Get current price for simulation
            price = await self.get_price(token_id) or 0.5
            size = amount / price
            
            logger.info(f"[PAPER] Would place market {side.value}: ${amount} @ ~{price}")
            return Order(
                id=f"paper_market_{datetime.now().timestamp()}",
                market_id="",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.FILLED,
                filled_size=size,
                filled_at=datetime.now(),
            )
        
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side.value,
            )
            
            response = self._client.create_market_order(order_args)
            
            if response:
                return Order(
                    id=response.get("orderID", ""),
                    market_id=response.get("market", ""),
                    token_id=token_id,
                    side=side,
                    price=response.get("avg_price", 0),
                    size=response.get("size", 0),
                    status=OrderStatus.FILLED,
                    filled_size=response.get("size", 0),
                    filled_at=datetime.now(),
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        self._ensure_initialized()
        
        if self.settings.is_paper_trading():
            logger.info(f"[PAPER] Would cancel order {order_id}")
            return True
        
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_open_orders(self, market_id: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.
        
        Args:
            market_id: Optional filter by market
            
        Returns:
            List of open Order objects
        """
        self._ensure_initialized()
        
        try:
            params = OpenOrderParams(market=market_id) if market_id else OpenOrderParams()
            response = self._client.get_orders(params)
            
            orders = []
            for order_data in response:
                orders.append(Order(
                    id=order_data.get("id", ""),
                    market_id=order_data.get("market", ""),
                    token_id=order_data.get("asset_id", ""),
                    side=Side(order_data.get("side", "BUY")),
                    price=float(order_data.get("price", 0)),
                    size=float(order_data.get("original_size", 0)),
                    status=OrderStatus.OPEN,
                    filled_size=float(order_data.get("size_matched", 0)),
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Account
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_balance(self) -> float:
        """
        Get USDC balance.
        
        Returns:
            USDC balance in USD
        """
        self._ensure_initialized()
        
        if self.settings.is_paper_trading():
            return self.settings.money.starting_balance
        
        try:
            params = BalanceAllowanceParams(asset_type="USDC")
            response = self._client.get_balance_allowance(params)
            return float(response.get("balance", 0))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
        """
        self._ensure_initialized()
        
        # Note: py-clob-client may not have direct position endpoint
        # This would need to be calculated from trade history
        # For now, return empty list - positions are tracked by PositionManager
        return []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Health Check
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def is_healthy(self) -> bool:
        """Check if the client is working properly."""
        try:
            # Try to fetch markets as a health check
            markets = await self.get_markets(limit=1)
            return len(markets) > 0
        except:
            return False
    
    def close(self):
        """Clean up resources."""
        self._client = None
        self._initialized = False


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════════


async def create_client(settings=None) -> PolymarketClient:
    """
    Create and initialize a Polymarket client.
    
    Args:
        settings: Optional Settings object
        
    Returns:
        Initialized PolymarketClient
    """
    client = PolymarketClient(settings)
    await client.initialize()
    return client


if __name__ == "__main__":
    # Test the client
    async def test():
        client = await create_client()
        
        print("Testing Polymarket Client...")
        print("=" * 60)
        
        # Test market fetch
        markets = await client.get_markets(limit=5)
        print(f"\nFetched {len(markets)} markets:")
        for m in markets:
            print(f"  - {m.question[:60]}...")
            print(f"    YES: {m.yes_price:.2f}, Volume: ${m.volume_24h:,.0f}")
        
        # Test orderbook
        if markets and markets[0].yes_token_id:
            book = await client.get_orderbook(markets[0].yes_token_id)
            if book:
                print(f"\nOrderbook for {markets[0].question[:40]}...")
                print(f"  Best Bid: {book.best_bid}")
                print(f"  Best Ask: {book.best_ask}")
                print(f"  Spread: {book.spread_percent:.2f}%")
        
        # Test balance
        balance = await client.get_balance()
        print(f"\nBalance: ${balance:.2f}")
        
        client.close()
    
    asyncio.run(test())