"""
execution.py — Execution Engine for the backtesting engine.

Defines:
  - ExecutionEngine          : Abstract base class all execution engines must implement.
  - SimulatedExecutionEngine : Concrete implementation — fills orders immediately at
                               the last known market close price (no slippage).

Contract:
  - Tracks the latest market price per symbol via update_price().
  - Receives OrderEvents from the engine loop.
  - Emits FillEvents at the current market price onto the shared EventQueue.
  - Deterministic: same prices + orders → same fills, always.

Slippage note:
  The fill_price hook is isolated in _get_fill_price() so a future subclass
  can override it with a slippage model without touching any other logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Optional

from engine.events import (
    MarketEvent,
    OrderEvent,
    FillEvent,
    OrderSide,
)
from engine.event_queue import EventQueue


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ExecutionEngine(ABC):
    """
    All execution engines implement this interface.

    The engine loop calls:
      - update_price(market_event, queue)  on every MarketEvent   (price tracking)
      - on_order_event(order, queue) on every OrderEvent   (fill generation)
    """

    @abstractmethod
    def update_price(self, event: MarketEvent, queue: Optional[EventQueue] = None) -> None:
        """Record the latest market price for a symbol and process any pending orders."""
        ...

    @abstractmethod
    def on_order_event(self, event: OrderEvent, queue: EventQueue) -> None:
        """Simulate execution of an order and emit a FillEvent."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementation — Simulated (no slippage)
# ---------------------------------------------------------------------------

class SimulatedExecutionEngine(ExecutionEngine):
    """
    Fills orders at the last known close price (either immediately or next-bar).

    Behaviour:
      - Maintains a per-symbol price register updated by update_price().
      - If next_bar_pricing is True: orders are queued and filled at the next bar's price.
      - If next_bar_pricing is False: orders are filled immediately at the last known price.
      - If no price has been seen for a symbol yet, the order is dropped
        with a warning (prevents fills at price=0 corrupting the portfolio).

    Args:
      next_bar_pricing: If True, fills orders at the next bar's price to avoid lookahead bias.
    """

    def __init__(self, next_bar_pricing: bool = False) -> None:
        # Last known close price per symbol
        self._prices: Dict[str, float] = {}
        # Last known timestamp per symbol (carried into FillEvent)
        self._timestamps: Dict[str, datetime] = {}
        # Next-bar pricing config
        self._next_bar_pricing = next_bar_pricing
        # Pending orders waiting for next-bar fill
        self._pending_orders: List[OrderEvent] = []

    # ------------------------------------------------------------------
    # ExecutionEngine interface
    # ------------------------------------------------------------------

    def update_price(self, event: MarketEvent, queue: Optional[EventQueue] = None) -> None:
        """Record the latest close price and timestamp for a symbol, and fill pending orders if in next-bar mode."""
        symbol = event.symbol
        self._prices[symbol]     = event.price
        self._timestamps[symbol] = event.timestamp

        # If next-bar pricing is enabled and queue is provided, fill pending orders for this symbol
        if self._next_bar_pricing and queue is not None:
            still_pending = []
            for order in self._pending_orders:
                if order.symbol == symbol:
                    price = self._get_fill_price(symbol)
                    if price is None:
                        print(f"[ExecutionEngine] Warning: no market price for '{symbol}' — order dropped.")
                        continue
                    fill = FillEvent(
                        symbol     = symbol,
                        side       = order.side,
                        quantity   = order.quantity,
                        fill_price = price,
                        timestamp  = event.timestamp,
                    )
                    queue.put(fill)
                else:
                    still_pending.append(order)
            self._pending_orders = still_pending

    def on_order_event(self, event: OrderEvent, queue: EventQueue) -> None:
        """
        Fill the order or queue it depending on next_bar_pricing.
        """
        symbol = event.symbol

        if self._next_bar_pricing:
            self._pending_orders.append(event)
            return

        price = self._get_fill_price(symbol)
        if price is None:
            print(
                f"[ExecutionEngine] Warning: no market price for '{symbol}' — "
                f"order dropped."
            )
            return

        timestamp = event.timestamp or self._timestamps.get(symbol, datetime.now(tz=timezone.utc))

        fill = FillEvent(
            symbol     = symbol,
            side       = event.side,
            quantity   = event.quantity,
            fill_price = price,
            timestamp  = timestamp,
        )

        queue.put(fill)

    # ------------------------------------------------------------------
    # Slippage hook — override in subclasses to add slippage models
    # ------------------------------------------------------------------

    def _get_fill_price(self, symbol: str) -> Optional[float]:
        """
        Return the price at which to fill an order for `symbol`.

        Base implementation: last known close price (zero slippage).
        Subclasses override this to inject slippage, spread, etc.
        """
        return self._prices.get(symbol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def last_price(self, symbol: str) -> Optional[float]:
        """Return the last recorded price for a symbol, or None."""
        return self._prices.get(symbol)

    def __repr__(self) -> str:
        return f"SimulatedExecutionEngine(symbols_tracked={list(self._prices.keys())}, next_bar_pricing={self._next_bar_pricing})"
