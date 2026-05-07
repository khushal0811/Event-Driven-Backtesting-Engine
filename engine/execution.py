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
      - update_price(market_event)  on every MarketEvent   (price tracking)
      - on_order_event(order, queue) on every OrderEvent   (fill generation)
    """

    @abstractmethod
    def update_price(self, event: MarketEvent) -> None:
        """Record the latest market price for a symbol."""
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
    Fills orders immediately at the last known close price.

    Behaviour:
      - Maintains a per-symbol price register updated by update_price().
      - On receiving an OrderEvent, fills the full quantity at the
        last registered price for that symbol.
      - If no price has been seen for a symbol yet, the order is dropped
        with a warning (prevents fills at price=0 corrupting the portfolio).
      - FillEvent timestamp is the current UTC time of the fill call,
        preserving the event-time relationship.

    Args:
        None — stateless at construction; state builds from MarketEvents.
    """

    def __init__(self) -> None:
        # Last known close price per symbol
        self._prices: Dict[str, float] = {}
        # Last known timestamp per symbol (carried into FillEvent)
        self._timestamps: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # ExecutionEngine interface
    # ------------------------------------------------------------------

    def update_price(self, event: MarketEvent) -> None:
        """Record the latest close price and timestamp for a symbol."""
        self._prices[event.symbol]     = event.price
        self._timestamps[event.symbol] = event.timestamp

    def on_order_event(self, event: OrderEvent, queue: EventQueue) -> None:
        """
        Fill the order at the last known market price and enqueue a FillEvent.

        Drops the order silently with a warning if no price is available.
        """
        symbol = event.symbol
        price  = self._get_fill_price(symbol)

        if price is None:
            print(
                f"[ExecutionEngine] Warning: no market price for '{symbol}' — "
                f"order dropped."
            )
            return

        timestamp = self._timestamps.get(symbol, datetime.now(tz=timezone.utc))

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
        return f"SimulatedExecutionEngine(symbols_tracked={list(self._prices.keys())})"
