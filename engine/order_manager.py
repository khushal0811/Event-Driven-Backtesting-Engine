"""
order_manager.py — Order Management layer for the backtesting engine.

Defines:
  - OrderManager        : Abstract base class all order managers must implement.
  - FixedSizeOrderManager : Concrete implementation — converts every SignalEvent
                            into a fixed-quantity market OrderEvent.

Contract:
  - Receives SignalEvents from the engine loop.
  - Pushes OrderEvents onto the shared EventQueue.
  - Applies position sizing logic before creating an order.
  - Never creates a duplicate order in the same direction for a symbol.
  - Fully deterministic: same signals → same orders, always.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from engine.events import (
    SignalEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalType,
)
from engine.event_queue import EventQueue


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OrderManager(ABC):
    """
    All order managers implement this interface.

    The engine loop calls on_signal_event() for every SignalEvent it pops
    from the queue.  The manager decides quantity and side, then puts an
    OrderEvent back onto the queue.

    Args:
        event : The SignalEvent to act on.
        queue : Shared FIFO EventQueue — use queue.put(order) to emit.
    """

    @abstractmethod
    def on_signal_event(self, event: SignalEvent, queue: EventQueue) -> None:
        """Convert a SignalEvent into an OrderEvent and enqueue it."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementation — Fixed-size position sizing
# ---------------------------------------------------------------------------

class FixedSizeOrderManager(OrderManager):
    """
    Translates every SignalEvent into a fixed-quantity market order.

    Position-sizing rule:
      - Each BUY  order is for exactly `quantity` units.
      - Each SELL order is for exactly `quantity` units.
      - If the signal direction matches the current open position for a
        symbol, the order is suppressed (no redundant orders).

    All orders are MARKET orders — execution price is determined by the
    ExecutionEngine in the next phase.

    Args:
        quantity : Number of units per order. Must be a positive integer.

    Raises:
        ValueError: If quantity <= 0.
    """

    def __init__(self, quantity: int = 100) -> None:
        if quantity <= 0:
            raise ValueError(f"quantity must be a positive integer, got {quantity}.")
        self._quantity = quantity

        # Tracks the current open-position direction per symbol.
        # None  → flat (no open position)
        # BUY   → currently long
        # SELL  → currently short
        self._open_position: Dict[str, Optional[SignalType]] = {}

    # ------------------------------------------------------------------
    # OrderManager interface
    # ------------------------------------------------------------------

    def on_signal_event(self, event: SignalEvent, queue: EventQueue) -> None:
        """
        Convert a SignalEvent to an OrderEvent and enqueue it.

        Suppresses the order if the symbol is already positioned in the
        requested direction to prevent duplicate orders.
        """
        symbol    = event.symbol
        signal    = event.signal_type
        current   = self._open_position.get(symbol)

        # Suppress if already in this direction
        if signal == current:
            return

        side = OrderSide.BUY if signal == SignalType.BUY else OrderSide.SELL

        order = OrderEvent(
            symbol     = symbol,
            side       = side,
            quantity   = self._quantity,
            order_type = OrderType.MARKET,
        )

        queue.put(order)
        self._open_position[symbol] = signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def quantity(self) -> int:
        """Fixed order size in units."""
        return self._quantity

    def open_position(self, symbol: str) -> Optional[SignalType]:
        """Return the current open-position direction for a symbol, or None."""
        return self._open_position.get(symbol)

    def __repr__(self) -> str:
        return f"FixedSizeOrderManager(quantity={self._quantity})"
