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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.portfolio import Portfolio

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


# ---------------------------------------------------------------------------
# Concrete implementation — Percentage-of-equity position sizing
# ---------------------------------------------------------------------------

class PercentageOrderManager(OrderManager):
    """
    Sizes each order as a fixed percentage of current portfolio equity.

    Example: 10% allocation, $100k portfolio, AAPL at $150
    → quantity = floor((100,000 × 0.10) / 150) = 66 shares

    Args:
        percent_of_equity : Fraction of total equity per position (e.g. 0.10 = 10%).
        portfolio         : Reference to the Portfolio for current equity lookup.

    Raises:
        ValueError: If percent_of_equity is not in (0, 1].
    """

    def __init__(self, percent_of_equity: float, portfolio: "Portfolio") -> None:
        if not (0 < percent_of_equity <= 1.0):
            raise ValueError(
                f"percent_of_equity must be in (0, 1], got {percent_of_equity}."
            )
        self._pct       = percent_of_equity
        self._portfolio = portfolio
        self._open_position: Dict[str, Optional[SignalType]] = {}

    def on_signal_event(self, event: SignalEvent, queue: EventQueue) -> None:
        symbol  = event.symbol
        signal  = event.signal_type
        current = self._open_position.get(symbol)

        if signal == current:
            return

        price = self._portfolio.last_price(symbol)
        if not price or price <= 0:
            return  # no price data yet — skip

        equity   = self._portfolio.total_value
        quantity = int((equity * self._pct) / price)

        if quantity <= 0:
            return  # position too small — skip

        side  = OrderSide.BUY if signal == SignalType.BUY else OrderSide.SELL
        order = OrderEvent(
            symbol=symbol, side=side,
            quantity=quantity, order_type=OrderType.MARKET
        )
        queue.put(order)
        self._open_position[symbol] = signal

    def __repr__(self) -> str:
        return f"PercentageOrderManager(percent={self._pct:.1%})"


# ---------------------------------------------------------------------------
# Concrete implementation — Risk-based position sizing
# ---------------------------------------------------------------------------

class RiskBasedOrderManager(OrderManager):
    """
    Sizes positions so that the risk per trade equals a fixed percentage
    of current portfolio equity — driven by the risk-per-trade slider.

    Position sizing formula:
        quantity = floor((equity × risk_pct) / (entry_price × stop_fraction))

    Where stop_fraction is the ATR-based or fixed stop distance as a
    fraction of entry price (default: 2% if no ATR available).

    This is the professional standard — position size is determined by
    how much you are willing to lose, not by how much you want to own.

    Args:
        risk_per_trade  : Fraction of equity to risk per trade (e.g. 0.02 = 2%).
        stop_fraction   : Stop distance as fraction of price (e.g. 0.02 = 2%).
                          Used when ATR data is not available.
        portfolio       : Reference to Portfolio for equity lookup.

    Raises:
        ValueError: If risk_per_trade or stop_fraction are out of range.
    """

    def __init__(
        self,
        risk_per_trade: float,
        stop_fraction:  float,
        portfolio:      "Portfolio",
    ) -> None:
        if not (0 < risk_per_trade <= 0.20):
            raise ValueError(
                f"risk_per_trade must be in (0, 0.20], got {risk_per_trade}."
            )
        if not (0 < stop_fraction <= 0.50):
            raise ValueError(
                f"stop_fraction must be in (0, 0.50], got {stop_fraction}."
            )

        self._risk          = risk_per_trade
        self._stop_fraction = stop_fraction
        self._portfolio     = portfolio
        self._open_position: Dict[str, Optional[SignalType]] = {}

    def on_signal_event(self, event: SignalEvent, queue: EventQueue) -> None:
        symbol  = event.symbol
        signal  = event.signal_type
        current = self._open_position.get(symbol)

        if signal == current:
            return

        price = self._portfolio.last_price(symbol)
        if not price or price <= 0:
            return

        equity    = self._portfolio.total_value
        risk_amt  = equity * self._risk             # e.g. $2,000 at 2% of $100k
        stop_dist = price * self._stop_fraction     # e.g. $3.00 at 2% of $150
        quantity  = int(risk_amt / stop_dist)

        if quantity <= 0:
            return

        side  = OrderSide.BUY if signal == SignalType.BUY else OrderSide.SELL
        order = OrderEvent(
            symbol=symbol, side=side,
            quantity=quantity, order_type=OrderType.MARKET
        )
        queue.put(order)
        self._open_position[symbol] = signal

    def __repr__(self) -> str:
        return (
            f"RiskBasedOrderManager("
            f"risk={self._risk:.1%}, stop={self._stop_fraction:.1%})"
        )
