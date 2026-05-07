"""
strategy.py — Strategy layer for the backtesting engine.

Defines:
  - Strategy        : Abstract base class all strategies must implement.
  - MovingAverageCrossover : Concrete strategy — emits BUY/SELL SignalEvents
                             when the short MA crosses the long MA.

Contract:
  - Receives MarketEvents from the engine loop.
  - Pushes SignalEvents onto the shared EventQueue.
  - Maintains no external state; all state is per-symbol inside the object.
  - Must be deterministic: same market data → same signals, always.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, Optional

from engine.events import MarketEvent, SignalEvent, SignalType
from engine.event_queue import EventQueue


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """
    All strategies implement this interface.

    The engine loop calls on_market_event() for every MarketEvent it pops
    from the queue.  The strategy is responsible for deciding whether to
    emit a SignalEvent and, if so, putting it onto the queue.

    Args:
        event : The current MarketEvent being processed.
        queue : The shared FIFO EventQueue — use queue.put(signal) to emit.
    """

    @abstractmethod
    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        """Process one MarketEvent and optionally emit SignalEvents."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategy — Moving Average Crossover
# ---------------------------------------------------------------------------

class MovingAverageCrossover(Strategy):
    """
    Classic dual moving-average crossover strategy.

    Logic (per symbol, independently):
      - Accumulate close prices in a rolling window of size `long_window`.
      - Once enough bars have arrived, compute:
          short_ma = mean of the last `short_window` prices
          long_ma  = mean of all `long_window` prices in the window
      - Emit BUY  when short_ma crosses above long_ma (and not already long).
      - Emit SELL when short_ma crosses below long_ma (and not already short).
      - No signal is emitted when already in the correct direction.

    Determinism guarantee:
      - Uses only close prices from MarketEvents in arrival order.
      - deque(maxlen) gives identical rolling windows for identical input.
      - Position state prevents duplicate consecutive signals.

    Args:
        short_window : Number of bars for the fast moving average. Default 5.
        long_window  : Number of bars for the slow moving average. Default 20.

    Raises:
        ValueError: If short_window >= long_window or either <= 0.
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window:  int = 20,
    ) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive integers.")
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be < long_window ({long_window})."
            )

        self._short_window = short_window
        self._long_window  = long_window

        # Per-symbol rolling price buffer (auto-drops oldest beyond long_window)
        self._prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=long_window)
        )

        # Last signal direction emitted per symbol — avoids duplicate signals
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(
            lambda: None
        )

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        """
        Consume a MarketEvent and emit a SignalEvent if a crossover occurred.

        Waits silently until `long_window` bars have accumulated.
        """
        symbol = event.symbol
        self._prices[symbol].append(event.price)

        prices = self._prices[symbol]

        # Not enough bars yet — warm-up period
        if len(prices) < self._long_window:
            return

        prices_list = list(prices)
        short_ma = sum(prices_list[-self._short_window:]) / self._short_window
        long_ma  = sum(prices_list) / self._long_window

        last = self._last_signal[symbol]

        if short_ma > long_ma and last != SignalType.BUY:
            signal = SignalEvent(symbol=symbol, signal_type=SignalType.BUY)
            queue.put(signal)
            self._last_signal[symbol] = SignalType.BUY

        elif short_ma < long_ma and last != SignalType.SELL:
            signal = SignalEvent(symbol=symbol, signal_type=SignalType.SELL)
            queue.put(signal)
            self._last_signal[symbol] = SignalType.SELL

        # short_ma == long_ma → no action (flat, ambiguous crossover)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MovingAverageCrossover("
            f"short={self._short_window}, long={self._long_window})"
        )
