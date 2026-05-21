"""
engine.py — Main event loop for the backtesting engine.

Wires all components together and drives the simulation.

Event routing:
  MarketEvent   → Strategy.on_market_event()
  SignalEvent   → OrderManager.on_signal_event()
  OrderEvent    → ExecutionEngine.on_order_event()
  FillEvent     → Portfolio.on_fill_event()
  DividendEvent → Portfolio.on_dividend_event()

Additionally, on every MarketEvent:
  - ExecutionEngine.update_price()   (keeps fill prices current)
  - Portfolio.update_market_value()  (keeps holdings value current)

Loop invariant:
  One event is pulled from the DataHandler at a time.
  The queue is fully drained before the next event is consumed.
  This preserves strict temporal ordering — no future information leaks.
"""

from typing import Callable, Dict, Optional

from engine.data_handler   import DataHandler
from engine.event_queue    import EventQueue
from engine.events         import EventType, MarketEvent
from engine.execution      import ExecutionEngine
from engine.metrics        import MetricsResult, compute_metrics
from engine.order_manager  import OrderManager
from engine.portfolio      import Portfolio
from engine.strategy       import Strategy


class Engine:
    """
    Drives the event-driven backtest simulation.

    Args:
        data_handler     : Source of MarketEvents and DividendEvents (DataHandler instance).
        strategy         : Strategy that converts MarketEvents → SignalEvents.
        order_manager    : Converts SignalEvents → OrderEvents.
        execution_engine : Converts OrderEvents → FillEvents.
        portfolio        : Tracks cash, positions, and equity curve.
        emit_callback    : Optional callback for live streaming progress/trade/dividend events.
        emit_frequency   : Emit progress every N bars (default: 10).

    Usage:
        engine = Engine(data_handler, strategy, order_manager,
                        execution_engine, portfolio)
        results = engine.run()
        print(results)
    """

    def __init__(
        self,
        data_handler:     DataHandler,
        strategy:         Strategy,
        order_manager:    OrderManager,
        execution_engine: ExecutionEngine,
        portfolio:        Portfolio,
        emit_callback:    Optional[Callable[[dict], None]] = None,
        emit_frequency:   int = 10,
        interval:         str = "1d",
    ) -> None:
        self._data        = data_handler
        self._strategy    = strategy
        self._orders      = order_manager
        self._execution   = execution_engine
        self._portfolio   = portfolio
        self._queue       = EventQueue()
        self._emit        = emit_callback or (lambda msg: None)
        self._emit_freq   = emit_frequency
        self._current_timestamp = None
        self._interval    = interval

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> MetricsResult:
        """
        Execute the backtest and return performance metrics.

        For each event from the DataHandler:
          1. If MarketEvent: update price registers (ExecutionEngine + Portfolio).
          2. Enqueue the event.
          3. Drain the queue fully — each event is routed to its handler,
             which may enqueue further downstream events.
          4. Move to the next event only when the queue is empty.
          5. Emit progress updates every N bars via the callback.

        DividendEvents skip price updates and go directly to the queue.

        Returns:
            MetricsResult with total return, Sharpe ratio, max drawdown, and more.
        """
        bars_processed = 0
        total_bars     = self._data.bar_count
        current_timestamp = None

        for event in self._data:
            if isinstance(event, MarketEvent):
                # If the timestamp changes, snapshot the portfolio for the PREVIOUS timestamp
                if current_timestamp is not None and event.timestamp != current_timestamp:
                    self._portfolio.record_bar_snapshot(current_timestamp)
                
                current_timestamp = event.timestamp
                self._current_timestamp = event.timestamp
                
                # Keep price registers current before any routing
                self._execution.update_price(event)
                self._portfolio.update_last_price(event.symbol, event.price)

                # Seed the queue with this bar's MarketEvent
                self._queue.put(event)

                # Drain — order matters: Market → Signal → Order → Fill
                while not self._queue.empty():
                    queued_event = self._queue.get()
                    self._route(queued_event)

                bars_processed += 1

                # Emit progress update every N bars
                if bars_processed % self._emit_freq == 0 or bars_processed == total_bars:
                    self._emit({
                        "type":      "progress",
                        "bar":       bars_processed,
                        "total":     total_bars,
                        "percent":   round((bars_processed / total_bars) * 100, 1),
                        "equity":    round(self._portfolio.total_value, 2),
                        "timestamp": event.timestamp.isoformat(),
                    })
            else:
                # DividendEvent — skip price update, route directly
                self._queue.put(event)
                while not self._queue.empty():
                    queued_event = self._queue.get()
                    self._route(queued_event)

        # Record the final bar snapshot
        if current_timestamp is not None:
            self._portfolio.record_bar_snapshot(current_timestamp)

        return compute_metrics(
            bar_history      = self._portfolio.bar_history,
            fill_history     = self._portfolio.history,
            initial_cash     = self._portfolio.initial_cash,
            dividend_income  = self._portfolio.total_dividend_income,
            trades           = self._portfolio.completed_trades,
            interval         = self._interval,
        )

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, event) -> None:
        """
        Dispatch a single event to its registered handler.

        Each handler may put new events onto self._queue, which will be
        processed in subsequent iterations of the drain loop.
        """
        if hasattr(event, "timestamp") and event.timestamp is None:
            event.timestamp = self._current_timestamp

        t = event.event_type

        if t == EventType.MARKET:
            self._strategy.on_market_event(event, self._queue)

        elif t == EventType.SIGNAL:
            self._orders.on_signal_event(event, self._queue)

        elif t == EventType.ORDER:
            self._execution.on_order_event(event, self._queue)

        elif t == EventType.FILL:
            self._portfolio.on_fill_event(event)
            # Emit trade event to frontend
            self._emit({
                "type":       "trade",
                "symbol":     event.symbol,
                "side":       event.side.value,
                "quantity":   event.quantity,
                "fill_price": round(event.fill_price, 4),
                "timestamp":  event.timestamp.isoformat(),
            })

        elif t == EventType.DIVIDEND:
            self._portfolio.on_dividend_event(event)
            # Emit dividend event to frontend
            self._emit({
                "type":                "dividend",
                "symbol":              event.symbol,
                "dividend_per_share":  event.dividend_per_share,
                "timestamp":           event.timestamp.isoformat(),
            })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Engine("
            f"strategy={self._strategy!r}, "
            f"order_manager={self._orders!r})"
        )
