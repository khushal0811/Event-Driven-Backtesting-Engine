"""
engine.py — Main event loop for the backtesting engine.

Wires all components together and drives the simulation.

Event routing:
  MarketEvent  → Strategy.on_market_event()
  SignalEvent  → OrderManager.on_signal_event()
  OrderEvent   → ExecutionEngine.on_order_event()
  FillEvent    → Portfolio.on_fill_event()

Additionally, on every MarketEvent:
  - ExecutionEngine.update_price()   (keeps fill prices current)
  - Portfolio.update_market_value()  (keeps holdings value current)

Loop invariant:
  One MarketEvent is pulled from the DataHandler at a time.
  The queue is fully drained before the next bar is consumed.
  This preserves strict temporal ordering — no future information leaks.
"""

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
        data_handler     : Source of MarketEvents (DataHandler instance).
        strategy         : Strategy that converts MarketEvents → SignalEvents.
        order_manager    : Converts SignalEvents → OrderEvents.
        execution_engine : Converts OrderEvents → FillEvents.
        portfolio        : Tracks cash, positions, and equity curve.

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
    ) -> None:
        self._data        = data_handler
        self._strategy    = strategy
        self._orders      = order_manager
        self._execution   = execution_engine
        self._portfolio   = portfolio
        self._queue       = EventQueue()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> MetricsResult:
        """
        Execute the backtest and return performance metrics.

        For each bar in the DataHandler:
          1. Update price registers (ExecutionEngine + Portfolio).
          2. Enqueue the MarketEvent.
          3. Drain the queue fully — each event is routed to its handler,
             which may enqueue further downstream events.
          4. Move to the next bar only when the queue is empty.

        Returns:
            MetricsResult with total return, Sharpe ratio, max drawdown.
        """
        bars_processed = 0

        for market_event in self._data:
            # Keep price registers current before any routing
            self._execution.update_price(market_event)
            self._portfolio.update_market_value(market_event)

            # Seed the queue with this bar's MarketEvent
            self._queue.put(market_event)

            # Drain — order matters: Market → Signal → Order → Fill
            while not self._queue.empty():
                event = self._queue.get()
                self._route(event)

            bars_processed += 1

        return compute_metrics(
            self._portfolio.history,
            self._portfolio.initial_cash,
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
        t = event.event_type

        if t == EventType.MARKET:
            self._strategy.on_market_event(event, self._queue)

        elif t == EventType.SIGNAL:
            self._orders.on_signal_event(event, self._queue)

        elif t == EventType.ORDER:
            self._execution.on_order_event(event, self._queue)

        elif t == EventType.FILL:
            self._portfolio.on_fill_event(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Engine("
            f"strategy={self._strategy!r}, "
            f"order_manager={self._orders!r})"
        )
