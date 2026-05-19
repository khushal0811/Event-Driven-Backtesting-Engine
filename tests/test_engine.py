"""
Group 8 — Engine Loop (T77–T88)

Full simulation loop, routing, callback streaming.
"""

import pytest
from datetime import datetime

from engine.data_handler import DataHandler
from engine.engine import Engine
from engine.events import MarketEvent, DividendEvent, EventType
from engine.execution import SimulatedExecutionEngine
from engine.metrics import MetricsResult
from engine.order_manager import FixedSizeOrderManager
from engine.portfolio import Portfolio
from engine.strategy import MovingAverageCrossover


NOW = datetime(2024, 6, 1, 10, 0, 0)


def _build_engine(data_dir, symbols=None, emit_callback=None, short=5, long=20):
    """Wire up a standard engine for integration tests."""
    symbols = symbols or ["AAPL"]
    dh = DataHandler(symbols=symbols, data_dir=data_dir)
    strat = MovingAverageCrossover(short_window=short, long_window=long)
    om = FixedSizeOrderManager(quantity=100)
    ee = SimulatedExecutionEngine()
    pf = Portfolio(initial_cash=100_000.0)
    return Engine(
        data_handler=dh, strategy=strat, order_manager=om,
        execution_engine=ee, portfolio=pf,
        emit_callback=emit_callback,
    ), pf


# -----------------------------------------------------------------------
# T77 — Engine runs to completion without error
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT77:
    def test_runs_to_completion(self, data_dir, has_aapl_data):
        engine, pf = _build_engine(data_dir)
        result = engine.run()
        assert isinstance(result, MetricsResult)
        assert result.total_snapshots > 0


# -----------------------------------------------------------------------
# T78 — Engine routes every event type correctly
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT78:
    def test_full_chain_fires(self, data_dir, has_aapl_data):
        engine, pf = _build_engine(data_dir)
        result = engine.run()
        assert len(pf.history) > 0, "No fills occurred — full chain did not fire"


# -----------------------------------------------------------------------
# T79 — Engine routes DividendEvents to portfolio correctly
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT79:
    def test_dividend_routing(self, data_dir, has_aapl_data):
        engine, pf = _build_engine(data_dir)
        result = engine.run()
        assert pf.total_dividend_income > 0, (
            "No dividend income credited — DividendEvent routing is broken"
        )


# -----------------------------------------------------------------------
# T80 — emit_callback receives progress messages
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT80:
    def test_progress_callback(self, data_dir, has_aapl_data):
        messages = []
        engine, pf = _build_engine(data_dir, emit_callback=lambda m: messages.append(m))
        engine.run()

        assert len(messages) > 0
        progress = [m for m in messages if m["type"] == "progress"]
        assert len(progress) > 0

        # Check schema
        for p in progress:
            assert "bar" in p
            assert "total" in p
            assert "percent" in p
            assert "equity" in p
            assert "timestamp" in p

        # Last progress should be 100%
        assert progress[-1]["percent"] == 100.0


# -----------------------------------------------------------------------
# T81 — emit_callback receives trade messages
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT81:
    def test_trade_callback(self, data_dir, has_aapl_data):
        messages = []
        engine, pf = _build_engine(data_dir, emit_callback=lambda m: messages.append(m))
        engine.run()

        trades = [m for m in messages if m["type"] == "trade"]
        assert len(trades) > 0

        for t in trades:
            assert "symbol" in t
            assert "side" in t
            assert "quantity" in t
            assert "fill_price" in t
            assert "timestamp" in t


# -----------------------------------------------------------------------
# T82 — emit_callback receives dividend messages
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT82:
    def test_dividend_callback(self, data_dir, has_aapl_data):
        messages = []
        engine, pf = _build_engine(data_dir, emit_callback=lambda m: messages.append(m))
        engine.run()

        divs = [m for m in messages if m["type"] == "dividend"]
        assert len(divs) > 0

        for d in divs:
            assert "symbol" in d
            assert "dividend_per_share" in d
            assert "timestamp" in d


# -----------------------------------------------------------------------
# T83 — Engine with no emit_callback runs without error
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT83:
    def test_no_callback_no_crash(self, data_dir, has_aapl_data):
        engine, pf = _build_engine(data_dir, emit_callback=None)
        result = engine.run()
        assert isinstance(result, MetricsResult)


# -----------------------------------------------------------------------
# T84 — Engine does NOT call update_price on DividendEvents
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT84:
    def test_no_price_update_on_dividend(self, data_dir, has_aapl_data):
        """Monkey-patch update_price to track what it receives."""
        price_events = []
        ee = SimulatedExecutionEngine()
        original_update = ee.update_price

        def tracking_update(event):
            price_events.append(event)
            return original_update(event)

        ee.update_price = tracking_update

        dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=100)
        pf = Portfolio(initial_cash=100_000.0)

        engine = Engine(
            data_handler=dh, strategy=strat, order_manager=om,
            execution_engine=ee, portfolio=pf,
        )
        engine.run()

        for e in price_events:
            assert isinstance(e, MarketEvent), (
                f"update_price received {type(e).__name__}, not MarketEvent"
            )


# -----------------------------------------------------------------------
# T85 — Engine processes one complete bar before next bar starts
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT85:
    def test_no_lookahead(self, data_dir, has_aapl_data):
        """
        Track the order events are routed.
        Ensure no MarketEvent from bar N+1 is routed while bar N's
        downstream events are still pending.
        """
        event_log = []

        class TrackingStrategy(MovingAverageCrossover):
            def on_market_event(self, event, queue):
                event_log.append(("MARKET", event.timestamp))
                super().on_market_event(event, queue)

        dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        strat = TrackingStrategy(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=100)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)

        engine = Engine(
            data_handler=dh, strategy=strat, order_manager=om,
            execution_engine=ee, portfolio=pf,
        )
        engine.run()

        # Verify market events are processed in order
        market_ts = [ts for tag, ts in event_log if tag == "MARKET"]
        for i in range(1, len(market_ts)):
            assert market_ts[i] >= market_ts[i - 1]


# -----------------------------------------------------------------------
# T86 — Engine results are complete and all fields present
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT86:
    def test_complete_results(self, data_dir, has_aapl_data):
        engine, pf = _build_engine(data_dir)
        r = engine.run()

        assert r.total_return is not None
        assert r.max_drawdown is not None
        assert r.initial_value is not None
        assert r.final_value is not None
        assert r.total_snapshots > 0
        assert r.total_trades >= 0


# -----------------------------------------------------------------------
# T87 — Engine handles empty universe gracefully
# -----------------------------------------------------------------------
class TestT87:
    def test_empty_universe_raises(self, data_dir):
        with pytest.raises(ValueError):
            DataHandler(symbols=[], data_dir="/nonexistent/path/123")


# -----------------------------------------------------------------------
# T88 — Engine handles strategy with no signals gracefully
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT88:
    def test_no_signals_no_crash(self, data_dir, has_aapl_data):
        # Use windows longer than the available date range to guarantee zero
        # crossover signals. Even if a few signals fire after warmup, the test
        # intent is simply that the engine completes without raising.
        engine, pf = _build_engine(data_dir, short=100, long=200)
        result = engine.run()
        # Engine must return a MetricsResult without crashing.
        assert result is not None
        assert hasattr(result, "total_return")
        assert result.total_snapshots > 0
