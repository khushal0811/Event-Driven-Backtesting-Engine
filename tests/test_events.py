"""
Group 1 — Events (T01–T06)

The building blocks. If these are wrong, everything is wrong.
"""

import dataclasses
from datetime import datetime

from engine.events import (
    EventType, Event,
    MarketEvent, SignalEvent, OrderEvent, FillEvent, DividendEvent,
    SignalType, OrderSide, OrderType,
)
from engine.event_queue import EventQueue


# -----------------------------------------------------------------------
# T01 — All five event types exist
# -----------------------------------------------------------------------
class TestT01:
    def test_event_type_has_exactly_five_values(self):
        values = set(EventType)
        expected = {
            EventType.MARKET,
            EventType.SIGNAL,
            EventType.ORDER,
            EventType.FILL,
            EventType.DIVIDEND,
        }
        assert values == expected, f"Expected {expected}, got {values}"

    def test_event_type_count(self):
        assert len(EventType) == 5


# -----------------------------------------------------------------------
# T02 — MarketEvent fields are correct
# -----------------------------------------------------------------------
class TestT02:
    def test_market_event_fields(self, now):
        event = MarketEvent(
            timestamp=now, symbol="AAPL", price=150.0, volume=1_000_000.0
        )
        assert event.event_type == EventType.MARKET
        assert event.symbol == "AAPL"
        assert event.price == 150.0
        assert event.volume == 1_000_000.0
        assert event.timestamp == now


# -----------------------------------------------------------------------
# T03 — DividendEvent fields are correct
# -----------------------------------------------------------------------
class TestT03:
    def test_dividend_event_fields(self, now):
        event = DividendEvent(
            timestamp=now, symbol="AAPL", dividend_per_share=0.24
        )
        assert event.event_type == EventType.DIVIDEND
        assert event.symbol == "AAPL"
        assert event.dividend_per_share == 0.24
        assert event.timestamp == now


# -----------------------------------------------------------------------
# T04 — FillEvent fields are correct
# -----------------------------------------------------------------------
class TestT04:
    def test_fill_event_fields(self, now):
        event = FillEvent(
            symbol="MSFT", side=OrderSide.BUY,
            quantity=100, fill_price=300.0, timestamp=now,
        )
        assert event.event_type == EventType.FILL
        assert event.side == OrderSide.BUY
        assert event.quantity == 100
        assert event.fill_price == 300.0


# -----------------------------------------------------------------------
# T05 — Events are plain data containers with no logic
# -----------------------------------------------------------------------
class TestT05:
    def test_events_are_dataclasses(self):
        """All event types should be dataclasses — no custom methods beyond __repr__, __eq__."""
        for cls in [MarketEvent, SignalEvent, OrderEvent, FillEvent, DividendEvent]:
            assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"

    def test_no_custom_methods(self):
        """Events should not have methods that modify state."""
        dataclass_methods = {
            "__init__", "__repr__", "__eq__", "__hash__",
            "__post_init__", "__setattr__", "__delattr__",
            "__getattribute__", "__str__", "__format__",
            "__sizeof__", "__reduce__", "__reduce_ex__",
            "__subclasshook__", "__init_subclass__",
            "__new__", "__class__", "__dir__", "__doc__",
            "__dict__", "__weakref__", "__module__",
            "__qualname__", "__match_args__", "__dataclass_fields__",
            "__dataclass_params__",
        }
        for cls in [MarketEvent, SignalEvent, OrderEvent, FillEvent, DividendEvent]:
            custom = {
                name for name in dir(cls)
                if not name.startswith("__") and callable(getattr(cls, name))
            }
            assert len(custom) == 0, (
                f"{cls.__name__} has custom methods: {custom}"
            )


# -----------------------------------------------------------------------
# T06 — EventQueue is strictly FIFO
# -----------------------------------------------------------------------
class TestT06:
    def test_fifo_order(self, now):
        q = EventQueue()
        e1 = MarketEvent(timestamp=now, symbol="AAPL", price=150.0, volume=1000.0)
        e2 = MarketEvent(timestamp=now, symbol="MSFT", price=300.0, volume=2000.0)
        e3 = SignalEvent(symbol="AAPL", signal_type=SignalType.BUY)

        q.put(e1)
        q.put(e2)
        q.put(e3)

        out1 = q.get()
        out2 = q.get()
        out3 = q.get()

        assert out1.symbol == "AAPL" and out1.event_type == EventType.MARKET
        assert out2.symbol == "MSFT" and out2.event_type == EventType.MARKET
        assert out3.symbol == "AAPL" and out3.event_type == EventType.SIGNAL

    def test_empty_after_drain(self, now):
        q = EventQueue()
        q.put(MarketEvent(timestamp=now, symbol="X", price=1.0, volume=1.0))
        q.get()
        assert q.empty()
