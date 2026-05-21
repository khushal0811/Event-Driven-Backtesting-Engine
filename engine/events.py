"""
events.py — Event type definitions for the backtesting engine.

Defines the base Event class and all concrete event types:
  - MarketEvent
  - SignalEvent
  - OrderEvent
  - FillEvent
  - DividendEvent

All events are plain data containers. No logic lives here.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    MARKET   = "MARKET"
    SIGNAL   = "SIGNAL"
    ORDER    = "ORDER"
    FILL     = "FILL"
    DIVIDEND = "DIVIDEND"


class SignalType(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """
    Abstract base for all events.

    Every event carries an event_type so the engine can route it
    to the correct handler without isinstance checks.
    """
    event_type: EventType


# ---------------------------------------------------------------------------
# Concrete Events
# ---------------------------------------------------------------------------

@dataclass
class MarketEvent(Event):
    """
    Emitted by the DataHandler when new market data is available.

    Attributes:
        timestamp : Bar timestamp (UTC).
        symbol    : Ticker symbol, e.g. "AAPL".
        price     : Close price for the bar.
        volume    : Volume for the bar.
    """
    timestamp: datetime
    symbol:    str
    price:     float
    volume:    float
    event_type: EventType = field(default=EventType.MARKET, init=False)


@dataclass
class SignalEvent(Event):
    """
    Emitted by the Strategy when it detects a trade opportunity.

    Attributes:
        symbol      : Ticker symbol.
        signal_type : BUY or SELL.
        strength    : Optional signal strength in [0.0, 1.0].
        timestamp   : Optional datetime when signal was generated.
    """
    symbol:      str
    signal_type: SignalType
    strength:    Optional[float] = None
    timestamp:   Optional[datetime] = None
    event_type:  EventType = field(default=EventType.SIGNAL, init=False)


@dataclass
class OrderEvent(Event):
    """
    Emitted by the OrderManager when it converts a signal into an order.

    Attributes:
        symbol     : Ticker symbol.
        side       : BUY or SELL.
        quantity   : Number of units to trade.
        order_type : Always MARKET in this engine.
        timestamp  : Optional datetime when order was generated.
    """
    symbol:     str
    side:       OrderSide
    quantity:   int
    order_type: OrderType = field(default=OrderType.MARKET)
    timestamp:  Optional[datetime] = None
    event_type: EventType = field(default=EventType.ORDER, init=False)


@dataclass
class FillEvent(Event):
    """
    Emitted by the ExecutionEngine when an order has been filled.

    Attributes:
        symbol     : Ticker symbol.
        side       : BUY or SELL.
        quantity   : Units filled.
        fill_price : Execution price.
        timestamp  : Fill timestamp (UTC).
    """
    symbol:     str
    side:       OrderSide
    quantity:   int
    fill_price: float
    timestamp:  datetime
    event_type: EventType = field(default=EventType.FILL, init=False)


@dataclass
class DividendEvent(Event):
    """
    Emitted by the DataHandler when a dividend ex-date is reached
    for a symbol during simulation.

    Attributes:
        timestamp          : Ex-dividend date (UTC).
        symbol             : Ticker symbol.
        dividend_per_share : Dividend in dollars per share.
    """
    timestamp:          datetime
    symbol:             str
    dividend_per_share: float
    event_type: EventType = field(default=EventType.DIVIDEND, init=False)
