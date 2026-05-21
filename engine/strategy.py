"""
strategy.py — Strategy layer for the backtesting engine.

Defines:
  - Strategy                            : Abstract base class all strategies must implement.
  - MovingAverageCrossover              : Dual MA crossover.
  - MomentumStrategy                    : Rate-of-change momentum.
  - MeanReversionStrategy               : Bollinger Band mean reversion.
  - RSIStrategy                         : RSI overbought/oversold.
  - MACDStrategy                        : MACD signal line crossover.
  - BreakoutStrategy                    : N-bar high/low breakout.
  - BollingerBandsStrategy              : Bollinger Band squeeze breakout.
  - DualMomentumStrategy                : Absolute + relative momentum.
  - TrendFollowingStrategy              : EMA slope direction.
  - VolumeWeightedMeanReversionStrategy : Volume-weighted mean reversion.
  - STRATEGY_REGISTRY                   : Maps config type strings to classes.
  - build_strategy()                    : Factory function for config-based instantiation.

Contract:
  - Receives MarketEvents from the engine loop.
  - Pushes SignalEvents onto the shared EventQueue.
  - Maintains no external state; all state is per-symbol inside the object.
  - Must be deterministic: same market data → same signals, always.
"""

import math
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


# ---------------------------------------------------------------------------
# Concrete strategy — Momentum (Rate of Change)
# ---------------------------------------------------------------------------

class MomentumStrategy(Strategy):
    """
    Rate-of-change momentum. BUY when ROC > threshold, SELL when ROC < -threshold.

    Args:
        lookback  : Number of bars to measure price change over.
        threshold : Minimum ROC to trigger a signal (e.g. 0.02 = 2%).
    """

    def __init__(self, lookback: int = 20, threshold: float = 0.02) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        self._lookback   = lookback
        self._threshold  = threshold
        self._prices:    Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 1))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._lookback + 1:
            return
        prices = list(self._prices[symbol])
        roc = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
        last = self._last_signal[symbol]
        if roc > self._threshold and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif roc < -self._threshold and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"MomentumStrategy(lookback={self._lookback}, threshold={self._threshold})"


# ---------------------------------------------------------------------------
# Concrete strategy — Mean Reversion (Bollinger Band)
# ---------------------------------------------------------------------------

class MeanReversionStrategy(Strategy):
    """
    Bollinger Band mean reversion.
    BUY when price drops below lower band, SELL when price rises above upper band.

    Args:
        window  : Rolling window for mean and std calculation.
        num_std : Number of standard deviations for the bands.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window    = window
        self._num_std   = num_std
        self._prices:   Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._window:
            return
        prices = list(self._prices[symbol])
        mean   = sum(prices) / self._window
        std    = math.sqrt(sum((p - mean) ** 2 for p in prices) / self._window)
        upper  = mean + self._num_std * std
        lower  = mean - self._num_std * std
        price  = event.price
        last   = self._last_signal[symbol]
        if price < lower and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif price > upper and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"MeanReversionStrategy(window={self._window}, num_std={self._num_std})"


# ---------------------------------------------------------------------------
# Concrete strategy — RSI
# ---------------------------------------------------------------------------

class RSIStrategy(Strategy):
    """
    RSI overbought/oversold. BUY below oversold, SELL above overbought.

    Args:
        period     : RSI calculation period.
        oversold   : RSI level to trigger BUY signal.
        overbought : RSI level to trigger SELL signal.
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self._period     = period
        self._oversold   = oversold
        self._overbought = overbought
        self._prices:    Dict[str, deque] = defaultdict(lambda: deque(maxlen=period + 1))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._period + 1:
            return
        prices  = list(self._prices[symbol])
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains   = [c for c in changes if c > 0]
        losses  = [-c for c in changes if c < 0]
        avg_gain = sum(gains) / self._period if gains else 0.0
        avg_loss = sum(losses) / self._period if losses else 0.0
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs  = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        last = self._last_signal[symbol]
        if rsi < self._oversold and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif rsi > self._overbought and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"RSIStrategy(period={self._period}, oversold={self._oversold}, overbought={self._overbought})"


# ---------------------------------------------------------------------------
# Concrete strategy — MACD
# ---------------------------------------------------------------------------

class MACDStrategy(Strategy):
    """
    MACD signal line crossover. BUY when MACD crosses above signal, SELL below.

    Args:
        fast   : Fast EMA period.
        slow   : Slow EMA period.
        signal : Signal line EMA period.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self._fast   = fast
        self._slow   = slow
        self._signal = signal
        self._prices:      Dict[str, deque] = defaultdict(lambda: deque(maxlen=slow + signal + 10))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def _ema(self, prices: list, period: int) -> float:
        if len(prices) < period:
            return sum(prices) / len(prices)
        k = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        prices = list(self._prices[symbol])
        if len(prices) < self._slow + self._signal:
            return
        macd_values = []
        for i in range(self._signal, len(prices) + 1):
            window = prices[:i]
            if len(window) < self._slow:
                continue
            fast_ema = self._ema(window, self._fast)
            slow_ema = self._ema(window, self._slow)
            macd_values.append(fast_ema - slow_ema)
        if len(macd_values) < self._signal:
            return
        macd_line   = macd_values[-1]
        signal_line = self._ema(macd_values[-self._signal:], self._signal)
        last = self._last_signal[symbol]
        if macd_line > signal_line and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif macd_line < signal_line and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"MACDStrategy(fast={self._fast}, slow={self._slow}, signal={self._signal})"


# ---------------------------------------------------------------------------
# Concrete strategy — Breakout
# ---------------------------------------------------------------------------

class BreakoutStrategy(Strategy):
    """
    Price breakout. BUY on N-bar high breakout, SELL on N-bar low breakdown.

    Args:
        lookback : Number of bars to define the high/low range.
    """

    def __init__(self, lookback: int = 52) -> None:
        self._lookback    = lookback
        self._prices:     Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._lookback:
            return
        prices = list(self._prices[symbol])
        highest = max(prices[:-1])
        lowest  = min(prices[:-1])
        price   = event.price
        last    = self._last_signal[symbol]
        if price > highest and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif price < lowest and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"BreakoutStrategy(lookback={self._lookback})"


# ---------------------------------------------------------------------------
# Concrete strategy — Bollinger Bands (squeeze breakout variant)
# ---------------------------------------------------------------------------

class BollingerBandsStrategy(Strategy):
    """
    Bollinger Band squeeze breakout. BUY when price breaks above upper band
    after a squeeze (low volatility), SELL when price breaks below lower band.

    Args:
        window         : Rolling window for mean and std calculation.
        num_std        : Number of standard deviations for the bands.
        squeeze_factor : Band width threshold to detect a squeeze (fraction of mean).
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, squeeze_factor: float = 0.02) -> None:
        self._window         = window
        self._num_std        = num_std
        self._squeeze_factor = squeeze_factor
        self._prices:        Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_signal:   Dict[str, Optional[SignalType]] = defaultdict(lambda: None)
        self._was_squeezed:  Dict[str, bool] = defaultdict(lambda: False)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._window:
            return

        prices = list(self._prices[symbol])
        mean   = sum(prices) / self._window
        std    = math.sqrt(sum((p - mean) ** 2 for p in prices) / self._window)
        upper  = mean + self._num_std * std
        lower  = mean - self._num_std * std
        price  = event.price

        # Detect squeeze: band width relative to mean is below threshold
        band_width = (upper - lower) / mean if mean > 0 else 0.0
        is_squeezed = band_width < self._squeeze_factor

        if is_squeezed:
            self._was_squeezed[symbol] = True

        last = self._last_signal[symbol]

        # Signal only on breakout after a squeeze
        if self._was_squeezed[symbol]:
            if price > upper and last != SignalType.BUY:
                queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
                self._last_signal[symbol] = SignalType.BUY
                self._was_squeezed[symbol] = False
            elif price < lower and last != SignalType.SELL:
                queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
                self._last_signal[symbol] = SignalType.SELL
                self._was_squeezed[symbol] = False

    def __repr__(self) -> str:
        return f"BollingerBandsStrategy(window={self._window}, num_std={self._num_std})"


# ---------------------------------------------------------------------------
# Concrete strategy — Dual Momentum
# ---------------------------------------------------------------------------

class DualMomentumStrategy(Strategy):
    """
    Dual momentum: combines absolute momentum (ROC > 0) with relative
    momentum (asset ROC > its own trailing average ROC).

    BUY when both absolute and relative momentum are positive.
    SELL when absolute momentum turns negative.

    Args:
        lookback      : Number of bars for ROC calculation.
        avg_lookback  : Number of ROC readings to average for relative comparison.
    """

    def __init__(self, lookback: int = 20, avg_lookback: int = 5) -> None:
        if lookback <= 0 or avg_lookback <= 0:
            raise ValueError("lookback and avg_lookback must be positive.")
        self._lookback     = lookback
        self._avg_lookback = avg_lookback
        self._prices:      Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 1))
        self._roc_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=avg_lookback))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._lookback + 1:
            return

        prices = list(self._prices[symbol])
        roc = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
        self._roc_history[symbol].append(roc)

        if len(self._roc_history[symbol]) < self._avg_lookback:
            return

        avg_roc = sum(self._roc_history[symbol]) / len(self._roc_history[symbol])
        last = self._last_signal[symbol]

        # Absolute momentum positive AND relative momentum above its average
        if roc > 0 and roc > avg_roc and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif roc < 0 and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"DualMomentumStrategy(lookback={self._lookback}, avg_lookback={self._avg_lookback})"


# ---------------------------------------------------------------------------
# Concrete strategy — Trend Following (EMA slope)
# ---------------------------------------------------------------------------

class TrendFollowingStrategy(Strategy):
    """
    Trend following via EMA slope direction.
    BUY when EMA is rising (current > previous), SELL when falling.

    Args:
        period : EMA period for trend detection.
    """

    def __init__(self, period: int = 50) -> None:
        if period <= 0:
            raise ValueError("period must be positive.")
        self._period = period
        self._prices:      Dict[str, deque] = defaultdict(lambda: deque(maxlen=period + 1))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def _ema(self, prices: list, period: int) -> float:
        if len(prices) < period:
            return sum(prices) / len(prices)
        k = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        if len(self._prices[symbol]) < self._period + 1:
            return

        prices   = list(self._prices[symbol])
        ema_curr = self._ema(prices, self._period)
        ema_prev = self._ema(prices[:-1], self._period)
        last     = self._last_signal[symbol]

        if ema_curr > ema_prev and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif ema_curr < ema_prev and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"TrendFollowingStrategy(period={self._period})"


# ---------------------------------------------------------------------------
# Concrete strategy — Volume-Weighted Mean Reversion
# ---------------------------------------------------------------------------

class VolumeWeightedMeanReversionStrategy(Strategy):
    """
    Volume-weighted mean reversion. Computes a VWAP-like mean and trades
    deviations from it. BUY when price is below VWAP by threshold,
    SELL when above.

    Args:
        window    : Rolling window for VWAP calculation.
        threshold : Minimum deviation from VWAP to trigger (as fraction, e.g. 0.02 = 2%).
    """

    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window    = window
        self._threshold = threshold
        self._prices:   Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._volumes:  Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_signal: Dict[str, Optional[SignalType]] = defaultdict(lambda: None)

    def on_market_event(self, event: MarketEvent, queue: EventQueue) -> None:
        symbol = event.symbol
        self._prices[symbol].append(event.price)
        self._volumes[symbol].append(event.volume)
        if len(self._prices[symbol]) < self._window:
            return

        prices  = list(self._prices[symbol])
        volumes = list(self._volumes[symbol])
        total_vol = sum(volumes)
        if total_vol == 0:
            return

        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        price = event.price
        deviation = (price - vwap) / vwap if vwap > 0 else 0.0
        last = self._last_signal[symbol]

        if deviation < -self._threshold and last != SignalType.BUY:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.BUY))
            self._last_signal[symbol] = SignalType.BUY
        elif deviation > self._threshold and last != SignalType.SELL:
            queue.put(SignalEvent(symbol=symbol, signal_type=SignalType.SELL))
            self._last_signal[symbol] = SignalType.SELL

    def __repr__(self) -> str:
        return f"VolumeWeightedMeanReversionStrategy(window={self._window}, threshold={self._threshold})"


# ---------------------------------------------------------------------------
# Strategy Registry — maps config type strings to strategy classes
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "moving_average_crossover":       MovingAverageCrossover,
    "momentum":                       MomentumStrategy,
    "mean_reversion":                 MeanReversionStrategy,
    "rsi":                            RSIStrategy,
    "macd":                           MACDStrategy,
    "breakout":                       BreakoutStrategy,
    "bollinger_bands":                BollingerBandsStrategy,
    "dual_momentum":                  DualMomentumStrategy,
    "trend_following":                TrendFollowingStrategy,
    "volume_weighted_mean_reversion": VolumeWeightedMeanReversionStrategy,
}


def build_strategy(config: dict) -> Strategy:
    """
    Instantiate the correct Strategy subclass from a structured config dict.

    Args:
        config: dict with keys:
            "type"        (str)  — must match a key in STRATEGY_REGISTRY, or "custom"
            "parameters"  (dict) — keyword arguments passed to the strategy __init__
            "python_code" (str)  — user-defined Python strategy implementation if type is "custom"

    Returns:
        Instantiated Strategy object.

    Raises:
        ValueError: If strategy type is not in registry/custom invalid.
        TypeError:  If parameters don't match the strategy constructor.
    """
    strategy_type = config.get("type")
    
    if strategy_type == "custom":
        python_code = config.get("python_code")
        if not python_code or not python_code.strip():
            raise ValueError("Custom strategy type requested but python_code is empty.")
            
        # Execute the python code in a clean local namespace
        local_namespace = {
            "Strategy": Strategy,
            "MarketEvent": MarketEvent,
            "SignalEvent": SignalEvent,
            "SignalType": SignalType,
            "EventQueue": EventQueue,
            "math": math,
        }
        try:
            exec(python_code, globals(), local_namespace)
        except Exception as e:
            raise ValueError(f"Failed to execute custom strategy Python code: {e}")
            
        # Find subclasses of Strategy inside the executed namespace
        strategy_classes = [
            v for k, v in local_namespace.items()
            if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
        ]
        
        if not strategy_classes:
            raise ValueError(
                "Could not find any subclass of Strategy in the provided python_code. "
                "Ensure your custom class inherits from Strategy."
            )
            
        cls = strategy_classes[0]
        params = config.get("parameters", {})
        try:
            return cls(**params)
        except TypeError as e:
            raise TypeError(
                f"Invalid parameters for custom strategy '{cls.__name__}': {e}"
            )

    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy type: '{strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    cls    = STRATEGY_REGISTRY[strategy_type]
    params = config.get("parameters", {})

    try:
        return cls(**params)
    except TypeError as e:
        raise TypeError(
            f"Invalid parameters for strategy '{strategy_type}': {e}"
        )
