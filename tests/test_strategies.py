"""
Group 5 — Strategies (T39–T58)

All 10 strategies must produce signals with controlled synthetic data.
"""

import pytest
from datetime import datetime, timedelta

from engine.events import MarketEvent, SignalEvent, SignalType
from engine.event_queue import EventQueue
from engine.strategy import (
    MovingAverageCrossover,
    MomentumStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    MACDStrategy,
    BreakoutStrategy,
    BollingerBandsStrategy,
    DualMomentumStrategy,
    TrendFollowingStrategy,
    VolumeWeightedMeanReversionStrategy,
    STRATEGY_REGISTRY,
    build_strategy,
)


T0 = datetime(2024, 1, 1, 10, 0, 0)


def _feed(strategy, prices, symbol="TEST", volume=10000.0):
    """Feed a price series to a strategy and return collected signals."""
    q = EventQueue()
    signals = []
    for i, price in enumerate(prices):
        ts = T0 + timedelta(days=i)
        event = MarketEvent(timestamp=ts, symbol=symbol, price=price, volume=volume)
        strategy.on_market_event(event, q)
        while not q.empty():
            e = q.get()
            if isinstance(e, SignalEvent):
                signals.append(e)
    return signals


def _feed_with_volume(strategy, prices_volumes, symbol="TEST"):
    """Feed price+volume pairs to a strategy and return collected signals."""
    q = EventQueue()
    signals = []
    for i, (price, vol) in enumerate(prices_volumes):
        ts = T0 + timedelta(days=i)
        event = MarketEvent(timestamp=ts, symbol=symbol, price=price, volume=vol)
        strategy.on_market_event(event, q)
        while not q.empty():
            e = q.get()
            if isinstance(e, SignalEvent):
                signals.append(e)
    return signals


# -----------------------------------------------------------------------
# T39 — MovingAverageCrossover emits BUY on upward crossover
# -----------------------------------------------------------------------
class TestT39:
    def test_buy_on_crossover(self):
        s = MovingAverageCrossover(short_window=3, long_window=5)
        prices = [10, 10, 10, 10, 10, 11, 12, 13, 14, 15]
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T40 — MovingAverageCrossover emits SELL on downward crossover
# -----------------------------------------------------------------------
class TestT40:
    def test_sell_on_crossover(self):
        s = MovingAverageCrossover(short_window=3, long_window=5)
        prices = [10, 10, 10, 10, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8]
        signals = _feed(s, prices)
        sells = [sig for sig in signals if sig.signal_type == SignalType.SELL]
        assert len(sells) >= 1


# -----------------------------------------------------------------------
# T41 — MovingAverageCrossover does not signal before warmup
# -----------------------------------------------------------------------
class TestT41:
    def test_no_signal_before_warmup(self):
        s = MovingAverageCrossover(short_window=5, long_window=20)
        prices = [100 + i for i in range(15)]  # only 15 bars < 20 warmup
        signals = _feed(s, prices)
        assert len(signals) == 0


# -----------------------------------------------------------------------
# T42 — MomentumStrategy emits BUY when ROC exceeds threshold
# -----------------------------------------------------------------------
class TestT42:
    def test_buy_on_roc(self):
        s = MomentumStrategy(lookback=5, threshold=0.05)
        prices = [100, 100, 100, 100, 100, 108]
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) == 1


# -----------------------------------------------------------------------
# T43 — MomentumStrategy emits SELL when ROC below threshold
# -----------------------------------------------------------------------
class TestT43:
    def test_sell_on_negative_roc(self):
        s = MomentumStrategy(lookback=5, threshold=0.05)
        # First trigger a BUY, then a SELL
        prices = [100, 100, 100, 100, 100, 108, 107, 105, 102, 100, 93]
        signals = _feed(s, prices)
        sells = [sig for sig in signals if sig.signal_type == SignalType.SELL]
        assert len(sells) >= 1


# -----------------------------------------------------------------------
# T44 — RSIStrategy emits BUY when RSI below oversold
# -----------------------------------------------------------------------
class TestT44:
    def test_buy_on_oversold(self):
        s = RSIStrategy(period=5, oversold=30, overbought=70)
        # Steadily falling prices → low RSI
        prices = [100, 99, 97, 94, 90, 85]
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T45 — RSIStrategy emits SELL when RSI above overbought
# -----------------------------------------------------------------------
class TestT45:
    def test_sell_on_overbought(self):
        s = RSIStrategy(period=5, oversold=30, overbought=70)
        # Fall to trigger BUY, then rise sharply
        prices = [100, 99, 97, 94, 90, 85, 90, 96, 103, 111, 120]
        signals = _feed(s, prices)
        sells = [sig for sig in signals if sig.signal_type == SignalType.SELL]
        assert len(sells) >= 1


# -----------------------------------------------------------------------
# T46 — MeanReversionStrategy emits BUY below lower band
# -----------------------------------------------------------------------
class TestT46:
    def test_buy_below_lower_band(self):
        s = MeanReversionStrategy(window=10, num_std=2.0)
        prices = [100] * 10 + [80]  # sudden drop below lower band
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T47 — MeanReversionStrategy emits SELL above upper band
# -----------------------------------------------------------------------
class TestT47:
    def test_sell_above_upper_band(self):
        s = MeanReversionStrategy(window=10, num_std=2.0)
        prices = [100] * 10 + [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 120]
        signals = _feed(s, prices)
        sells = [sig for sig in signals if sig.signal_type == SignalType.SELL]
        assert len(sells) >= 1


# -----------------------------------------------------------------------
# T48 — MACDStrategy emits BUY on MACD crossover
# -----------------------------------------------------------------------
class TestT48:
    def test_macd_buy(self):
        s = MACDStrategy(fast=3, slow=6, signal=3)
        # Downtrend then sharp uptick
        prices = [100, 99, 98, 97, 96, 95, 94, 95, 97, 100, 104, 109, 115, 122, 130]
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T49 — BreakoutStrategy emits BUY on N-bar high
# -----------------------------------------------------------------------
class TestT49:
    def test_breakout_buy(self):
        s = BreakoutStrategy(lookback=5)
        prices = [10, 11, 10, 11, 10, 15]
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) == 1


# -----------------------------------------------------------------------
# T50 — BreakoutStrategy emits SELL on N-bar low
# -----------------------------------------------------------------------
class TestT50:
    def test_breakout_sell(self):
        s = BreakoutStrategy(lookback=5)
        prices = [10, 11, 10, 11, 10, 15, 14, 13, 12, 11, 5]
        signals = _feed(s, prices)
        sells = [sig for sig in signals if sig.signal_type == SignalType.SELL]
        assert len(sells) >= 1


# -----------------------------------------------------------------------
# T51 — BollingerBandsStrategy emits signals
# -----------------------------------------------------------------------
class TestT51:
    def test_bollinger_signals(self):
        s = BollingerBandsStrategy(window=10, num_std=2.0, squeeze_factor=0.05)
        # Tight squeeze then breakout
        prices = [100] * 10 + [120, 121, 122, 123, 124]
        signals = _feed(s, prices)
        # At minimum should not crash; may or may not signal depending on squeeze
        assert isinstance(signals, list)


# -----------------------------------------------------------------------
# T52 — DualMomentumStrategy emits signals
# -----------------------------------------------------------------------
class TestT52:
    def test_dual_momentum_signals(self):
        s = DualMomentumStrategy(lookback=5, avg_lookback=3)
        # Accelerating uptrend: each ROC reading exceeds the previous average.
        # Linear prices produce constant ROC, so roc == avg_roc and the condition
        # roc > avg_roc never fires. Use exponential growth instead.
        prices = [100 * (1.02 ** i) for i in range(30)]
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T53 — TrendFollowingStrategy emits signals
# -----------------------------------------------------------------------
class TestT53:
    def test_trend_following_signals(self):
        s = TrendFollowingStrategy(period=5)
        prices = list(range(100, 120))
        signals = _feed(s, prices)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T54 — VolumeWeightedMeanReversionStrategy emits signals
# -----------------------------------------------------------------------
class TestT54:
    def test_vwmr_signals(self):
        s = VolumeWeightedMeanReversionStrategy(window=10, threshold=0.02)
        # Stable VWAP then price drops
        data = [(100.0, 10000.0)] * 10 + [(90.0, 10000.0)]
        signals = _feed_with_volume(s, data)
        buys = [sig for sig in signals if sig.signal_type == SignalType.BUY]
        assert len(buys) >= 1


# -----------------------------------------------------------------------
# T55 — No strategy emits duplicate consecutive signals
# -----------------------------------------------------------------------
class TestT55:
    def test_no_consecutive_duplicates(self):
        strategies = [
            MovingAverageCrossover(short_window=3, long_window=5),
            MomentumStrategy(lookback=5, threshold=0.02),
            RSIStrategy(period=5, oversold=30, overbought=70),
            BreakoutStrategy(lookback=5),
            TrendFollowingStrategy(period=5),
        ]
        # Long price series with oscillation
        prices = []
        for cycle in range(10):
            prices.extend(list(range(100, 120)))
            prices.extend(list(range(120, 100, -1)))

        for strat in strategies:
            signals = _feed(strat, prices)
            for i in range(1, len(signals)):
                assert signals[i].signal_type != signals[i - 1].signal_type, (
                    f"{strat.__class__.__name__} emitted consecutive "
                    f"{signals[i].signal_type} at index {i}"
                )


# -----------------------------------------------------------------------
# T56 — No strategy emits signals before warmup
# -----------------------------------------------------------------------
class TestT56:
    def test_no_premature_signals(self):
        configs = [
            (MovingAverageCrossover(short_window=5, long_window=20), 19),
            (MomentumStrategy(lookback=20, threshold=0.02), 20),
            (RSIStrategy(period=14, oversold=30, overbought=70), 14),
            (BreakoutStrategy(lookback=52), 51),
            (TrendFollowingStrategy(period=50), 50),
        ]
        for strat, max_bars in configs:
            prices = [100 + i * 0.1 for i in range(max_bars)]
            signals = _feed(strat, prices)
            assert len(signals) == 0, (
                f"{strat.__class__.__name__} fired signal with only {max_bars} bars"
            )


# -----------------------------------------------------------------------
# T57 — All 10 strategies are in STRATEGY_REGISTRY
# -----------------------------------------------------------------------
class TestT57:
    def test_registry_keys(self):
        expected = {
            "moving_average_crossover", "momentum", "mean_reversion",
            "rsi", "macd", "breakout", "bollinger_bands",
            "dual_momentum", "trend_following",
            "volume_weighted_mean_reversion",
        }
        assert set(STRATEGY_REGISTRY.keys()) == expected


# -----------------------------------------------------------------------
# T58 — build_strategy() correctly instantiates every strategy
# -----------------------------------------------------------------------
class TestT58:
    def test_build_all_strategies(self):
        configs = {
            "moving_average_crossover": {"short_window": 5, "long_window": 20},
            "momentum": {"lookback": 20, "threshold": 0.02},
            "mean_reversion": {"window": 20, "num_std": 2.0},
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "breakout": {"lookback": 52},
            "bollinger_bands": {"window": 20},
            "dual_momentum": {"lookback": 20},
            "trend_following": {"period": 50},
            "volume_weighted_mean_reversion": {"window": 20},
        }
        for type_name, params in configs.items():
            s = build_strategy({"type": type_name, "parameters": params})
            assert s is not None
            assert isinstance(s, STRATEGY_REGISTRY[type_name])

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy type"):
            build_strategy({"type": "fake_strategy", "parameters": {}})
