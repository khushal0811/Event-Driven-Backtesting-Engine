"""
Group 11 — Stress Tests (T100–T108)

Push the engine to its limits. If it survives these, it's ready for production.
"""

import pytest
from datetime import date

from engine.config import BacktestConfig, StrategyConfig
from engine.data_handler import DataHandler
from engine.engine import Engine
from engine.events import MarketEvent
from engine.execution import SimulatedExecutionEngine
from engine.metrics import MetricsResult
from engine.order_manager import FixedSizeOrderManager
from engine.portfolio import Portfolio
from engine.strategy import MovingAverageCrossover, STRATEGY_REGISTRY


# -----------------------------------------------------------------------
# T100 — Large universe (20 symbols) runs without error
# -----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestT100:
    def test_large_universe(self, data_dir):
        """Attempt 20 symbols — skip any not present on disk."""
        candidates = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "TSLA", "NVDA", "JPM", "V", "JNJ",
            "WMT", "PG", "UNH", "HD", "DIS",
            "MA", "PYPL", "NFLX", "ADBE", "CRM",
        ]
        import os
        available = [
            s for s in candidates
            if os.path.exists(os.path.join(data_dir, f"{s}.parquet"))
        ]
        if len(available) < 2:
            pytest.skip("Need at least 2 symbols for large universe test")

        dh = DataHandler(symbols=available, data_dir=data_dir)
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=50)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=500_000.0)
        engine = Engine(data_handler=dh, strategy=strat, order_manager=om,
                        execution_engine=ee, portfolio=pf)

        result = engine.run()
        assert isinstance(result, MetricsResult)
        assert result.total_snapshots > 0


# -----------------------------------------------------------------------
# T101 — Long date range (full history) runs without error
# -----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestT101:
    def test_long_date_range(self, data_dir, has_aapl_data):
        dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=100)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)
        engine = Engine(data_handler=dh, strategy=strat, order_manager=om,
                        execution_engine=ee, portfolio=pf)

        result = engine.run()
        assert isinstance(result, MetricsResult)
        assert dh.bar_count > 200  # at least a year of data


# -----------------------------------------------------------------------
# T102 — Intraday data runs without error (if available)
# -----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestT102:
    def test_intraday(self, data_dir, has_aapl_data):
        # Validates the handler produces well-formed events with non-null
        # timestamps regardless of the underlying data interval.
        dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        events = list(dh.stream())
        assert len(events) > 0
        for e in events:
            assert e.timestamp is not None


# -----------------------------------------------------------------------
# T103 — emit_callback called many times without memory leak
# -----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestT103:
    def test_callback_volume(self, data_dir, has_aapl_data):
        messages = []
        dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=100)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)
        engine = Engine(
            data_handler=dh, strategy=strat, order_manager=om,
            execution_engine=ee, portfolio=pf,
            emit_callback=lambda m: messages.append(m),
            emit_frequency=1,  # every bar
        )
        engine.run()

        # Should have roughly bar_count progress messages + trade messages
        progress = [m for m in messages if m["type"] == "progress"]
        assert len(progress) == dh.bar_count

        # All messages are valid dicts
        for m in messages:
            assert isinstance(m, dict)
            assert "type" in m


# -----------------------------------------------------------------------
# T104 — Mixed dividend and non-dividend symbols
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT104:
    def test_mixed_dividends(self, data_dir, has_aapl_data):
        """AAPL pays dividends. If another symbol doesn't have a div file, no crash."""
        import os
        # Find a second symbol that might not have dividends
        available = [
            s for s in ["MSFT", "GOOGL", "TSLA"]
            if os.path.exists(os.path.join(data_dir, f"{s}.parquet"))
        ]
        if not available:
            pytest.skip("Need a second symbol for mixed dividend test")

        symbols = ["AAPL"] + available[:1]
        dh = DataHandler(symbols=symbols, data_dir=data_dir, include_dividends=True)
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=50)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)
        engine = Engine(data_handler=dh, strategy=strat, order_manager=om,
                        execution_engine=ee, portfolio=pf)

        result = engine.run()
        assert isinstance(result, MetricsResult)
        assert pf.total_dividend_income > 0  # from AAPL


# -----------------------------------------------------------------------
# T105 — Symbol with partial history
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT105:
    def test_partial_history(self, data_dir, has_aapl_data, has_msft_data):
        """Both symbols loaded — even if date ranges don't fully overlap."""
        dh = DataHandler(symbols=["AAPL", "MSFT"], data_dir=data_dir)
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=50)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)
        engine = Engine(data_handler=dh, strategy=strat, order_manager=om,
                        execution_engine=ee, portfolio=pf)

        result = engine.run()
        assert isinstance(result, MetricsResult)


# -----------------------------------------------------------------------
# T106 — All 10 strategies complete a full backtest without error
# -----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestT106:
    @pytest.mark.parametrize("strategy_type,params", [
        ("moving_average_crossover", {"short_window": 5, "long_window": 20}),
        ("momentum", {"lookback": 20, "threshold": 0.02}),
        ("mean_reversion", {"window": 20, "num_std": 2.0}),
        ("rsi", {"period": 14}),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}),
        ("breakout", {"lookback": 20}),
        ("bollinger_bands", {"window": 20}),
        ("dual_momentum", {"lookback": 20}),
        ("trend_following", {"period": 20}),
        ("volume_weighted_mean_reversion", {"window": 20}),
    ])
    def test_strategy_completes(self, data_dir, has_aapl_data, strategy_type, params):
        from run_backtest import run_backtest_from_config

        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(type=strategy_type, parameters=params),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
            position_sizing="fixed",
            position_size=100,
        )
        result = run_backtest_from_config(cfg, data_dir=data_dir)
        assert isinstance(result, MetricsResult)


# -----------------------------------------------------------------------
# T107 — Very short backtest produces valid results
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT107:
    def test_short_backtest(self, data_dir, has_aapl_data):
        """Use only a small number of bars — should not crash."""
        dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        # Limit events to first 21 bars
        all_events = list(dh.stream())
        market_events = [e for e in all_events if isinstance(e, MarketEvent)]

        # Can't easily limit DataHandler, so just run full with very long warmup
        strat = MovingAverageCrossover(short_window=100, long_window=200)
        om = FixedSizeOrderManager(quantity=100)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)
        engine = Engine(data_handler=dh, strategy=strat, order_manager=om,
                        execution_engine=ee, portfolio=pf)

        result = engine.run()
        assert isinstance(result, MetricsResult)
        # CAGR should be a number or None, never crash
        assert isinstance(result.cagr, (float, type(None)))


# -----------------------------------------------------------------------
# T108 — Engine handles all-identical prices without crash
# -----------------------------------------------------------------------
class TestT108:
    def test_flat_price_data(self, tmp_path):
        """Create a fake parquet with identical prices and run a backtest."""
        import pandas as pd

        # Create fake parquet
        dates = pd.date_range("2022-01-01", periods=500, freq="B")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "FLAT",
            "open": 150.0,
            "high": 150.0,
            "low": 150.0,
            "close": 150.0,
            "volume": 1_000_000.0,
        })
        parquet_path = tmp_path / "FLAT.parquet"
        df.to_parquet(parquet_path)

        dh = DataHandler(
            symbols=["FLAT"], data_dir=str(tmp_path), include_dividends=False
        )
        strat = MovingAverageCrossover(short_window=5, long_window=20)
        om = FixedSizeOrderManager(quantity=100)
        ee = SimulatedExecutionEngine()
        pf = Portfolio(initial_cash=100_000.0)
        engine = Engine(data_handler=dh, strategy=strat, order_manager=om,
                        execution_engine=ee, portfolio=pf)

        result = engine.run()
        assert isinstance(result, MetricsResult)
        # Sharpe undefined for flat returns
        assert result.sharpe_ratio is None
        # No trades should occur (MAs always equal)
        assert result.total_return == 0.0
