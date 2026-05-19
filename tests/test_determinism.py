"""
Group 10 — Determinism (T96–T99)

Same input must always give same output. The most important property.
"""

import pytest
from datetime import date

from engine.config import BacktestConfig, StrategyConfig
from engine.metrics import MetricsResult


# -----------------------------------------------------------------------
# T96 — Two runs with identical config produce identical results
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT96:
    def test_identical_results(self, data_dir, has_aapl_data):
        from run_backtest import run_backtest_from_config

        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(
                type="moving_average_crossover",
                parameters={"short_window": 5, "long_window": 20},
            ),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
        )

        r1 = run_backtest_from_config(cfg, data_dir=data_dir)
        r2 = run_backtest_from_config(cfg, data_dir=data_dir)

        assert r1.total_return == r2.total_return
        assert r1.sharpe_ratio == r2.sharpe_ratio
        assert r1.max_drawdown == r2.max_drawdown
        assert r1.final_value == r2.final_value
        assert r1.cagr == r2.cagr
        assert r1.total_trades == r2.total_trades
        assert r1.total_dividend_income == r2.total_dividend_income


# -----------------------------------------------------------------------
# T97 — Equity curve points are identical across two runs
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT97:
    def test_identical_equity_curve(self, data_dir, has_aapl_data):
        from engine.data_handler import DataHandler
        from engine.engine import Engine
        from engine.execution import SimulatedExecutionEngine
        from engine.order_manager import FixedSizeOrderManager
        from engine.portfolio import Portfolio
        from engine.strategy import MovingAverageCrossover

        def run_once():
            dh = DataHandler(symbols=["AAPL"], data_dir=data_dir)
            strat = MovingAverageCrossover(short_window=5, long_window=20)
            om = FixedSizeOrderManager(quantity=100)
            ee = SimulatedExecutionEngine()
            pf = Portfolio(initial_cash=100_000.0)
            eng = Engine(data_handler=dh, strategy=strat, order_manager=om,
                         execution_engine=ee, portfolio=pf)
            eng.run()
            return pf.bar_history

        h1 = run_once()
        h2 = run_once()

        assert len(h1) == len(h2)
        for s1, s2 in zip(h1, h2):
            assert s1.timestamp == s2.timestamp
            assert s1.total_value == s2.total_value


# -----------------------------------------------------------------------
# T98 — Changing only the strategy type changes the result
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT98:
    def test_different_strategies_differ(self, data_dir, has_aapl_data):
        from run_backtest import run_backtest_from_config

        base = dict(
            symbols=["AAPL"],
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
            position_sizing="fixed",
            position_size=100,
        )

        r1 = run_backtest_from_config(
            BacktestConfig(
                **base,
                strategy=StrategyConfig(
                    type="moving_average_crossover",
                    parameters={"short_window": 5, "long_window": 20},
                ),
            ),
            data_dir=data_dir,
        )

        r2 = run_backtest_from_config(
            BacktestConfig(
                **base,
                strategy=StrategyConfig(
                    type="momentum",
                    parameters={"lookback": 20, "threshold": 0.02},
                ),
            ),
            data_dir=data_dir,
        )

        # Different strategies should produce different trades
        assert r1.total_return != r2.total_return or r1.total_trades != r2.total_trades


# -----------------------------------------------------------------------
# T99 — Changing risk_per_trade changes position sizes
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT99:
    def test_risk_slider_effect(self, data_dir, has_aapl_data):
        from run_backtest import run_backtest_from_config

        base = dict(
            symbols=["AAPL"],
            strategy=StrategyConfig(
                type="moving_average_crossover",
                parameters={"short_window": 5, "long_window": 20},
            ),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
            position_sizing="risk_based",
            stop_fraction=0.02,
        )

        r1 = run_backtest_from_config(
            BacktestConfig(**base, risk_per_trade=0.01), data_dir=data_dir,
        )
        r2 = run_backtest_from_config(
            BacktestConfig(**base, risk_per_trade=0.05), data_dir=data_dir,
        )

        # 5x risk should produce different P&L
        assert r1.final_value != r2.final_value
