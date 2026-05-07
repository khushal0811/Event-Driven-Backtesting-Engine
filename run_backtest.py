"""
run_backtest.py — Backtest Runner for the Event-Driven Backtesting Engine.

Wires all components, runs the simulation, and prints a results summary.

Usage (CLI):
    python run_backtest.py
    python run_backtest.py --symbols AAPL MSFT --cash 50000
    python run_backtest.py --symbols AAPL --short 5 --long 20 --qty 100

Usage (import):
    from run_backtest import run_backtest
    results = run_backtest(symbols=['AAPL'], initial_cash=100_000)
    print(results)
"""

import argparse
import os
import sys

# Resolve the pipeline path relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.abspath(
    os.path.join(_PROJECT_ROOT, "..", "Backtester-Oriented-Market-Data-Pipeline")
)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from engine.data_handler   import DataHandler
from engine.engine         import Engine
from engine.execution      import SimulatedExecutionEngine
from engine.metrics        import MetricsResult
from engine.order_manager  import FixedSizeOrderManager
from engine.portfolio      import Portfolio
from engine.strategy       import MovingAverageCrossover


# ---------------------------------------------------------------------------
# Core runner function (importable)
# ---------------------------------------------------------------------------

def run_backtest(
    symbols:      list,
    data_dir:     str   = None,
    initial_cash: float = 100_000.0,
    short_window: int   = 5,
    long_window:  int   = 20,
    quantity:     int   = 100,
) -> MetricsResult:
    """
    Run a full backtest and return performance metrics.

    Args:
        symbols      : List of ticker symbols to backtest.
        data_dir     : Path to the directory containing Parquet files.
                       Defaults to the pipeline's data/ directory.
        initial_cash : Starting capital in dollars.
        short_window : Fast MA window for the strategy.
        long_window  : Slow MA window for the strategy.
        quantity     : Fixed order size in units.

    Returns:
        MetricsResult — total return, Sharpe ratio, max drawdown, etc.

    Raises:
        ValueError       : If components are misconfigured.
        FileNotFoundError: If Parquet files are missing.
    """
    if data_dir is None:
        data_dir = os.path.join(_PIPELINE_ROOT, "data")

    # -- Instantiate components --
    data_handler     = DataHandler(symbols=symbols, data_dir=data_dir)
    strategy         = MovingAverageCrossover(short_window=short_window,
                                              long_window=long_window)
    order_manager    = FixedSizeOrderManager(quantity=quantity)
    execution_engine = SimulatedExecutionEngine()
    portfolio        = Portfolio(initial_cash=initial_cash)

    # -- Print run configuration --
    print("\n" + "=" * 50)
    print("  Event-Driven Backtesting Engine")
    print("=" * 50)
    print(f"  Symbols      : {data_handler.symbols}")
    print(f"  Total bars   : {data_handler.bar_count}")
    print(f"  Strategy     : {strategy!r}")
    print(f"  Order size   : {quantity} units")
    print(f"  Initial cash : ${initial_cash:,.2f}")
    print("=" * 50)

    # -- Run --
    engine  = Engine(data_handler, strategy, order_manager,
                     execution_engine, portfolio)
    results = engine.run()

    # -- Print trade summary --
    history = portfolio.history
    if history:
        print(f"\n  Trades executed : {len(history)}")
        print(f"  Final positions : {portfolio.positions}")
    else:
        print("\n  No trades executed (strategy did not trigger).")
        print("  Tip: try a shorter --long window or more historical data.")

    # -- Print metrics --
    print(results)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Event-Driven Backtesting Engine — Backtest Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols", nargs="+", default=[],
        help="Ticker symbols to backtest. Defaults to all available.",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to Parquet data directory. Defaults to pipeline data/.",
    )
    parser.add_argument(
        "--cash", type=float, default=100_000.0,
        help="Starting capital in dollars.",
    )
    parser.add_argument(
        "--short", type=int, default=5,
        help="Short moving average window.",
    )
    parser.add_argument(
        "--long", type=int, default=20,
        help="Long moving average window.",
    )
    parser.add_argument(
        "--qty", type=int, default=100,
        help="Fixed order quantity in units.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        symbols      = args.symbols,
        data_dir     = args.data_dir,
        initial_cash = args.cash,
        short_window = args.short,
        long_window  = args.long,
        quantity     = args.qty,
    )
