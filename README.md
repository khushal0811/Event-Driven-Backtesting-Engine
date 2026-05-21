# Event-Driven Backtesting Engine

A deterministic, event-driven backtesting engine that simulates trading strategies over historical market data and produces realistic performance metrics.

Built to explore trading system architecture, event-driven simulation, and portfolio execution workflows.

Part of a three-repo trading system workspace:
- [Market Data Pipeline](https://github.com/khushal0811/Backtester-Oriented-Market-Data-Pipeline) — Data ingestion, normalization, event streaming
- **This repo** — Event-Driven Backtesting Engine (strategy simulation)
- [Strategy Research Terminal](https://github.com/khushal0811/strategy-research-terminal) — Full-stack research UI

---

## Architecture

```
Market Data (Parquet)
        ↓
   DataHandler
        ↓
   Event Queue  ←─────────────────────────────┐
        ↓                                     │
   [MarketEvent] → Strategy → [SignalEvent]   │
                                    ↓         │
                           OrderManager       │
                                    ↓         │
                            [OrderEvent]      │
                                    ↓         │
                         ExecutionEngine      │
                                    ↓         │
                             [FillEvent] ─────┘
                                    ↓
                               Portfolio
                                    ↓
                            Performance Metrics
```

Each component communicates **only through events** on a shared FIFO queue. No direct calls between modules. The loop drains one bar at a time — guaranteeing look-ahead-free, deterministic simulation.

---

## Project Structure

```
Event-Driven-Backtesting-Engine/
├── engine/
│   ├── events.py          # Event types: Market, Dividend, Signal, Order, Fill
│   ├── event_queue.py     # FIFO queue (collections.deque)
│   ├── data_handler.py    # Loads Parquet data → streams MarketEvents + DividendEvents
│   ├── strategy.py        # 10 built-in strategies + strategy registry
│   ├── order_manager.py   # Fixed, Percentage, and Risk-Based order managers
│   ├── execution.py       # SimulatedExecutionEngine with slippage hook
│   ├── portfolio.py       # Tracks cash, positions, dividends, equity curve
│   ├── metrics.py         # Full performance metrics suite
│   ├── config.py          # BacktestConfig + StrategyConfig dataclasses
│   └── engine.py          # Main event loop — wires all components
├── run_backtest.py         # CLI + importable backtest runner
├── tests/                 # 126-test suite covering all components
│   ├── conftest.py
│   ├── test_events.py
│   ├── test_data_handler.py
│   ├── test_portfolio.py
│   ├── test_order_managers.py
│   ├── test_strategies.py
│   ├── test_execution.py
│   ├── test_metrics.py
│   ├── test_engine.py
│   ├── test_config.py
│   ├── test_determinism.py
│   └── test_stress.py
├── requirements.txt
└── LICENSE
```

---

## Setup

### 1. Repository layout

Both repos must be siblings under a shared parent directory:

```
market-data-pipeline/
├── Event-Driven-Backtesting-Engine/   ← this repo
└── Backtester-Oriented-Market-Data-Pipeline/
    └── data/
        ├── AAPL.parquet
        ├── AAPL_dividends.parquet
        ├── MSFT.parquet
        └── MSFT_dividends.parquet
```

### 2. Create and activate a virtual environment

```bash
cd Event-Driven-Backtesting-Engine
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install engine dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the market-data pipeline as an editable package

```bash
pip install -e ../Backtester-Oriented-Market-Data-Pipeline
```

This makes the `market_data` package importable inside the engine's venv without copying files. Only needs to be done once.

### 5. Fetch market data

```bash
cd ../Backtester-Oriented-Market-Data-Pipeline
python scripts/fetch_data.py \
  --symbols AAPL,MSFT \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --interval 1d \
  --dividends \
  --data-dir data/
cd ../Event-Driven-Backtesting-Engine
```

---

## Running the Tests

```bash
pytest tests/ -v
```

Expected: **126 passed** across all functional groups — events, data handler, portfolio, order managers, strategies, execution, metrics, engine, config, determinism, and stress tests.

---

## Usage

### Command Line

```bash
# Run with default settings (AAPL, moving average crossover)
python run_backtest.py

# Specify symbols
python run_backtest.py --symbols AAPL MSFT

# Full configuration
python run_backtest.py \
  --symbols AAPL MSFT \
  --strategy moving_average_crossover \
  --short 5 --long 20 \
  --cash 100000 \
  --qty 100
```

### In Python (importable)

```python
from datetime import date
from engine.config import BacktestConfig, StrategyConfig
from run_backtest import run_backtest_from_config

cfg = BacktestConfig(
    symbols=["AAPL", "MSFT"],
    strategy=StrategyConfig(
        type="moving_average_crossover",
        parameters={"short_window": 5, "long_window": 20},
    ),
    start_date=date(2022, 1, 1),
    end_date=date(2024, 1, 1),
    initial_cash=100_000,
    position_sizing="risk_based",
    risk_per_trade=0.02,
)

result = run_backtest_from_config(cfg)

print(f"Total return  : {result.total_return * 100:.2f}%")
print(f"CAGR          : {result.cagr * 100:.2f}%")
print(f"Sharpe ratio  : {result.sharpe_ratio:.4f}")
print(f"Max drawdown  : {result.max_drawdown * 100:.2f}%")
print(f"Win rate      : {result.win_rate * 100:.1f}%")
print(f"Total trades  : {result.total_trades}")
print(f"Dividend income: ${result.total_dividend_income:,.2f}")
print(f"Final NAV     : ${result.final_value:,.2f}")
```

---

## Components

### Event System (`events.py`, `event_queue.py`)

Five event types flow through the system:

| Event | Fields | Emitted by |
|---|---|---|
| `MarketEvent` | timestamp, symbol, open, high, low, close, volume | DataHandler |
| `DividendEvent` | timestamp, symbol, amount | DataHandler |
| `SignalEvent` | symbol, signal_type (BUY/SELL), strength | Strategy |
| `OrderEvent` | symbol, side, quantity, order_type | OrderManager |
| `FillEvent` | symbol, side, quantity, fill_price, timestamp | ExecutionEngine |

### Strategies (`strategy.py`)

Ten built-in strategies, all accessible via the strategy registry:

| Key | Strategy | Signal Logic |
|---|---|---|
| `moving_average_crossover` | MovingAverageCrossover | Short MA crosses above/below long MA |
| `momentum` | MomentumStrategy | Rate-of-change exceeds threshold |
| `mean_reversion` | MeanReversionStrategy | Price deviates from rolling mean |
| `rsi` | RSIStrategy | RSI crosses oversold/overbought levels |
| `macd` | MACDStrategy | MACD line crosses signal line |
| `breakout` | BreakoutStrategy | Price breaks N-bar high/low |
| `bollinger_bands` | BollingerBandsStrategy | Price crosses Bollinger bands |
| `dual_momentum` | DualMomentumStrategy | ROC exceeds its own rolling average |
| `trend_following` | TrendFollowingStrategy | Price above/below long-term moving average |
| `volume_weighted_mean_reversion` | VWMRStrategy | VWAP-based mean reversion |

```python
from engine.strategy import build_strategy
strategy = build_strategy("rsi", {"period": 14})
```

### Order Managers (`order_manager.py`)

| Class | `position_sizing` | Behaviour |
|---|---|---|
| `FixedSizeOrderManager` | `"fixed"` | Fixed quantity per trade |
| `PercentageOrderManager` | `"percentage"` | % of current portfolio equity |
| `RiskBasedOrderManager` | `"risk_based"` | Kelly-style: risk a fixed % per trade |

All managers suppress duplicate signals (no repeated BUY when already long).

### Execution Engine (`execution.py`)

`SimulatedExecutionEngine` fills orders at the last known close price. Override `_get_fill_price()` to add slippage:

```python
from engine.execution import SimulatedExecutionEngine

class SlippageEngine(SimulatedExecutionEngine):
    def _get_fill_price(self, symbol):
        base = super()._get_fill_price(symbol)
        return base * 1.001 if base else None  # 0.1% slippage
```

### Portfolio (`portfolio.py`)

Tracks cash, net positions, dividend income, and records a `PortfolioSnapshot` on every bar — forming the equity curve fed to the metrics module.

### Metrics (`metrics.py`)

| Metric | Description |
|---|---|
| `total_return` | `(final − initial) / initial` |
| `price_return` | Return excluding dividend income |
| `total_return_with_dividends` | Price return + dividend yield |
| `cagr` | Compound annual growth rate |
| `sharpe_ratio` | Annualised Sharpe (risk-free = 0, √252) |
| `max_drawdown` | Largest peak-to-trough decline |
| `volatility` | Annualised standard deviation of daily returns |
| `win_rate` | Fraction of profitable trades |
| `total_trades` | Total fill count |
| `total_dividend_income` | Cumulative dividends credited |
| `alpha` | Return above benchmark (if benchmark provided) |

---

## Design Principles

- **Event-driven** — no vectorised shortcuts; every bar triggers real event flow
- **Deterministic** — same input always produces identical output
- **Modular** — swap any component (strategy, execution, sizing) without touching others
- **Extensible** — abstract base classes for Strategy, OrderManager, ExecutionEngine

---

## License

MIT — see [LICENSE](LICENSE).
