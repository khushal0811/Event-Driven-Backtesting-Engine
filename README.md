# Event-Driven Backtesting Engine

A deterministic, event-driven backtesting engine that simulates trading strategies over historical market data and produces realistic performance metrics.

Built to explore trading system architecture, event-driven simulation, and portfolio execution workflows.

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
│   ├── events.py          # Event types: Market, Signal, Order, Fill
│   ├── event_queue.py     # FIFO queue (collections.deque)
│   ├── data_handler.py    # Loads Parquet data → streams MarketEvents
│   ├── strategy.py        # Strategy interface + MovingAverageCrossover
│   ├── order_manager.py   # OrderManager interface + FixedSizeOrderManager
│   ├── execution.py       # ExecutionEngine interface + SimulatedExecutionEngine
│   ├── portfolio.py       # Tracks cash, positions, equity curve
│   ├── metrics.py         # Total return, Sharpe ratio, max drawdown
│   └── engine.py          # Main event loop — wires all components
├── run_backtest.py         # CLI + importable backtest runner
├── requirements.txt
└── LICENSE
```

---

## Prerequisites

### 1. Python

Python 3.9+ required.

### 2. Dependencies

```bash
pip install -r requirements.txt
```

### 3. Market Data

This engine reads Parquet files produced by the companion [Backtester-Oriented-Market-Data-Pipeline](https://github.com/khushal0811/Backtester-Oriented-Market-Data-Pipeline).

Clone that repo as a sibling directory and run it to fetch data:

```
Project_2/
├── Event-Driven-Backtesting-Engine/   ← this repo
└── Backtester-Oriented-Market-Data-Pipeline/
    └── data/
        ├── AAPL.parquet
        └── MSFT.parquet
```

The engine will automatically find the `data/` directory from the sibling pipeline repo.

---

## Usage

### Command Line

```bash
# Run with all available symbols and default settings
python run_backtest.py

# Specify symbols
python run_backtest.py --symbols AAPL MSFT

# Full configuration
python run_backtest.py \
  --symbols AAPL MSFT TSLA \
  --short 5      \
  --long  20     \
  --cash  100000 \
  --qty   100

# Custom data directory
python run_backtest.py --symbols AAPL --data-dir /path/to/parquet/files
```

**CLI Options**

| Flag | Default | Description |
|---|---|---|
| `--symbols` | all in `data/` | Ticker symbols to backtest |
| `--short` | `5` | Fast moving average window (bars) |
| `--long` | `20` | Slow moving average window (bars) |
| `--cash` | `100000` | Starting capital in dollars |
| `--qty` | `100` | Fixed order size in units |
| `--data-dir` | sibling pipeline `data/` | Path to Parquet files |

### Example Output

```
==================================================
  Event-Driven Backtesting Engine
==================================================
  Symbols      : ['AAPL', 'MSFT']
  Total bars   : 502
  Strategy     : MovingAverageCrossover(short=5, long=20)
  Order size   : 100 units
  Initial cash : $100,000.00
==================================================

  Trades executed : 12
  Final positions : {'AAPL': 100, 'MSFT': 0}

========================================
  Backtest Performance Summary
========================================
  Initial Value   : $  100,000.00
  Final Value     : $  112,430.00
  Total Return    :      +12.43%
  Sharpe Ratio    :        1.4821
  Max Drawdown    :         4.21%
  Equity Points   :            12
========================================
```

### In Python (importable)

```python
from run_backtest import run_backtest

results = run_backtest(
    symbols      = ['AAPL', 'MSFT'],
    initial_cash = 100_000,
    short_window = 5,
    long_window  = 20,
    quantity     = 100,
)

print(f"Total return : {results.total_return * 100:.2f}%")
print(f"Sharpe ratio : {results.sharpe_ratio:.4f}")
print(f"Max drawdown : {results.max_drawdown * 100:.2f}%")
print(f"Final NAV    : ${results.final_value:,.2f}")
```

---

## Components

### Event System (`events.py`, `event_queue.py`)

Four event types flow through the system:

| Event | Fields | Emitted by |
|---|---|---|
| `MarketEvent` | timestamp, symbol, price, volume | DataHandler |
| `SignalEvent` | symbol, signal_type (BUY/SELL), strength | Strategy |
| `OrderEvent` | symbol, side, quantity, order_type | OrderManager |
| `FillEvent` | symbol, side, quantity, fill_price, timestamp | ExecutionEngine |

### Strategy (`strategy.py`)

The `Strategy` ABC defines the interface. `MovingAverageCrossover` is the included implementation:

- Emits **BUY** when the short MA crosses above the long MA
- Emits **SELL** when the short MA crosses below the long MA
- Signals only fire on direction changes — no duplicate emissions

```python
from engine.strategy import MovingAverageCrossover
strategy = MovingAverageCrossover(short_window=5, long_window=20)
```

Extend by subclassing `Strategy` and implementing `on_market_event()`.

### Order Manager (`order_manager.py`)

`FixedSizeOrderManager` converts every signal into a fixed-quantity MARKET order. Suppresses orders that would duplicate the current open position.

```python
from engine.order_manager import FixedSizeOrderManager
order_manager = FixedSizeOrderManager(quantity=100)
```

### Execution Engine (`execution.py`)

`SimulatedExecutionEngine` fills orders immediately at the last known close price (zero slippage). Subclass and override `_get_fill_price()` to add a slippage model:

```python
from engine.execution import SimulatedExecutionEngine

class SlippageEngine(SimulatedExecutionEngine):
    def _get_fill_price(self, symbol):
        base = super()._get_fill_price(symbol)
        return base * 1.001 if base else None  # 0.1% slippage
```

### Portfolio (`portfolio.py`)

Tracks cash and net positions per symbol. Records a `PortfolioSnapshot` on every fill — these snapshots form the equity curve consumed by the metrics module.

### Metrics (`metrics.py`)

| Metric | Formula |
|---|---|
| Total return | `(final - initial) / initial` |
| Sharpe ratio | `mean(returns) / std(returns) * √252` (annualised, risk-free=0) |
| Max drawdown | Largest peak-to-trough decline (positive decimal) |

---

## Design Principles

- **Event-driven** — no vectorised shortcuts; every bar triggers real event flow
- **Deterministic** — same input data always produces identical results
- **Modular** — swap any component (strategy, execution, order sizing) without touching others
- **Extensible** — abstract base classes make it straightforward to add new strategies or slippage models

---

## License

MIT — see [LICENSE](LICENSE).
