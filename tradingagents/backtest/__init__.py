"""Deterministic backtesting and evaluation for agent trading decisions.

Public surface:
- run_backtest / run_walk_forward: replay dated BUY/SELL/HOLD signals over
  OHLCV bars via backtrader with next-open execution (no lookahead).
- run_recorded_backtest: evaluate the decisions already persisted under
  eval_results/ without spending a single LLM call.
- metrics: pure-math Sharpe / drawdown / win-rate helpers.
"""

from .engine import (
    BacktestResult,
    WalkForwardResult,
    normalize_price_frame,
    run_backtest,
    run_recorded_backtest,
    run_recorded_walk_forward,
    run_walk_forward,
)
from .metrics import (
    CRYPTO_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    summarize_performance,
    win_rate,
)
from .signals import ACTION_ALIASES, load_recorded_signals, normalize_action
from .teach import (
    compute_decision_outcomes,
    default_agent_memories,
    teach_memories_from_history,
)

__all__ = [
    "ACTION_ALIASES",
    "BacktestResult",
    "CRYPTO_DAYS_PER_YEAR",
    "TRADING_DAYS_PER_YEAR",
    "WalkForwardResult",
    "annualized_return",
    "compute_decision_outcomes",
    "cumulative_return",
    "default_agent_memories",
    "load_recorded_signals",
    "max_drawdown",
    "normalize_action",
    "normalize_price_frame",
    "run_backtest",
    "run_recorded_backtest",
    "run_recorded_walk_forward",
    "run_walk_forward",
    "sharpe_ratio",
    "summarize_performance",
    "teach_memories_from_history",
    "win_rate",
]
