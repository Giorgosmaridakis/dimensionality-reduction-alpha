# Backtester

This folder turns saved model signals into portfolio returns. In the main flow, `run_pipeline.py` first creates yearly `predictions_*.csv` files, then calls `Backtester.run_backtest(...)` to evaluate the long/short strategy.

For the full end-to-end explanation from evaluation framework to stock weights, see `../SIGNAL_TO_TRADES_README.md`.

## Where It Enters From `run_pipeline.py`

After `WalkingForwardForecaster.forecast(...)` finishes, `run_pipeline.py` globs:

```text
<Results_Pipeline_*>/<period>/predictions_*.csv
```

and passes the concatenated predictions plus adjusted-close prices into `Backtester`.

## Main Files

- `backtester.py` contains the portfolio simulator.
- `run_backtester.py` is a standalone script that reruns backtests for existing `Results_Pipeline_*` folders.
- `build_annualized_metrics.py` writes `long_short_Annualized_oos_metric.csv` with calendar-year annualized return and volatility.

## Signal-To-Trade Logic

`Backtester.run_backtest(...)` expects one row per `Date` and `Ticker`.

For prediction-based backtests:

1. Rank predictions cross-sectionally each day.
2. Select top `selection_pct` for longs and bottom `selection_pct` for shorts.
3. Allocate 50% gross to the long side and 50% gross to the short side.
4. Shift target weights by one row:

```python
size_df = size_df.shift(1).fillna(0.0)
```

This means a signal dated `t` is traded at the next adjusted close. The backtester no longer uses adjusted open prices.

## Important Outputs

Each backtest folder may include:

- `vbt_generated_positions.csv`: realized target weights after the one-bar shift.
- `long_short_time_series.csv`: daily portfolio time series.
- `long_short_final_oos_metrics.csv`: IC, ICIR, Sharpe, drawdown, and turnover-style stats.
- `long_short_portfolio_performance_results.csv`: vectorbt summary statistics.
- `long_short_portfolio_cumulative_returns.html`: interactive cumulative return chart.

## Things To Watch

- Dates in prediction CSVs are usually `DD-MM-YYYY`; `run_backtester.py` parses them with `dayfirst=True`.
- The strategy currently supports the long/short path used in the thesis experiments.
- Backtest execution is adjusted-close only, so target construction in `data/dataset_builder.py` must stay aligned with the one-row weight shift.
- Sharpe, Sortino and Calmar use vectorbt's default 365-day year (`freq='D'`, `year_freq='365 days'`), even though the data contains trading days only. This is intentional. Sharpe and Sortino are about 1.2035 times larger in size than under the common 252-day convention (`sqrt(365/252)`), so divide by 1.2035 before comparing them with published results. Calmar has no fixed conversion factor. Trades and P&L are not affected.
