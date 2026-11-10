# Evaluation, Backtesting, and Signal-to-Portfolio Framework

This README explains how `Pipeline` evaluates model predictions and turns them into simulated portfolios. The short version is:

```text
alpha features at date t
  -> supervised target aligned to next-close execution
  -> walk-forward model signal
  -> predictions_<period>.csv
  -> daily cross-sectional ranks
  -> long/short target weights
  -> vectorbt portfolio simulation
  -> IC, Sharpe, PnL, exposure, drawdown, and turnover outputs
```

The pipeline does not place live broker orders. It creates target portfolio weights and sends them to `vectorbt.Portfolio.from_orders(...)` for a backtest.

---

## 1. Main Components

| Component | File | Role |
|---|---|---|
| Dataset target alignment | `data/dataset_builder.py` | Builds tradable future return targets from adjusted close prices. |
| Walk-forward evaluation | `walking_forward/forecaster.py` | Runs outer train/test periods, tuning, fitting, prediction, and fold metrics. |
| Prediction export | `walking_forward/outputs.py` | Saves model signals, ranks, daily IC, and fold-level prediction metrics. |
| Backtest bridge | `run_pipeline.py` | Reloads all `predictions_*.csv` files and calls the backtester. |
| Portfolio simulator | `Backtester/backtester.py` | Converts predictions into weights and runs the VectorBT simulation. |
| Statistical diagnostics | `evaluation/metrics.py`, `evaluation/statistical_tests.py` | Computes IC diagnostics and optional post-experiment tests. |

---

## 2. Target and Execution Timing

The project uses adjusted-close execution only. Alpha features dated `t` are treated as known after `close_t`. The first tradable price is therefore the next adjusted close, `close_{t+1}`.

`DatasetBuilder.merge_features_and_targets(...)` aligns the supervised target to that trading convention:

```python
close_to_close_returns = adjusted_close.pct_change()
tradable_next_close_returns = close_to_close_returns.shift(-2)
```

So the target for signal date `t` is:

```text
Target_t = adjusted_close_{t+2} / adjusted_close_{t+1} - 1
```

This matches the backtester, which builds weights for signal date `t` and then shifts those weights forward one row before simulation:

```python
size_df = size_df.shift(1).fillna(0.0)
```

That shift means a prediction dated `t` is first executed at the adjusted close of `t+1`.

---

## 3. Walk-Forward Evaluation Framework

`run_pipeline.py` builds the dataset, selects an experiment profile, and calls:

```python
WalkingForwardForecaster.forecast(...)
```

The forecaster uses nested walk-forward evaluation:

```text
1. Choose an outer test period.
2. Train only on data before that test period (`embargo_days=1`, so training ends one calendar day before the test period starts).
3. Tune hyperparameters inside the training window.
4. Fit preprocessing and the final model on the outer training window.
5. Predict only the outer test period.
6. Save predictions and fold-level evaluation metrics.
7. Move to the next outer test period.
```

The outer test labels are not used for tuning or early stopping. They are used only after prediction for out-of-sample evaluation.

---

## 4. Prediction Modes

The final model signal can come from three prediction sources:

| `prediction_source` | Signal Used |
|---|---|
| `tree` | Tree model predictions from raw, PCA, FA, or neural-transformed features. |
| `neural_expected_return` | Direct neural pricing output: `gkx_expected_return` or `factorvae_expected_return`. |
| `neural_then_tree` | A downstream tree trained only on frozen neural pricing outputs. |

For ranking models, higher model score means stronger long candidate. For direct neural expected-return mode, higher expected return also means stronger long candidate.

---

## 5. Prediction Files

Each outer period writes:

```text
Results_Pipeline_*/<period>/predictions_<period>.csv
```

For ranking models, `walking_forward/outputs.py` saves:

```python
Date
Ticker
Predicted_Score
Predicted_Rank
Actual_Score
Actual_Return
Actual_Rank
```

The ranking convention is:

```text
higher Predicted_Score -> better signal
Predicted_Rank = 1     -> strongest long candidate
largest rank           -> weakest name / short candidate
```

For regression or direct expected-return style predictions, the key signal column is usually:

```text
Prediction
```

If `Predicted_Rank` is missing but `Prediction` exists, the backtester ranks `Prediction` cross-sectionally each day.

---

## 6. Bridge From Predictions to Backtester

After `forecast(...)` finishes for an experiment, `run_pipeline.py` collects all period-level prediction files:

```python
pred_files = glob.glob(os.path.join(output_dir, "*", "predictions_*.csv"))
predictions_df = pd.concat([pd.read_csv(path) for path in sorted(pred_files)])
predictions_df["Date"] = pd.to_datetime(predictions_df["Date"], dayfirst=True)
```

It then calls:

```python
bt = Backtester(output_folder=portfolio_dir, universe=universe)
bt.run_backtest(
    data=close_df,
    predictions=predictions_df,
    strategy="long_short",
    selection_pct=selection_pct,
)
```

`close_df` is the adjusted-close price panel loaded by `DataLoader.get_adjusted_close()`.

---

## 7. Signal-to-Weight Logic

`Backtester.run_backtest(...)` converts prediction rows into a wide date-by-ticker matrix.

For ranking predictions:

```python
pred_wide = predictions.pivot(
    index="Date",
    columns="Ticker",
    values="Predicted_Rank",
)
```

For regression or expected-return predictions:

```python
raw_pred_wide = predictions.pivot(
    index="Date",
    columns="Ticker",
    values="Prediction",
)
pred_wide = raw_pred_wide.rank(axis=1, ascending=False, method="min")
```

Then, for each date:

```python
valid_ranks = row.dropna()
n = len(valid_ranks)
percentage = selection_pct if selection_pct is not None else (
    0.20 if self.universe == "ETFs" else 0.10
)

top_k = max(1, int(np.ceil(n * percentage)))
bottom_k = max(1, int(np.ceil(n * percentage)))

long_tickers = valid_ranks.nsmallest(top_k).index.tolist()
short_tickers = valid_ranks.nlargest(bottom_k).index.tolist()
```

So:

```text
long book  = best-ranked selection_pct of assets
short book = worst-ranked selection_pct of assets
```

The default selection percentage is:

| Universe | Default if `selection_pct=None` |
|---|---:|
| ETFs | 20% |
| S&P500 | 10% |

All thesis experiments use `selection_pct=0.05`, meaning top 5% long and bottom 5% short.

---

## 8. Stock Weight Formula

For the main `long_short` strategy, the backtester allocates:

```text
50% gross exposure to longs
50% gross exposure to shorts
```

Weights are equal within each side:

```python
long_weight_per_stock = +0.5 / top_k
short_weight_per_stock = -0.5 / bottom_k
```

In code:

```python
if strategy in ["long", "long_short"]:
    alloc = 0.5 if strategy == "long_short" else 1.0
    size_df.loc[date, long_tickers] = alloc / top_k

if strategy in ["short", "long_short"]:
    alloc = 0.5 if strategy == "long_short" else 1.0
    size_df.loc[date, short_tickers] = -alloc / bottom_k
```

Example with `selection_pct=0.05` and the 483 S&P 500 stocks:

```text
top_k = ceil(483 * 0.05) = 25
bottom_k = ceil(483 * 0.05) = 25

each long stock weight  = +0.5 / 25 = +0.02  (+2%)
each short stock weight = -0.5 / 25 = -0.02  (-2%)

long gross exposure  = 50%
short gross exposure = 50%
total gross exposure = 100%
net exposure         = 0%
```

With the 60 ETFs and `selection_pct=0.05`:

```text
top_k = ceil(60 * 0.05) = 3
bottom_k = 3

each long ETF weight  = +0.5 / 3 = +0.1667  (+16.7%)
each short ETF weight = -0.5 / 3 = -0.1667  (-16.7%)
```

The generated weight matrix is saved as:

```text
portfolio_metrics_*/vbt_generated_positions.csv
```

This is the main audit file for checking the exact target weight of each stock on each date.

---

## 9. VectorBT Portfolio Simulation

The backtester sends the shifted target weights to VectorBT:

```python
pf = vbt.Portfolio.from_orders(
    close_price,
    price=close_price,
    size=size_df,
    size_type="targetpercent",
    freq="D",
    init_cash=self.init_cash,
    fees=self.fees,
    slippage=self.slippage,
    group_by=True,
    cash_sharing=True,
    call_seq="auto",
    log=True,
)
```

Important details:

```text
size_df values are target portfolio percentages, not share counts.
Positive weights create long exposure.
Negative weights create short exposure.
targetpercent rebalances toward the requested percentage weight.
cash_sharing=True treats all assets as one portfolio.
fees and slippage are applied by universe.
```

Default trading assumptions:

| Universe | Fee | Slippage |
|---|---:|---:|
| ETFs | 0.0001 | 0.0002 |
| S&P500 | 0.0001 | 0.0005 |

Initial cash defaults to:

```python
10_000_000
```

---

## 10. Evaluation Metrics

There are two evaluation layers.

### Prediction Evaluation

During each walk-forward fold, `walking_forward/outputs.py` computes daily out-of-sample IC:

```text
ranking task    -> Spearman correlation
regression task -> Pearson correlation
```

Per-period files:

```text
daily_oos_ic_<period>.csv
prediction_metrics_<period>.csv
```

After all periods, the forecaster aggregates:

```text
daily_oos_ic_series.csv
```

### Portfolio Evaluation

The backtester computes portfolio-level outputs from the VectorBT portfolio:

```text
long_short_time_series.csv
long_short_final_oos_metrics.csv
long_short_portfolio_performance_results.csv
long_short_portfolio_cumulative_returns.html
vbt_generated_positions.csv
```

The daily time-series output includes:

```text
Total PnL
Total PnL(%)
Cumulative PnL(%)
Drawdown(%)
No Longs
No Shorts
Long $ Value
Short $ Value
Gross Exposure
Net Exposure
Turnover
```

The final OOS metrics include:

```text
IC
ICIR
Annualized Return
Annualized Volatility
Sharpe Ratio
Sortino Ratio
Max Drawdown
Calmar Ratio
Total Trades
```

---

## 11. What to Inspect After a Run

Use these files to trace one experiment end to end:

```text
Results_Pipeline_*/<period>/predictions_<period>.csv
Results_Pipeline_*/<period>/daily_oos_ic_<period>.csv
Results_Pipeline_*/<period>/prediction_metrics_<period>.csv
Results_Pipeline_*/daily_oos_ic_series.csv

portfolio_metrics_*/vbt_generated_positions.csv
portfolio_metrics_*/long_short_time_series.csv
portfolio_metrics_*/long_short_final_oos_metrics.csv
portfolio_metrics_*/long_short_portfolio_performance_results.csv
```

Interpretation:

| File | What It Shows |
|---|---|
| `predictions_<period>.csv` | Model signal for each date and ticker. |
| `daily_oos_ic_<period>.csv` | Prediction quality by date before portfolio construction. |
| `vbt_generated_positions.csv` | Actual target portfolio weight per stock per date after the one-row shift. |
| `long_short_time_series.csv` | Daily PnL, exposure, drawdown, position counts, and turnover. |
| `long_short_final_oos_metrics.csv` | Final prediction and portfolio metrics. |

---

## 12. Important Caveats

The backtester simulates orders only. It does not connect to a broker.

`entries`, `exits`, `short_entries`, and `short_exits` are built inside `Backtester.run_backtest(...)`, but the actual instruction sent to VectorBT is the target percentage matrix `size_df`.

The active main strategy is `long_short`. Standalone scripts may expose other paths, but the thesis pipeline calls:

```python
strategy="long_short"
```

Because weights are shifted by one row, the first row in `vbt_generated_positions.csv` is usually all zero. That is expected: the first signal cannot be traded until the next adjusted-close bar.

The portfolio is equal-weighted within the selected long and short sides. The current main backtest does not weight stocks by signal strength after selection.

---

## 13. Thesis Methodology Wording

> Model outputs were evaluated in a walk-forward framework. For each out-of-sample period, predictions were saved at the date-ticker level and then converted into daily cross-sectional ranks. The backtester formed a dollar-neutral long-short portfolio by selecting the top `selection_pct` of assets as longs and the bottom `selection_pct` as shorts. In the long-short strategy, 50% gross exposure was allocated equally across selected long positions and 50% gross exposure was allocated equally across selected short positions. Target weights were shifted forward by one trading day so that signals dated `t` were executed at the next adjusted close, consistent with the supervised target return from `t+1` to `t+2`. The resulting target percentage weights were simulated using VectorBT with shared cash, transaction fees, and slippage, producing portfolio returns, Sharpe ratio, drawdown, turnover, exposure, and IC diagnostics.
