# Pipeline

The code behind the thesis experiments. For the project overview, setup and results, see the [root README](../README.md).

## Flow

```text
run_pipeline.py
  │
  ├─ data/             load alphas and prices, build the Date × ticker panel and target
  ├─ features/         scaling, PCA, Factor Analysis, GKX autoencoder, FactorVAE
  ├─ modeling/         LightGBM and CatBoost wrappers, Optuna tuning
  ├─ walking_forward/  yearly train, tune and predict loop
  ├─ Backtester/       long/short portfolio simulation with vectorbt
  └─ evaluation/       IC metrics and statistical tests
```

The target for signal date `t` is `Adjusted_Close[t+2] / Adjusted_Close[t+1] - 1`. The backtester shifts weights by one row, so a signal dated `t` is traded at the next adjusted close.

## Main files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Runs the experiments listed in `DEFAULT_EXPERIMENTS` |
| `experiment_config.py` | Experiments, search grids, profiles and output folder names |
| `Backtester/run_backtester.py` | Reruns backtests from saved predictions |
| `Backtester/build_annualized_metrics.py` | Writes corrected annualized return and volatility files |
| `build_statistical_inputs.py` | Collects daily IC and return series for statistical tests |

## Prediction sources

| `prediction_source` | Meaning |
|---|---|
| `tree` | A tree model ranks stocks using raw alphas, PCA factors or FA factors |
| `neural_expected_return` | No tree. Stocks are ranked directly by the neural model's expected return |
| `neural_then_tree` | A tree model is trained on the outputs of the selected neural model |

`neural_then_tree` profiles reuse the neural settings saved by a standalone run. The folder names are listed in `GKX_AE_REUSE_ARTIFACTS` and `FACTORVAE_REUSE_ARTIFACTS` in `experiment_config.py`, and those folders must exist inside `Pipeline/`.

## Output folders

Each run writes two folders inside `Pipeline/`:

```text
Results_Pipeline_<model>_<task>_<profile>_<scaling>_<metric>_<selection>_<universe>/
    <period>/predictions_<period>.csv
    <period>/daily_oos_ic_<period>.csv
    <period>/prediction_metrics_<period>.csv
    <period>/ic_raw_features_<period>.csv
    <period>/tuning_metrics_<period>.csv           (tree runs)
    <period>/neural_tuning_metrics_<period>.csv    (standalone neural runs)
    <period>/best_<model>_<period>/                (neural runs: best_params.json, state_dict.pt, predictions)
    daily_oos_ic_series.csv

portfolio_metrics_<model>_<task>_<profile>_<scaling>_<metric>_<selection>_<universe>/
    long_short_final_oos_metrics.csv
    long_short_time_series.csv
    long_short_portfolio_performance_results.csv
    long_short_portfolio_cumulative_returns.html
    vbt_generated_positions.csv
```

Dates in prediction files use `DD-MM-YYYY`.

## Documentation

| Document | Covers |
|---|---|
| [README.md](README.md) | Each pipeline phase and its tests, in depth |
| [SIGNAL_TO_TRADES_README.md](SIGNAL_TO_TRADES_README.md) | How a signal becomes stock weights |
| [Tuning_Architecture_README.md](Tuning_Architecture_README.md) | Hyperparameter tuning design |
| [data/README.md](data/README.md) | Data layout, target alignment and survivorship bias |
| [Backtester/README.md](Backtester/README.md) | Backtest logic and the 365-day annualization |
| [features/README.md](features/README.md) | Feature extraction |
| [features/GKX_VALIDATION_README.md](features/GKX_VALIDATION_README.md) | GKX validation |
| [modeling/README.md](modeling/README.md) | Tree models |
| [walking_forward/README.md](walking_forward/README.md) | Walk-forward modules |
| [evaluation/README.md](evaluation/README.md) | Metrics and statistical tests |
