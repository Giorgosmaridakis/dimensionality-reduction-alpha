# Evaluation

This folder owns model evaluation metrics. The walk-forward experiment engine
now lives in `../walking_forward/`.

## Where It Enters From `run_pipeline.py`

`run_pipeline.py` builds `full_dataset`, chooses a model/profile, then calls:

```python
WalkingForwardForecaster.forecast(...)
```

That call controls yearly train/test splits, feature preprocessing, neural tuning, tree tuning, prediction saving, and fold-level metrics.

## Main Files

- `metrics.py` computes IC and p-values for raw, decomposed, neural, and final model features.
- `statistical_tests.py` contains optional post-experiment tests: Newey-West IC statistics, Diebold-Mariano, block-bootstrap metric differences, multiple-testing corrections and a Sharpe ratio test. The Sharpe test uses an incorrect variance formula, so correct it before use (see the root README).

## Prediction Modes

`WalkingForwardForecaster.forecast(...)` supports:

- `prediction_source="tree"`: current tree-only or feature-reduction-plus-tree behavior.
- `prediction_source="neural_expected_return"`: no tree; directly ranks `factorvae_expected_return` or `gkx_expected_return`.
- `prediction_source="neural_then_tree"`: tune or reuse a neural model first, then train a tree only on neural pricing outputs.

## Neural Reuse

For faster experiments, `neural_then_tree` can reuse a previous expected-return run:

```python
reuse_neural_artifacts_from="Results_Pipeline_factorvae_expected_return_no_tree_..."
```

The reuse path loads `best_params.json` for the same period, refits the neural model on the current training window, regenerates aligned neural outputs, and then trains the tree. It does not blindly reuse old prediction rows.

## Key Saved Files

Per period:

- `predictions_<period>.csv`: final signal used by the main backtest.
- `daily_oos_ic_<period>.csv`: daily IC of final predictions.
- `prediction_metrics_<period>.csv`: fold IC and fold Sharpe summary.
- `ic_raw_features_<period>.csv`: raw feature IC.
- `ic_factorvae_features_<period>.csv` or similar: transformed feature IC.
- `best_factorvae_<period>/...`: neural params, neural expected-return predictions, IC, metrics, state dict, and reuse manifest when applicable.

## Things To Watch

- Final model fitting never uses test labels for early stopping.
- For direct neural mode, `Prediction` is the neural expected return.
- For neural-then-tree mode, original alpha features are excluded from the tree by default.
