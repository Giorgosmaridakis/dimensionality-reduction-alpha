# Hyperparameter Tuning and Architecture Guide

This document explains how hyperparameter tuning, model selection, neural expected-return tuning, and Optuna optimization are structured in `Pipeline`.

The current implementation uses nested walk-forward validation. For every outer test period, the model is tuned only inside the current training window. The selected parameters are then applied to final preprocessing and fitting for that same outer period.

The tuning system has two layers:

1. Tree and feature-reduction tuning through `modeling/tuning/`.
2. Neural expected-return tuning and reuse through `walking_forward/neural_tuning.py` and `walking_forward/neural_artifacts.py`.

---

## 1. Experiment-Level Model Selection

The model type and feature branch are not trial-level hyperparameters. They are experiment/profile choices defined in `experiment_config.py`.

Supported tree families are:

```python
lightgbm
catboost
```

The tree task can be ranking or regression, with ranking using model-specific ranking losses:

| Model | Ranking Loss |
|---|---|
| LightGBM | `lambdarank` |
| CatBoost | `YetiRank` |

Feature/profile choices are separate experiments:

```text
Baseline
PCA-Only
FA-Only
GKX-AE expected return
FactorVAE expected return
GKX-AE then tree
FactorVAE then tree
```

This keeps comparisons clean: each profile gets its own parameter grid, tuning outputs, and walk-forward result directory.

---

## 2. Prediction Sources

`WalkingForwardForecaster.forecast(...)` supports three prediction sources:

| `prediction_source` | Meaning |
|---|---|
| `tree` | Fit a tree directly on raw or reduced alpha features. This covers Baseline, PCA, FA, and legacy neural-feature-plus-tree tuning. |
| `neural_expected_return` | Do not train a tree. Rank directly by the neural pricing output: `gkx_expected_return` or `factorvae_expected_return`. |
| `neural_then_tree` | Tune or reuse a neural expected-return model first, freeze its pricing outputs, then tune and train a tree only on those neural pricing outputs. |

For neural modes, exactly one of `use_autoencoder=True` or `use_factor_vae=True` must be selected.

In `neural_then_tree`, raw Zura alpha columns are intentionally excluded from the tree. The final tree receives only neural pricing-equation outputs:

```text
GKX-AE:
gkx_expected_return
ae_component_*
gkx_factor_forecast_*

FactorVAE:
factorvae_expected_return
factorvae_beta_*
factorvae_alpha_mu
factorvae_alpha_sigma
```

---

## 3. Current Search Spaces

The base tree grids are defined in `experiment_config.py`.

### LightGBM

```python
{
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "reg_lambda": [3.0, 5.0, 10.0],
    "n_estimators": [300, 500],
    "min_child_samples": [25, 50, 100],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
```

Grid size: `648`.

### CatBoost

```python
{
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [4, 6],
    "l2_leaf_reg": [3, 7, 10],
    "iterations": [300, 500],
    "colsample_bylevel": [1.0],
}
```

Grid size: `36`.

### PCA and FA

PCA and FA add one component-count parameter:

```python
"pca__n_components": valid_components
"fa__n_components": valid_components
```

`valid_components` is filtered from:

```python
[5, 8, 10, 12, 15, 20]
```

Only component counts smaller than the number of available features are kept.

### GKX-AE

GKX-AE adds:

```python
{
    "ae__latent_dim": valid_components,
    "ae__learning_rate": [5e-4, 1e-4, 5e-5, 1e-5],
    "ae__weight_decay": [1e-6, 1e-5, 1e-4],
    "ae__l1_lambda": [0.0, 1e-5, 1e-4],
    "ae__max_epochs": [200],
    "ae__patience": [20],
}
```

For six valid component counts, the GKX-AE neural-only grid size is `216`.

### FactorVAE

FactorVAE adds:

```python
{
    "factorvae__seq_len": [10, 20],
    "factorvae__hidden_size": [32, 64, 128],
    "factorvae__num_factors": valid_components,
    "factorvae__num_portfolios": [8, 16, 32],
    "factorvae__learning_rate": [5e-4, 1e-4, 5e-5, 1e-5],
    "factorvae__weight_decay": [1e-6, 1e-5, 1e-4],
    "factorvae__max_epochs": [200],
    "factorvae__patience": [25],
}
```

For six valid component counts, the FactorVAE neural-only grid size is `1,296`.

---

## 4. Trial Budget

Tree and feature-reduction profiles use:

```python
def sqrt_grid_trials(param_grid, cap=50):
    return min(cap, max(1, math.ceil(math.sqrt(grid_size(param_grid))) + 10))
```

The active `neural_then_tree` reuse profiles use:

```python
"optuna_trials": 40
```

Trials actually used in the saved thesis runs:

| Profiles | Trials per period |
|---|---:|
| LightGBM: Baseline, PCA, FA and neural + LightGBM (both universes) | 40 |
| CatBoost: ETF Baseline, PCA and FA | 40 |
| CatBoost: S&P 500 Baseline | 16 |
| CatBoost: S&P 500 PCA and FA | 25 |
| Neural + CatBoost (both universes) | 36, the full CatBoost grid |
| Standalone GKX-AE and FactorVAE | 40 neural candidates |

The S&P 500 CatBoost runs follow the square-root rule. The LightGBM and ETF CatBoost runs used a fixed budget of 40, so rerunning them with the current code gives a different number of trials: 36 for LightGBM Baseline and 50 for LightGBM PCA and FA.

The square-root rule is a practical budget heuristic:

1. Full grid search is too expensive for large tree and neural grids.
2. Larger grids receive more trials, but the cap prevents runaway cost.
3. The `+10` term gives Optuna enough early observations before TPE exploitation matters.

---

## 5. Tree Tuning Flow

Tree tuning is implemented by `HyperparameterTuner` in `modeling/tuning/tuner.py`.

For each outer walk-forward period:

```text
1. Take only the current outer training window.
2. Split unique dates with TimeSeriesSplit(n_splits=3, gap=embargo_days), where embargo_days=1.
3. For each candidate parameter combination:
   a. Fit scaling/PCA/FA/neural preprocessing only on the inner train fold.
   b. Transform the inner validation fold.
   c. Fit a fresh temporary tree model.
   d. Score the validation fold.
4. Average validation fold scores.
5. Save one row per trial to tuning_metrics_<period>.csv.
6. Update the real model wrapper with the best parameters.
```

This keeps the outer test period completely hidden during tuning.

---

## 6. Neural Expected-Return Tuning

Neural expected-return tuning is implemented in `walking_forward/neural_tuning.py`.

The neural grid is split away from the tree grid by prefix:

```text
ae__*          -> GKX-AE neural parameters
factorvae__*   -> FactorVAE neural parameters
tree params    -> downstream tree parameters
```

When `prediction_source` is `neural_expected_return` or `neural_then_tree`, the forecaster first tunes or loads neural parameters. The selected neural model is then used during final preprocessing to create the neural pricing outputs for the outer test period.

Supported `neural_selection_metric` aliases are:

| Input Alias | Internal Metric |
|---|---|
| `rank_ic`, `validation_rank_ic` | `validation_rank_ic` |
| `in_sample_rank_ic` | `in_sample_rank_ic` |
| `sharpe`, `sharp`, `validation_sharpe`, `validation_sharp` | `validation_sharpe` |

For `validation_rank_ic` and `validation_sharpe`, neural candidates are evaluated on inner time-series validation folds. For `in_sample_rank_ic`, the model is scored once on the current training window.

Neural tuning writes, inside each period folder:

```text
neural_tuning_metrics_<period>.csv
best_gkx_ae_<period>/gkx_ae_tuning_metrics_<period>.csv
best_factorvae_<period>/factorvae_tuning_metrics_<period>.csv
```

Standalone neural candidates are drawn from the neural grid at random without replacement (`numpy` seed 42), 40 per period in the thesis runs.

The selected neural parameters are also saved as best-neural artifacts for reuse.

---

## 7. Reusing Best Neural Artifacts

`active_feature_reduction_configs(...)` builds reuse profiles for both neural models, both tree models and both normalizations:

```python
GKX-AE-Then-<LightGBM|CatBoost>-ReuseBest-<Standard|CrossSectional>
FactorVAE-Then-<LightGBM|CatBoost>-ReuseBest-<Standard|CrossSectional>
```

All eight were run for each universe. As committed, `DEFAULT_EXPERIMENTS` runs the four S&P 500 `Standard` ones.

For these profiles:

```text
1. Load best neural parameters (`best_params.json`) for the current outer period.
2. Save a local reuse manifest (`reuse_manifest.json`) in the new result directory.
3. Recreate frozen neural pricing outputs using those parameters.
4. Tune the downstream tree on the frozen neural pricing outputs only.
5. Train/predict with the selected downstream tree.
```

The reuse source directories are configured in:

```python
GKX_AE_REUSE_ARTIFACTS
FACTORVAE_REUSE_ARTIFACTS
```

This separates expensive neural hyperparameter discovery from downstream tree comparison. LightGBM and CatBoost can therefore be compared on the same reused neural representation.

---

## 8. Optuna Search and Duplicate Control

Tree tuning uses Optuna when `use_optuna=True` and Optuna is installed. Otherwise it falls back to exhaustive grid search.

Optuna samples categorical choices from the explicit grid:

```python
trial.suggest_categorical(key, values)
```

The sampler is:

```python
optuna.samplers.TPESampler(seed=42)
```

The search space is finite and categorical, so pure TPE can suggest the same complete configuration more than once. V2 keeps the duplicate-controlled hybrid:

```text
1. TPE suggests a configuration.
2. If it is new, evaluate it.
3. If it was already evaluated, prune the duplicate trial.
4. Queue one unused grid configuration into the same Optuna study.
5. Continue until the target number of unique completed trials is reached,
   the grid is exhausted, or the attempt limit is reached.
```

This is implemented in `modeling/tuning/search.py` with:

```python
seen_param_keys
queued_param_keys
study.enqueue_trial(fallback_params)
```

The log reports:

```text
Optuna unique completed trials: X/Y
attempts=...
unique sampled=...
duplicates=...
fallbacks queued=...
total grid=...
```

This is not standard Optuna TPE. It is a project-specific safeguard for finite categorical grids and expensive walk-forward validation.

---

## 9. Pruning

Tree-only trials use:

```python
NopPruner()
```

Neural preprocessing inside the tree tuner enables:

```python
MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=2,
    interval_steps=1,
)
```

During neural-feature tree tuning, partial fold scores are reported to Optuna. A weak neural trial can be pruned after at least two completed folds.

The separate `walking_forward/neural_tuning.py` expected-return tuner uses deterministic grid enumeration or random sampling without replacement when `use_optuna=True` and `optuna_trials` is smaller than the neural grid. It does not use Optuna TPE directly.

---

## 10. Tuning Metrics

Tree tuning supports:

| `tuning_metric` | Behavior |
|---|---|
| `ic` | Mean daily cross-sectional IC. Ranking uses Spearman correlation; regression uses Pearson correlation. |
| `sharpe` / `sharp` | Runs validation predictions through the same long-short backtester convention used for OOS evaluation. |
| other | Falls back to the tree library validation metric where available. |

The default output naming normalizes:

```python
tuning_metric = config.get("tuning_metric", "sharp").lower()
```

For neural expected-return selection, `sharp` is normalized to `validation_sharpe`.

---

## 11. Leakage Controls

The V2 tuning architecture has several explicit leakage controls:

1. Outer test periods are never passed into hyperparameter tuning.
2. Inner validation folds are date-based, not row-shuffled.
3. `embargo_days` is passed as the `TimeSeriesSplit` gap.
4. Scaling, PCA, FA, GKX-AE, and FactorVAE are fit only on each inner train fold.
5. Validation Sharpe prices are capped to the current outer training window.
6. Reused neural artifacts are loaded per period and re-applied inside the current walk-forward period.

---

## 12. Output Files

Important tuning outputs are:

| File | Meaning |
|---|---|
| `tuning_metrics_<period>.csv` | Tree or feature-reduction tuning results for one outer period. |
| `deep_learning_training_history_<period>.csv` | Per-epoch neural training history collected during neural-feature tree tuning. |
| `neural_tuning_metrics_<period>.csv` | Neural expected-return tuning results. |
| `best_gkx_ae_<period>/gkx_ae_tuning_metrics_<period>.csv` | GKX-AE tuning artifact copy. |
| `best_factorvae_<period>/factorvae_tuning_metrics_<period>.csv` | FactorVAE tuning artifact copy. |
| `best_<model>_<period>/best_params.json` | Selected neural parameters for the period. |
| `best_<model>_<period>/reuse_manifest.json` | Records which prior artifact directory supplied the neural parameters. |

The tuning result rows include validation score, fold scores, fold completion flags, pruning flags where applicable, and the tested hyperparameters.

---

## 13. Methodology Statement for the Thesis

> Hyperparameter optimization was performed inside each walk-forward training window using date-based inner validation folds with an embargo gap. Tree models and feature-reduction branches were tuned separately from experiment-level model selection. For tree tuning, Optuna's Tree-structured Parzen Estimator sampled from a finite categorical grid, while a custom duplicate-control mechanism skipped repeated configurations and enqueued unused grid points into the same study. Neural expected-return models were tuned separately over GKX-AE or FactorVAE hyperparameters using validation rank IC or validation Sharpe, then either used directly as signals or frozen as pricing-equation outputs for a downstream tree. In reuse experiments, previously selected neural parameters were loaded per period so that downstream LightGBM and CatBoost models could be compared on the same neural representation. All preprocessing and neural feature extraction were fit within the relevant training fold only, preventing validation and test-period leakage.

---

## 14. Supporting Literature

| Paper | Role in Methodology |
|---|---|
| Bergstra, Bardenet, Bengio, and Kegl (2011), *Algorithms for Hyper-Parameter Optimization* | Introduces TPE and motivates adaptive sequential hyperparameter optimization. |
| Bergstra and Bengio (2012), *Random Search for Hyper-Parameter Optimization* | Shows why non-exhaustive search can be more efficient than full grid search when only some hyperparameters matter strongly. |
| Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework* | Provides the Optuna framework and `TPESampler` used in tree tuning. |
| Thornton et al. (2013), *Auto-WEKA* | Supports separating pipeline/model selection from inner hyperparameter optimization. |
| Bailey et al. (2014), *The Probability of Backtest Overfitting* | Motivates careful validation-budget use in noisy financial model selection. |
| Harvey, Liu, and Zhu (2016), *... and the Cross-Section of Expected Returns* | Motivates caution around multiple testing and factor discovery in cross-sectional return prediction. |
| Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine Learning* | Supports machine learning and rank-based predictive evaluation in empirical asset pricing. |
