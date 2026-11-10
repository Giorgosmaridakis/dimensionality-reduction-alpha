# Features

This folder transforms raw alpha features before modeling.

## Where It Enters From `run_pipeline.py`

`run_pipeline.py` passes booleans such as `use_pca`, `use_fa`, `use_ae`, and `use_factor_vae` into `WalkingForwardForecaster.forecast(...)`. The forecaster then calls these feature helpers during each walk-forward fold.

## Main Files

- `scaling.py` contains standard and cross-sectional z-score scaling.
- `decomposition.py` contains PCA and Factor Analysis feature compression.
- `neural_extractors.py` contains GKX autoencoder and FactorVAE feature extraction.
- `GKX_VALIDATION_README.md` explains why GKX early stopping now uses forecast Rank IC instead of validation pricing Rank IC.

## Neural Outputs Used By The New Experiments

For `FactorVAE`:

```text
factorvae_expected_return
factorvae_beta_*
factorvae_alpha_mu
factorvae_alpha_sigma
```

For `GKX-AE`:

```text
gkx_expected_return
ae_component_*
gkx_factor_forecast_*
```

Direct neural backtests rank the expected-return column. Neural-then-tree experiments train the tree only on the neural pricing outputs.

## Leakage Rules

- Scalers and decomposers are fit on the training fold and then applied to test.
- FactorVAE and GKX are fit on the training fold before transforming test.
- GKX expected return uses a factor forecast estimated from the training window, not test-period realized returns.

## Things To Watch

- `keep_original_features=False` is important for clean neural-only and neural-then-tree experiments.
- FactorVAE needs `Date` and `ticker` metadata to build per-ticker sequences and per-date batches.
- Neural training histories are saved by the evaluation layer, not directly from this folder.
