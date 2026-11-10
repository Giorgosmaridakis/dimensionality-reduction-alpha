# Modeling

This folder contains the model wrappers, hyperparameter tuner, and neural architectures used by the pipeline.

## Where It Enters From `run_pipeline.py`

`run_pipeline.py` creates:

```python
ml_model = Model(model_type=model_choice, task_type=task, loss_function=loss)
```

Then it passes that wrapper into:

```python
WalkingForwardForecaster(model_wrapper=ml_model, ...)
```

The forecaster uses `HyperparameterTuner` for nested validation and calls `Model.fit(...)` / `Model.predict(...)` for the final fold model.

## Main Files

- `trees.py` is a unified wrapper for LightGBM and CatBoost.
- `tuning/` performs 3-fold time-series hyperparameter tuning inside each walk-forward training window.
- `Factor_vae/FactorVAEModel.py` is the FactorVAE implementation used by the current pipeline.
- `autoencoder/GKXAutoencoder.py` contains the active GKX autoencoder implementation.

## Tree Tuning

Tree tuning uses the same time-series fold structure as neural tuning. For `tuning_metric="sharp"` or `"sharpe"`, the tuner:

1. predicts validation fold scores,
2. ranks assets,
3. runs a long/short validation backtest,
4. uses validation Sharpe as the fold score.

For `tuning_metric="ic"`, it uses daily validation IC.

## Neural-Then-Tree Inputs

When `prediction_source="neural_then_tree"`, the tree receives only neural pricing-equation outputs. Raw Zura alpha features are intentionally excluded to keep interpretation clean.

## Things To Watch

- `Model.fit(...)` filters out preprocessing parameters like `factorvae__*` before calling the tree library.
- Ranking tasks require date/query IDs so LightGBM and CatBoost know which assets belong to the same cross-section.
