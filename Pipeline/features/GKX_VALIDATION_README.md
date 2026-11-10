# GKX Validation Rank IC

## What Changed

GKX conditional autoencoder training still uses the original pricing loss:

```text
r_i,t = beta_i,t-1 * f_t
```

The model learns:

```text
beta_i,t-1 = beta_net(characteristics_i,t-1)
f_t = factor_net(managed_portfolio_t)
```

The important change is the validation Rank IC used for early stopping.

Previously, validation Rank IC used:

```text
beta_val * factor_val
```

where `factor_val` came from validation managed portfolios. Those managed portfolios require validation realized returns, so this measured validation pricing fit, not a tradable expected-return forecast.

Now, validation Rank IC uses:

```text
beta_val * train_factor_forecast
```

where:

```text
beta_val = beta_net(validation_features)
train_factor_forecast = average factor_net(train_managed_portfolio_t) over train dates only
```

This matches the tradable `gkx_expected_return` signal used later in the no-tree backtest.

## Why

The old validation IC could look very large because validation-period factors were inferred from validation realized returns. That is valid as an asset-pricing diagnostic, but it is too optimistic for forecasting.

The new validation IC answers the forecast question:

```text
Using only training-period factor information, do validation betas rank future validation returns?
```

This should make the printed early-stopping IC closer to the final `gkx_expected_return` validation and OOS behavior.

## Output Columns

GKX still outputs:

```text
gkx_expected_return
ae_component_*
gkx_factor_forecast_*
```

For `prediction_source="neural_expected_return"`, the final prediction is:

```text
gkx_expected_return
```

The `ae_component_*` columns are the learned conditional betas. The `gkx_factor_forecast_*` columns are the train-window factor forecast used to convert those betas into expected returns.

## Training History

The training history now keeps both validation diagnostics:

```text
val_forecast_rank_ic
val_pricing_rank_ic
```

For backward compatibility, `val_rank_ic` is the same as `val_forecast_rank_ic` and is the metric used for scheduler updates and early stopping.
