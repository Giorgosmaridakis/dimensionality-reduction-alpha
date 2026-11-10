"""GKX autoencoder feature extraction helpers."""

import numpy as np
import pandas as pd
import torch

try:
    from ..modeling.autoencoder.GKXAutoencoder import (
        GKXConditionalAutoencoder,
        compute_managed_portfolios,
        expand_managed_portfolio,
        train_autoencoder_with_early_stopping,
    )
except (ImportError, ValueError):
    try:
        from modeling.autoencoder.GKXAutoencoder import (
            GKXConditionalAutoencoder,
            compute_managed_portfolios,
            expand_managed_portfolio,
            train_autoencoder_with_early_stopping,
        )
    except (ImportError, ValueError):
        GKXConditionalAutoencoder = None
        compute_managed_portfolios = None
        expand_managed_portfolio = None
        train_autoencoder_with_early_stopping = None

class GKXAutoencoderMixin:
    def fit_transform_autoencoder(
        self,
        df,
        feature_cols,
        latent_dim=10,
        hidden_dims=None,
        target=None,
        dates=None,
        tickers=None,
        l1_lambda=0.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=64,
        max_epochs=200,
        patience=10,
        random_seed=42,
        keep_original_features=False,
        progress_bar=None,
    ):
        """
        Fits a GKX conditional autoencoder on realized returns.

        The returned ae_component_* columns are conditional betas learned from
        characteristics. Expected return uses those betas with a fixed
        training-window factor forecast.
        """
        if GKXConditionalAutoencoder is None:
            raise ImportError("GKXConditionalAutoencoder could not be imported. Ensure GKXAutoencoder.py is present.")
        if target is None:
            raise ValueError("fit_transform_autoencoder now requires target returns for GKX conditional autoencoder training.")

        self.ae_feature_cols = feature_cols
        self.keep_original_features_ae = keep_original_features
        df_ae = df.copy()

        X_full = df_ae[feature_cols].fillna(0).values
        y_full = np.asarray(target, dtype=np.float32).reshape(-1)
        date_values = self._resolve_dates_for_autoencoder(df_ae, dates)
        train_mask, val_mask = self._date_train_val_masks(date_values, val_fraction=0.2)
        X_train, X_val = X_full[train_mask], X_full[val_mask]
        y_train, y_val = y_full[train_mask], y_full[val_mask]
        train_dates = np.asarray(date_values)[train_mask]
        val_dates = np.asarray(date_values)[val_mask]

        date_codes = self._encode_dates_for_autoencoder(date_values)
        train_date_codes, val_date_codes = date_codes[train_mask], date_codes[val_mask]
        X_managed_train = self._build_managed_portfolios_by_date(X_train, y_train, train_dates)
        X_managed_val = self._build_managed_portfolios_by_date(X_val, y_val, val_dates)

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.autoencoder = GKXConditionalAutoencoder(
            input_dim=X_full.shape[1],
            latent_dim=latent_dim,
            managed_portfolio_dim=X_managed_train.shape[1],
            hidden_dims=hidden_dims,
        )

        self.autoencoder, self.ae_history = train_autoencoder_with_early_stopping(
            model=self.autoencoder,
            X_train=X_train,
            X_managed_train=X_managed_train,
            y_train=y_train,
            X_val=X_val,
            X_managed_val=X_managed_val,
            y_val=y_val,
            date_codes_train=train_date_codes,
            date_codes_val=val_date_codes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_seed=random_seed,
            progress_bar=progress_bar,
        )

        self.autoencoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_full, dtype=torch.float32, device=device)
            latent_features = self.autoencoder.encode(X_tensor).cpu().numpy()
        self.autoencoder.to("cpu")

        self.ae_col_names = [f"ae_component_{i + 1}" for i in range(latent_dim)]
        self.ae_factor_col_names = [f"gkx_factor_forecast_{i + 1}" for i in range(latent_dim)]
        self.ae_expected_return_col = "gkx_expected_return"
        self.ae_factor_forecast_ = self._estimate_gkx_factor_forecast(X_full, y_full, date_values)
        expected_return = latent_features.dot(self.ae_factor_forecast_)
        self.ae_pricing_col_names = (
            [self.ae_expected_return_col] +
            self.ae_col_names +
            self.ae_factor_col_names
        )

        if not self.keep_original_features_ae:
            df_ae = df_ae.drop(columns=feature_cols)

        df_ae[self.ae_expected_return_col] = expected_return
        for i, col_name in enumerate(self.ae_col_names):
            df_ae[col_name] = latent_features[:, i]
        for i, col_name in enumerate(self.ae_factor_col_names):
            df_ae[col_name] = self.ae_factor_forecast_[i]

        return df_ae

    def _estimate_gkx_factor_forecast(self, X_full, y_full, dates):
        managed = self._build_managed_portfolios_by_date(X_full, y_full, dates)
        self.autoencoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        with torch.no_grad():
            managed_tensor = torch.as_tensor(managed, dtype=torch.float32, device=device)
            factor_values = self.autoencoder.factor_net(managed_tensor).cpu().numpy()
        self.autoencoder.to("cpu")

        factor_df = pd.DataFrame(factor_values)
        factor_df["_Date"] = pd.to_datetime(np.asarray(dates))
        by_date = factor_df.groupby("_Date", sort=True).mean()
        return by_date.to_numpy(dtype=np.float32).mean(axis=0)

    @staticmethod
    def _resolve_dates_for_autoencoder(df, dates=None):
        if dates is not None:
            return pd.to_datetime(np.asarray(dates))
        if "Date" in df.columns:
            return pd.to_datetime(df["Date"].to_numpy())
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.to_datetime(df.index.to_numpy())
        return pd.Series(np.zeros(len(df), dtype=int))

    @staticmethod
    def _date_train_val_masks(dates, val_fraction=0.2):
        date_series = pd.Series(pd.to_datetime(np.asarray(dates)))
        unique_dates = date_series.drop_duplicates().to_numpy()

        if len(unique_dates) < 2:
            raise ValueError(
                "GKX conditional autoencoder requires at least two unique dates "
                "for date-based train/validation splitting."
            )

        split_idx = max(1, int(len(unique_dates) * (1.0 - val_fraction)))
        if split_idx >= len(unique_dates):
            split_idx = len(unique_dates) - 1

        train_dates = set(unique_dates[:split_idx])
        train_mask = date_series.isin(train_dates).to_numpy()
        return train_mask, ~train_mask

    @staticmethod
    def _build_managed_portfolios_by_date(X_full, y_full, dates):
        date_series = pd.Series(dates)
        groups = [
            np.asarray(list(idx), dtype=int)
            for _, idx in date_series.groupby(date_series, sort=False).groups.items()
        ]
        if not groups:
            raise ValueError("Cannot build managed portfolios for an empty dataset.")

        first_x = compute_managed_portfolios(X_full[groups[0]], y_full[groups[0]])
        managed = np.empty((len(X_full), int(first_x.numel())), dtype=np.float32)
        managed[groups[0]] = expand_managed_portfolio(first_x, len(groups[0])).numpy()

        for positions in groups[1:]:
            x_t = compute_managed_portfolios(X_full[positions], y_full[positions])
            managed[positions] = expand_managed_portfolio(x_t, len(positions)).numpy()
        return managed

    @staticmethod
    def _encode_dates_for_autoencoder(dates):
        date_values = pd.to_datetime(np.asarray(dates))
        unique_dates = pd.Index(date_values).drop_duplicates().sort_values()
        date_map = {date: idx for idx, date in enumerate(unique_dates)}
        return np.asarray([date_map[date] for date in date_values], dtype=np.int64)

    def transform_autoencoder(self, df, keep_original_features=None):
        """Transforms features using fitted GKX conditional betas."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not fitted. Call fit_transform_autoencoder first.")

        df_ae = df.copy()
        X_test = df_ae[self.ae_feature_cols].fillna(0).values

        self.autoencoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            latent_features = self.autoencoder.encode(X_tensor).cpu().numpy()
        self.autoencoder.to("cpu")

        keep_orig = getattr(self, "keep_original_features_ae", False) if keep_original_features is None else keep_original_features
        if not keep_orig:
            df_ae = df_ae.drop(columns=self.ae_feature_cols)

        factor_forecast = getattr(self, "ae_factor_forecast_", None)
        if factor_forecast is None:
            raise ValueError("GKX factor forecast missing. Refit the autoencoder before transforming.")

        df_ae[getattr(self, "ae_expected_return_col", "gkx_expected_return")] = latent_features.dot(factor_forecast)
        for i, col_name in enumerate(self.ae_col_names):
            df_ae[col_name] = latent_features[:, i]
        for i, col_name in enumerate(getattr(self, "ae_factor_col_names", [])):
            df_ae[col_name] = factor_forecast[i]

        return df_ae
