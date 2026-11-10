"""FactorVAE feature extraction helpers."""

import numpy as np
import pandas as pd
import torch

try:
    from ..modeling.Factor_vae import FactorVAEModel, train_factor_vae_with_early_stopping
except (ImportError, ValueError):
    try:
        from modeling.Factor_vae import FactorVAEModel, train_factor_vae_with_early_stopping
    except (ImportError, ValueError):
        FactorVAEModel, train_factor_vae_with_early_stopping = None, None


class FactorVAEExtractorMixin:
    def _prepare_factor_vae_frame(self, df, feature_cols, dates=None, tickers=None, target=None, ticker_col="ticker"):
        """Create the Date/ticker/feature frame required by FactorVAE sequence building."""
        frame = df.copy()

        if "Date" not in frame.columns:
            if dates is not None:
                frame["Date"] = pd.to_datetime(np.asarray(dates))
            elif isinstance(frame.index, pd.DatetimeIndex):
                frame["Date"] = pd.to_datetime(frame.index)
            else:
                raise ValueError("FactorVAE requires a Date column, DatetimeIndex, or dates argument.")

        if ticker_col not in frame.columns:
            if tickers is None:
                raise ValueError("FactorVAE requires ticker metadata via a ticker column or tickers argument.")
            frame[ticker_col] = np.asarray(tickers)

        if target is not None:
            frame["_factorvae_target"] = np.asarray(target)

        required_cols = ["Date", ticker_col] + list(feature_cols)
        if target is not None:
            required_cols.append("_factorvae_target")

        frame = frame[required_cols].copy()
        frame = frame.reset_index(drop=True)
        frame["Date"] = pd.to_datetime(frame["Date"])
        return frame

    @staticmethod
    def _build_factor_vae_sequences(frame, feature_cols, seq_len, ticker_col="ticker"):
        sequences = []
        output_positions = []

        ordered = frame.copy().reset_index(drop=True)
        ordered["_original_index"] = np.arange(len(ordered))
        ordered = ordered.sort_values([ticker_col, "Date", "_original_index"]).reset_index(drop=True)

        for _, group in ordered.groupby(ticker_col, sort=False):
            values = group[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            for pos in range(len(group)):
                start = max(0, pos - seq_len + 1)
                seq = values[start:pos + 1]
                if len(seq) < seq_len:
                    pad = np.zeros((seq_len - len(seq), len(feature_cols)), dtype=np.float32)
                    seq = np.vstack([pad, seq])
                sequences.append(seq)
                output_positions.append(group.index[pos])

        sequence_df = ordered.loc[output_positions].copy()
        X_seq = (
            np.stack(sequences).astype(np.float32)
            if sequences
            else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
        )

        restore_order = np.argsort(sequence_df["_original_index"].to_numpy())
        X_seq = X_seq[restore_order]
        sequence_df = sequence_df.iloc[restore_order].reset_index(drop=True)
        return X_seq, sequence_df

    def _factor_vae_features_from_sequences(self, X_seq, sequence_df=None, batch_size=256):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_vae_model.to(device)
        self.factor_vae_model.eval()

        if sequence_df is None:
            group_indices = [np.arange(len(X_seq))]
        else:
            group_indices = [
                positions.to_numpy(dtype=int)
                for _, positions in sequence_df.groupby("Date", sort=False).groups.items()
            ]

        features = np.empty((len(X_seq), len(self.factor_vae_col_names)), dtype=np.float32)

        with torch.no_grad():
            for group_idx in group_indices:
                batch = torch.as_tensor(X_seq[group_idx], dtype=torch.float32, device=device)
                outputs = self.factor_vae_model.encode_prior_features(batch)

                beta = outputs["beta"].cpu().numpy()
                alpha_mu = outputs["alpha_mu"].cpu().numpy().reshape(-1, 1)
                alpha_sigma = outputs["alpha_sigma"].cpu().numpy().reshape(-1, 1)
                expected_return = outputs["expected_return"].cpu().numpy().reshape(-1, 1)

                features[group_idx] = np.hstack([
                    beta,
                    alpha_mu,
                    alpha_sigma,
                    expected_return,
                ])

        self.factor_vae_model.to("cpu")
        return features

    def fit_transform_factor_vae(
        self,
        df,
        feature_cols,
        target=None,
        dates=None,
        tickers=None,
        ticker_col="ticker",
        seq_len=20,
        hidden_size=32,
        num_factors=8,
        num_portfolios=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=200,
        patience=10,
        random_seed=42,
        keep_original_features=False,
        progress_bar=None,
    ):
        """Fit FactorVAE with training targets and return prior-path pricing outputs."""
        if FactorVAEModel is None:
            raise ImportError("FactorVAEModel could not be imported. Ensure Factor_vae is present.")
        if target is None:
            raise ValueError("fit_transform_factor_vae requires training targets.")

        self.factor_vae_feature_cols = list(feature_cols)
        self.factor_vae_ticker_col = ticker_col
        self.factor_vae_seq_len = seq_len
        self.keep_original_features_factor_vae = keep_original_features
        self.factor_vae_batch_size = batch_size

        frame = self._prepare_factor_vae_frame(
            df,
            feature_cols,
            dates=dates,
            tickers=tickers,
            target=target,
            ticker_col=ticker_col,
        )
        history_len = max(seq_len - 1, 0)
        self.factor_vae_history_frame = (
            frame.drop(columns=["_factorvae_target"], errors="ignore")
                 .sort_values([ticker_col, "Date"])
                 .groupby(ticker_col, group_keys=False)
                 .tail(history_len)
                 .reset_index(drop=True)
        ) if history_len > 0 else frame.iloc[0:0].drop(columns=["_factorvae_target"], errors="ignore").copy()

        X_seq, sequence_df = self._build_factor_vae_sequences(frame, feature_cols, seq_len, ticker_col=ticker_col)
        y_full = sequence_df["_factorvae_target"].to_numpy(dtype=np.float32)
        if len(X_seq) < 2:
            raise ValueError("FactorVAE requires at least two samples to create train/validation splits.")

        date_groups = sequence_df.groupby("Date", sort=True).groups
        unique_dates = sorted(list(date_groups.keys()))
        split_i = max(1, int(len(unique_dates) * 0.8))
        if split_i >= len(unique_dates):
            split_i = len(unique_dates) - 1

        train_dates = set(unique_dates[:split_i])
        val_dates = set(unique_dates[split_i:])
        train_groups = [
            positions.to_numpy(dtype=int)
            for date, positions in date_groups.items()
            if date in train_dates
        ]
        val_groups = [
            positions.to_numpy(dtype=int)
            for date, positions in date_groups.items()
            if date in val_dates
        ]

        self.factor_vae_model = FactorVAEModel(
            input_dim=len(feature_cols),
            hidden_size=hidden_size,
            num_factors=num_factors,
            num_portfolios=num_portfolios,
        )
        self.factor_vae_model, self.factor_vae_history = train_factor_vae_with_early_stopping(
            model=self.factor_vae_model,
            X_seq=X_seq,
            y_full=y_full,
            train_groups=train_groups,
            val_groups=val_groups,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            random_seed=random_seed,
            progress_bar=progress_bar,
        )

        self.factor_vae_col_names = (
            [f"factorvae_beta_{i + 1}" for i in range(num_factors)] +
            ["factorvae_alpha_mu", "factorvae_alpha_sigma", "factorvae_expected_return"]
        )
        factor_features = self._factor_vae_features_from_sequences(
            X_seq,
            sequence_df=sequence_df,
            batch_size=batch_size,
        )

        df_factor = df.copy()
        if not keep_original_features:
            df_factor = df_factor.drop(columns=feature_cols)

        for i, col_name in enumerate(self.factor_vae_col_names):
            df_factor[col_name] = factor_features[:, i]

        return df_factor

    def transform_factor_vae(self, df, dates=None, tickers=None, keep_original_features=None):
        """Transform data using the fitted FactorVAE prior path only."""
        if self.factor_vae_model is None:
            raise ValueError("FactorVAE not fitted. Call fit_transform_factor_vae first.")

        frame = self._prepare_factor_vae_frame(
            df,
            self.factor_vae_feature_cols,
            dates=dates,
            tickers=tickers,
            target=None,
            ticker_col=self.factor_vae_ticker_col,
        )
        frame["_factorvae_output"] = True
        history_frame = getattr(self, "factor_vae_history_frame", None)
        if history_frame is not None and not history_frame.empty:
            history_frame = history_frame.copy()
            history_frame["_factorvae_output"] = False
            sequence_input = pd.concat([history_frame, frame], ignore_index=True)
        else:
            sequence_input = frame

        X_seq, sequence_df = self._build_factor_vae_sequences(
            sequence_input,
            self.factor_vae_feature_cols,
            self.factor_vae_seq_len,
            ticker_col=self.factor_vae_ticker_col,
        )
        factor_features = self._factor_vae_features_from_sequences(
            X_seq,
            sequence_df=sequence_df,
            batch_size=self.factor_vae_batch_size,
        )
        output_mask = sequence_df["_factorvae_output"].to_numpy(dtype=bool)
        factor_features = factor_features[output_mask]

        df_factor = df.copy()
        keep_orig = (
            getattr(self, "keep_original_features_factor_vae", False)
            if keep_original_features is None
            else keep_original_features
        )
        if not keep_orig:
            df_factor = df_factor.drop(columns=self.factor_vae_feature_cols)

        for i, col_name in enumerate(self.factor_vae_col_names):
            df_factor[col_name] = factor_features[:, i]

        return df_factor
