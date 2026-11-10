from __future__ import annotations

import numpy as np
import pandas as pd

from features.neural_extractors import NeuralFeatureExtractor


def _factor_vae_panel(periods=18):
    rng = np.random.default_rng(321)
    feature_cols = [f"alpha_{i}" for i in range(3)]
    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        for date in pd.date_range("2020-01-01", periods=periods, freq="D"):
            values = rng.normal(size=len(feature_cols))
            row = {"Date": date, "ticker": ticker, "Target": rng.normal()}
            row.update(dict(zip(feature_cols, values)))
            rows.append(row)
    return pd.DataFrame(rows), feature_cols


def test_transform_factor_vae_uses_prior_path_without_targets(monkeypatch):
    df, feature_cols = _factor_vae_panel()
    extractor = NeuralFeatureExtractor()
    extractor.fit_transform_factor_vae(
        df[["Date", "ticker"] + feature_cols],
        feature_cols,
        target=df["Target"],
        seq_len=4,
        hidden_size=5,
        num_factors=2,
        num_portfolios=3,
        batch_size=16,
        max_epochs=1,
        patience=1,
        keep_original_features=False,
    )

    def fail_if_posterior_path_is_used(*args, **kwargs):
        raise AssertionError("transform_factor_vae must not call FactorVAE.forward with realized returns")

    monkeypatch.setattr(extractor.factor_vae_model, "forward", fail_if_posterior_path_is_used)

    transformed = extractor.transform_factor_vae(
        df[["Date", "ticker"] + feature_cols],
        keep_original_features=False,
    )

    assert "Target" not in transformed.columns
    assert "factorvae_expected_return" in transformed.columns
    assert len(transformed) == len(df)


def test_factor_vae_sequences_never_include_future_rows():
    df, feature_cols = _factor_vae_panel(periods=6)
    extractor = NeuralFeatureExtractor()
    X_seq, sequence_df = extractor._build_factor_vae_sequences(df, feature_cols, seq_len=3)

    for row_number, row in sequence_df.iterrows():
        ticker = row["ticker"]
        output_date = row["Date"]
        ticker_history = (
            df.loc[(df["ticker"] == ticker) & (df["Date"] <= output_date), feature_cols]
              .tail(3)
              .to_numpy(dtype=np.float32)
        )
        expected = np.zeros((3, len(feature_cols)), dtype=np.float32)
        expected[-len(ticker_history):] = ticker_history
        np.testing.assert_array_equal(X_seq[row_number], expected)
