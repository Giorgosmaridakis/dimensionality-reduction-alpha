from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from features.neural_extractors import NeuralFeatureExtractor
from modeling.Factor_vae import FactorVAEModel


def test_factor_vae_forward_shapes():
    model = FactorVAEModel(input_dim=6, hidden_size=8, num_factors=3, num_portfolios=4)
    x_seq = torch.randn(12, 5, 6)
    returns = torch.randn(12, 1)

    outputs = model(x_seq, returns)
    prior_features = model.encode_prior_features(x_seq)

    assert outputs["reconstruction"].shape == (12, 1)
    assert outputs["stock_latent"].shape == (12, 8)
    assert outputs["posterior_mu"].shape == (3,)
    assert outputs["posterior_sigma"].shape == (3,)
    assert outputs["prior_mu"].shape == (3,)
    assert outputs["prior_sigma"].shape == (3,)
    assert outputs["beta"].shape == (12, 3)
    assert prior_features["expected_return"].shape == (12, 1)


def test_factor_vae_sequence_builder_uses_only_past_and_present_rows():
    extractor = NeuralFeatureExtractor()
    feature_cols = ["alpha_1", "alpha_2"]
    rows = []
    for ticker in ["AAA", "BBB"]:
        for i, date in enumerate(pd.date_range("2020-01-01", periods=4, freq="D")):
            rows.append({
                "Date": date,
                "ticker": ticker,
                "alpha_1": i + (10 if ticker == "BBB" else 0),
                "alpha_2": i + 100 + (10 if ticker == "BBB" else 0),
            })
    frame = pd.DataFrame(rows)

    X_seq, sequence_df = extractor._build_factor_vae_sequences(frame, feature_cols, seq_len=3)

    aaa_third = sequence_df[(sequence_df["ticker"] == "AAA") & (sequence_df["Date"] == pd.Timestamp("2020-01-03"))].index[0]
    np.testing.assert_array_equal(
        X_seq[aaa_third],
        np.array([[0, 100], [1, 101], [2, 102]], dtype=np.float32),
    )

    aaa_first = sequence_df[(sequence_df["ticker"] == "AAA") & (sequence_df["Date"] == pd.Timestamp("2020-01-01"))].index[0]
    np.testing.assert_array_equal(
        X_seq[aaa_first],
        np.array([[0, 0], [0, 0], [0, 100]], dtype=np.float32),
    )


def test_factor_vae_feature_extraction_schema_is_stable():
    rng = np.random.default_rng(123)
    feature_cols = [f"alpha_{i}" for i in range(4)]
    rows = []
    for ticker in ["AAA", "BBB"]:
        for date in pd.date_range("2020-01-01", periods=30, freq="D"):
            values = rng.normal(size=len(feature_cols))
            row = {"Date": date, "ticker": ticker, "Target": rng.normal()}
            row.update(dict(zip(feature_cols, values)))
            rows.append(row)
    df = pd.DataFrame(rows)

    extractor = NeuralFeatureExtractor()
    transformed = extractor.fit_transform_factor_vae(
        df[["Date", "ticker"] + feature_cols],
        feature_cols,
        target=df["Target"],
        seq_len=5,
        hidden_size=6,
        num_factors=2,
        num_portfolios=3,
        batch_size=16,
        max_epochs=2,
        patience=2,
        keep_original_features=False,
    )
    transformed_test = extractor.transform_factor_vae(df[["Date", "ticker"] + feature_cols])

    expected_cols = [
        "factorvae_beta_1",
        "factorvae_beta_2",
        "factorvae_alpha_mu",
        "factorvae_alpha_sigma",
        "factorvae_expected_return",
    ]
    assert all(col in transformed.columns for col in expected_cols)
    assert not any(col.startswith("factorvae_component_") for col in transformed.columns)
    assert not any(col.startswith("factorvae_prior_factor_mu_") for col in transformed.columns)
    assert "factorvae_expected_return_prior" not in transformed.columns
    assert not any(col in transformed.columns for col in feature_cols)
    assert transformed[expected_cols].shape == transformed_test[expected_cols].shape


def test_factorvae_params_are_excluded_from_tree_params():
    grid_params = {
        "learning_rate": 0.01,
        "max_depth": 3,
        "factorvae__seq_len": 20,
        "factorvae__hidden_size": 32,
    }
    tree_grid_params = {
        k: v
        for k, v in grid_params.items()
        if not k.startswith(("pca__", "ae__", "fa__", "factorvae__"))
    }
    assert tree_grid_params == {"learning_rate": 0.01, "max_depth": 3}
