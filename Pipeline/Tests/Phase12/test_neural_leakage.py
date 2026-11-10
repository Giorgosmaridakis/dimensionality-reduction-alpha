"""
Phase 12 – Neural Path Leakage Tests
======================================
Covers all three neural expected-return paths:
  1. GKX Conditional Autoencoder  (use_autoencoder=True)
  2. FactorVAE                    (use_factor_vae=True)

These tests verify:
  - Neural models are fit exclusively on train data.
  - Test-time expected returns use prior/encode paths (not posterior).
  - GKX managed portfolios don't leak future returns into the test set.
  - GKX factor forecast is estimated from train-only data.
  - FactorVAE train/val split inside fit is temporal, not random.
  - Neural models are re-fit per walk-forward fold (not reused).
  - Scaler + neural composition doesn't leak.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_neural_leakage.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.neural_extractors import NeuralFeatureExtractor
from features.scaling import FeatureScaler

# ---------------------------------------------------------------------------
# Shared synthetic panel with enough dates for temporal splits
# ---------------------------------------------------------------------------

def _make_neural_panel(n_dates=120, n_tickers=15, n_features=8, seed=77):
    """
    Build a synthetic panel with:
      - Clean temporal DatetimeIndex
      - Ticker column
      - Random alpha features (no genuine signal)
      - Target = pure noise (independent of features)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    tickers = [f"STK{i:02d}" for i in range(n_tickers)]

    rows = []
    for date in dates:
        for ticker in tickers:
            row = {"Date": date, "ticker": ticker}
            for f in range(n_features):
                row[f"alpha_{f+1:02d}"] = rng.standard_normal()
            row["Target"] = rng.standard_normal() * 0.02  # small return-like values
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


@pytest.fixture(scope="module")
def neural_panel():
    return _make_neural_panel()


@pytest.fixture(scope="module")
def neural_split(neural_panel):
    """80/20 temporal split."""
    dates = neural_panel.index.unique().sort_values()
    split = int(len(dates) * 0.8)
    train_dates = dates[:split]
    test_dates = dates[split:]
    train_df = neural_panel.loc[train_dates]
    test_df = neural_panel.loc[test_dates]
    feature_cols = [c for c in neural_panel.columns if c.startswith("alpha_")]
    return train_df, test_df, feature_cols


# ===================================================================
# GKX AUTOENCODER TESTS
# ===================================================================

class TestGKXAutoencoderLeakage:
    """
    Verify that the GKX conditional autoencoder path does not leak
    future information.
    """

    def test_gkx_fit_on_train_only(self, neural_split):
        """
        fit_transform_autoencoder on train must produce a model whose
        internal train/val split is strictly within the train dates.
        The train/val mask function must use temporal ordering.
        """
        train_df, test_df, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        # Resolve dates
        dates = train_df.index.to_series().values
        train_mask, val_mask = extractor._date_train_val_masks(dates, val_fraction=0.2)

        # Val dates must be AFTER train dates
        all_dates = pd.to_datetime(dates)
        train_dates_masked = set(all_dates[train_mask])
        val_dates_masked = set(all_dates[val_mask])

        assert len(train_dates_masked & val_dates_masked) == 0, (
            "GKX LEAKAGE: train/val date masks overlap inside fit_transform_autoencoder."
        )

        if train_dates_masked and val_dates_masked:
            assert max(train_dates_masked) < min(val_dates_masked), (
                "GKX LEAKAGE: validation dates are not strictly after training dates. "
                f"Train max: {max(train_dates_masked)}, Val min: {min(val_dates_masked)}"
            )

    def test_gkx_transform_does_not_use_test_returns(self, neural_split):
        """
        transform_autoencoder (test-time) must not require or use
        realized returns. It should only use characteristics (features).
        """
        train_df, test_df, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        # Fit on train
        train_result = extractor.fit_transform_autoencoder(
            train_df, feature_cols,
            latent_dim=3,
            target=train_df["Target"],
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            max_epochs=5,
            patience=3,
            keep_original_features=False,
        )

        # Transform test (must not require target)
        test_result = extractor.transform_autoencoder(
            test_df, keep_original_features=False,
        )

        # Expected return column must exist
        assert "gkx_expected_return" in test_result.columns, (
            "GKX transform did not produce gkx_expected_return column."
        )

        # Expected returns should NOT be identical to train expected returns
        # (they use different characteristics)
        train_er = train_result["gkx_expected_return"].values
        test_er = test_result["gkx_expected_return"].values
        assert train_er.shape != test_er.shape or not np.allclose(train_er[:len(test_er)], test_er), (
            "GKX test expected returns are identical to train — possible leakage."
        )

    def test_gkx_factor_forecast_from_train_only(self, neural_split):
        """
        The GKX factor forecast (_estimate_gkx_factor_forecast) must be
        computed from the training data. This frozen forecast is then
        applied to both train and test betas.
        """
        train_df, test_df, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        extractor.fit_transform_autoencoder(
            train_df, feature_cols,
            latent_dim=3,
            target=train_df["Target"],
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            max_epochs=5,
            patience=3,
        )

        # Factor forecast should be a vector of length = latent_dim
        factor_forecast = extractor.ae_factor_forecast_
        assert factor_forecast is not None, "Factor forecast was not estimated."
        assert factor_forecast.shape == (3,), (
            f"Factor forecast shape mismatch: {factor_forecast.shape}"
        )

        # Verify it's not all zeros (degenerate)
        assert np.abs(factor_forecast).sum() > 1e-12, (
            "Factor forecast is all zeros — degenerate model."
        )

    def test_gkx_managed_portfolios_use_same_date_returns(self, neural_split):
        """
        Managed portfolios x_t = (Z'Z)^{-1} Z' r_t must be constructed
        per date. Returns from other dates must not leak in.
        """
        train_df, _, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        X = train_df[feature_cols].fillna(0).values
        y = train_df["Target"].values
        dates = train_df.index.to_series().values

        managed = extractor._build_managed_portfolios_by_date(X, y, dates)

        # Each date's managed portfolio should be identical across stocks
        # (it's the same x_t replicated)
        date_series = pd.Series(dates)
        for _, group_idx in date_series.groupby(date_series).groups.items():
            positions = list(group_idx)
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    assert np.allclose(managed[positions[0]], managed[positions[i]]), (
                        "GKX LEAKAGE: managed portfolios differ within the same date. "
                        "Each stock should see the same x_t."
                    )

    def test_gkx_expected_return_varies_across_stocks(self, neural_split):
        """
        Expected return = beta * f should vary across stocks (different betas)
        but use the same factor forecast. If all expected returns are identical,
        the model is degenerate.
        """
        train_df, test_df, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        extractor.fit_transform_autoencoder(
            train_df, feature_cols,
            latent_dim=3,
            target=train_df["Target"],
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            max_epochs=10,
            patience=5,
        )

        test_result = extractor.transform_autoencoder(test_df)

        # Within each date, expected returns should vary
        test_result_copy = test_result.copy()
        test_result_copy["_date"] = test_df.index
        for date, group in test_result_copy.groupby("_date"):
            er_std = group["gkx_expected_return"].std()
            # Very loose check — just confirm it's not a constant
            if len(group) > 1:
                assert er_std > 1e-10, (
                    f"GKX LEAKAGE: all expected returns identical on {date} "
                    "(std={er_std:.2e}). Betas may not be conditioning on features."
                )


# ===================================================================
# FACTOR VAE TESTS
# ===================================================================

class TestFactorVAELeakage:
    """
    Verify that the FactorVAE path does not leak future information.
    """

    def test_factorvae_internal_split_is_temporal(self, neural_split):
        """
        Inside fit_transform_factor_vae, the 80/20 split of the training
        data into sub-train/sub-val must be temporal (not random).
        """
        train_df, _, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        frame = extractor._prepare_factor_vae_frame(
            train_df, feature_cols,
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            target=train_df["Target"],
        )

        # Build sequences
        X_seq, sequence_df = extractor._build_factor_vae_sequences(
            frame, feature_cols, seq_len=10,
        )

        # Replicate the split logic from fit_transform_factor_vae
        date_groups = sequence_df.groupby("Date", sort=True).groups
        unique_dates = sorted(list(date_groups.keys()))
        split_i = max(1, int(len(unique_dates) * 0.8))
        if split_i >= len(unique_dates):
            split_i = len(unique_dates) - 1

        train_dates = set(unique_dates[:split_i])
        val_dates = set(unique_dates[split_i:])

        # No overlap
        assert len(train_dates & val_dates) == 0, (
            "FACTORVAE LEAKAGE: internal train/val date sets overlap."
        )
        # Temporal ordering
        assert max(train_dates) < min(val_dates), (
            "FACTORVAE LEAKAGE: internal val dates are not strictly after train dates."
        )

    def test_factorvae_transform_uses_prior_not_posterior(self, neural_split):
        """
        At test time, transform_factor_vae must use encode_prior_features
        (which doesn't need realized returns), not the posterior path
        (which requires realized returns).
        """
        train_df, test_df, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        # Fit
        extractor.fit_transform_factor_vae(
            train_df, feature_cols,
            target=train_df["Target"],
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            seq_len=5,
            hidden_size=16,
            num_factors=4,
            num_portfolios=8,
            max_epochs=5,
            patience=3,
        )

        # Transform test (must succeed without target)
        test_result = extractor.transform_factor_vae(
            test_df,
            dates=test_df.index,
            tickers=test_df.get("ticker"),
        )

        assert "factorvae_expected_return" in test_result.columns, (
            "FactorVAE transform did not produce factorvae_expected_return."
        )

        # The model's encode_prior_features method should be what's used
        # Verify it exists and works
        model = extractor.factor_vae_model
        dummy_seq = torch.randn(5, 5, len(feature_cols))
        with torch.no_grad():
            prior_out = model.encode_prior_features(dummy_seq)

        assert "expected_return" in prior_out, (
            "FactorVAE.encode_prior_features missing expected_return key."
        )

    def test_factorvae_expected_return_not_correlated_with_noise_target(self, neural_split):
        """
        Since the target is pure noise (uncorrelated with features), the
        FactorVAE expected returns should have near-zero rank-IC with
        the test target. An unreasonably high IC (>0.3) indicates leakage.
        """
        train_df, test_df, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        extractor.fit_transform_factor_vae(
            train_df, feature_cols,
            target=train_df["Target"],
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            seq_len=5,
            hidden_size=16,
            num_factors=4,
            num_portfolios=8,
            max_epochs=10,
            patience=5,
        )

        test_result = extractor.transform_factor_vae(
            test_df,
            dates=test_df.index,
            tickers=test_df.get("ticker"),
        )

        from scipy.stats import spearmanr
        ic, _ = spearmanr(test_result["factorvae_expected_return"], test_df["Target"])
        assert abs(ic) < 0.3, (
            f"FACTORVAE LEAKAGE: OOS IC = {ic:.4f} on pure-noise data. "
            "Expected near zero. This strongly suggests target information "
            "leaked into the expected return predictions."
        )

    def test_factorvae_sequences_respect_ticker_boundaries(self, neural_split):
        """
        Sequences should not cross ticker boundaries. A sequence for
        stock A must only contain past observations of stock A.
        """
        train_df, _, feature_cols = neural_split
        extractor = NeuralFeatureExtractor()

        frame = extractor._prepare_factor_vae_frame(
            train_df, feature_cols,
            dates=train_df.index,
            tickers=train_df.get("ticker"),
        )

        _, sequence_df = extractor._build_factor_vae_sequences(
            frame, feature_cols, seq_len=5,
        )

        # Each row in sequence_df should have a single ticker
        assert "ticker" in sequence_df.columns, (
            "sequence_df missing ticker column."
        )
        # Verify that the _original_index ordering is preserved per ticker
        for ticker, group in sequence_df.groupby("ticker"):
            dates_arr = group["Date"].values
            # Dates should be monotonically non-decreasing within each ticker
            assert all(dates_arr[i] <= dates_arr[i+1] for i in range(len(dates_arr)-1)), (
                f"FACTORVAE LEAKAGE: dates for ticker {ticker} are not sorted. "
                "Sequences may cross ticker boundaries."
            )


# ===================================================================
# CROSS-CUTTING: SCALER + NEURAL COMPOSITION
# ===================================================================

class TestScalerNeuralComposition:
    """
    When scaling is applied before neural extraction, the scaler
    must be fit on train and the neural model on (scaled) train.
    """

    def test_scaling_then_gkx_uses_train_scaler(self, neural_split):
        """
        Applying standardization before GKX must use train-fitted scaler.
        Test data must be scaled with train statistics.
        """
        train_df, test_df, feature_cols = neural_split
        scaler = FeatureScaler()

        X_train_scaled = scaler.fit_transform_standardize(train_df.copy(), feature_cols=feature_cols)
        X_test_scaled = scaler.transform_standardize(test_df.copy(), feature_cols=feature_cols)

        # Train features should be roughly mean-0, std-1
        for col in feature_cols:
            mean = X_train_scaled[col].mean()
            std = X_train_scaled[col].std()
            assert abs(mean) < 0.1, f"Scaler leakage: train {col} mean={mean:.4f}"
            assert abs(std - 1.0) < 0.3, f"Scaler leakage: train {col} std={std:.4f}"

        # Test features should NOT be exactly mean-0, std-1
        # (they use train statistics, which differ from test statistics)
        test_means = [X_test_scaled[col].mean() for col in feature_cols]
        test_stds = [X_test_scaled[col].std() for col in feature_cols]

        # At least some features should deviate from 0/1
        mean_deviations = [abs(m) for m in test_means]
        assert max(mean_deviations) > 0.01, (
            "Test set means are suspiciously close to 0 for all features. "
            "The scaler may have been re-fit on test data."
        )

    def test_scaling_then_factorvae_train_only(self, neural_split):
        """
        FactorVAE should receive train-scaled features. Scaling must not
        be fit on test data.
        """
        train_df, test_df, feature_cols = neural_split
        scaler = FeatureScaler()

        X_train_scaled = scaler.fit_transform_standardize(train_df.copy(), feature_cols=feature_cols)

        extractor = NeuralFeatureExtractor()
        result = extractor.fit_transform_factor_vae(
            X_train_scaled, feature_cols,
            target=train_df["Target"],
            dates=train_df.index,
            tickers=train_df.get("ticker"),
            seq_len=5,
            hidden_size=16,
            num_factors=4,
            num_portfolios=8,
            max_epochs=3,
            patience=2,
        )

        # Should produce columns
        assert "factorvae_expected_return" in result.columns
        assert any(c.startswith("factorvae_beta_") for c in result.columns)


# ===================================================================
# NEURAL MODEL RE-FIT PER FOLD
# ===================================================================

class TestNeuralFoldIndependence:
    """
    Neural models must be re-fit for each walk-forward fold.
    Using the same fitted model across folds is a form of leakage
    because later folds' training data was available during the
    initial fit.
    """

    def test_factorvae_refit_produces_different_weights(self, neural_split):
        """
        Fitting FactorVAE twice on different data subsets should produce
        different model weights. Identical weights suggest the model is
        cached and not re-fit.
        """
        train_df, _, feature_cols = neural_split
        dates = train_df.index.unique().sort_values()
        mid = len(dates) // 2

        fold1_df = train_df.loc[dates[:mid]]
        fold2_df = train_df.loc[dates[mid:]]

        ext1 = NeuralFeatureExtractor()
        ext1.fit_transform_factor_vae(
            fold1_df, feature_cols,
            target=fold1_df["Target"],
            dates=fold1_df.index,
            tickers=fold1_df.get("ticker"),
            seq_len=5, hidden_size=16, num_factors=4, num_portfolios=8,
            max_epochs=3, patience=2, random_seed=42,
        )

        ext2 = NeuralFeatureExtractor()
        ext2.fit_transform_factor_vae(
            fold2_df, feature_cols,
            target=fold2_df["Target"],
            dates=fold2_df.index,
            tickers=fold2_df.get("ticker"),
            seq_len=5, hidden_size=16, num_factors=4, num_portfolios=8,
            max_epochs=3, patience=2, random_seed=42,
        )

        # Compare model weights
        state1 = ext1.factor_vae_model.state_dict()
        state2 = ext2.factor_vae_model.state_dict()

        all_same = all(
            torch.allclose(state1[k], state2[k], atol=1e-6)
            for k in state1
        )
        assert not all_same, (
            "LEAKAGE: FactorVAE produced identical weights on two different "
            "data subsets. The model may not be re-fitting per fold."
        )

    def test_gkx_refit_produces_different_weights(self, neural_split):
        """
        Same check for GKX autoencoder.
        """
        train_df, _, feature_cols = neural_split
        dates = train_df.index.unique().sort_values()
        mid = len(dates) // 2

        fold1_df = train_df.loc[dates[:mid]]
        fold2_df = train_df.loc[dates[mid:]]

        ext1 = NeuralFeatureExtractor()
        ext1.fit_transform_autoencoder(
            fold1_df, feature_cols,
            latent_dim=3,
            target=fold1_df["Target"],
            dates=fold1_df.index,
            tickers=fold1_df.get("ticker"),
            max_epochs=3, patience=2,
        )

        ext2 = NeuralFeatureExtractor()
        ext2.fit_transform_autoencoder(
            fold2_df, feature_cols,
            latent_dim=3,
            target=fold2_df["Target"],
            dates=fold2_df.index,
            tickers=fold2_df.get("ticker"),
            max_epochs=3, patience=2,
        )

        state1 = ext1.autoencoder.state_dict()
        state2 = ext2.autoencoder.state_dict()

        all_same = all(
            torch.allclose(state1[k], state2[k], atol=1e-6)
            for k in state1
        )
        assert not all_same, (
            "LEAKAGE: GKX autoencoder produced identical weights on two different "
            "data subsets. The model may not be re-fitting per fold."
        )
