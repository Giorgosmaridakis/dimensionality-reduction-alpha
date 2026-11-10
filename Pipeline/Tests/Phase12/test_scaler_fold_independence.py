"""
Phase 12 – Test 8: Scaler Independence Across Walk-Forward Folds
=================================================================

Each walk-forward fold re-fits the FeatureScaler exclusively on its own
training window. This test verifies that the scaler state (mean / std)
differs between consecutive folds, proving the scaler is being re-fitted
and not reused from a previous fold.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_scaler_fold_independence.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.trees import Model
from features.scaling import FeatureScaler
from walking_forward import WalkingForwardForecaster
pytest_plugins = ["Tests.Phase12.test_config"]


# ---------------------------------------------------------------------------
# Instrumented scaler that records its fitted mean after each fit_transform
# ---------------------------------------------------------------------------

class _InstrumentedScaler(FeatureScaler):
    """Wraps FeatureScaler and records the fitted mean of feature_1 each fold."""

    def __init__(self):
        super().__init__()
        self.recorded_means: list[float] = []

    def fit_transform_standardize(self, df, feature_cols=None):
        result = super().fit_transform_standardize(df, feature_cols=feature_cols)
        if "feature_1" in self.scalers:
            self.recorded_means.append(self.scalers["feature_1"]["mean"])
        return result


class TestScalerFoldIndependence:
    """
    The scaler must be re-fitted independently on each fold's training data.
    Identical scaler states across folds indicate the scaler is being reused.
    """

    def test_scaler_state_differs_across_folds(self, dummy_df, clean_test_dirs):
        """
        Run the forecaster with an instrumented scaler and confirm the fitted
        mean of feature_1 changes between consecutive folds.
        """
        instrumented = _InstrumentedScaler()

        model = Model(
            model_type="lightgbm",
            task_type="regression",
            loss_function="rmse",
            random_seed=42,
            use_gpu=False,
            n_estimators=10
        )

        forecaster = WalkingForwardForecaster(
            model_wrapper=model,
            start_date="2020-01-01",
            end_date="2020-12-31",
            max_training_window_years=1,
            testing_window="Quarter",
            embargo_days=1,
        )

        # Inject our instrumented scaler
        forecaster.scaler = instrumented

        forecaster.forecast(
            full_df=dummy_df,
            candidate_features=["feature_1", "feature_2"],
            base_output_dir="Test_Results_1",
            use_standardization=True,
            as_percentages=False,
        )

        means = instrumented.recorded_means
        assert len(means) >= 2, (
            f"Expected at least 2 folds, got {len(means)}. "
            "Check the date range / testing_window config."
        )

        for i in range(1, len(means)):
            diff = abs(means[i] - means[i - 1])
            assert diff > 0, (
                f"SCALER REUSED – scaler mean for feature_1 is identical "
                f"between fold {i-1} ({means[i-1]:.6f}) and fold {i} ({means[i]:.6f}). "
                "The scaler may not be re-fitted on each fold's training window."
            )
