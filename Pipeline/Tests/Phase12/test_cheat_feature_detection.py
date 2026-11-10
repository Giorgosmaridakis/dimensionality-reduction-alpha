"""
Phase 12 – Test 7: Cheating Feature Exposes Leakage via MSE Collapse
======================================================================
Migrated from test_pipeline.py (run_test_2).

An honest model is trained on {feature_1, feature_2}.
A cheating model is given {feature_1, feature_2, feature_3_cheat} where
feature_3_cheat == Target (a perfect future-peek feature).

The cheating model's MSE must be dramatically lower (< 10 % of honest MSE).
If it isn't, either the cheating feature is not being used or the honest
model already has access to the target somehow.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_cheat_feature_detection.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.trees import Model
from walking_forward import WalkingForwardForecaster

pytest_plugins = ["Tests.Phase12.test_config"]

# Prediction CSV written by the first quarter fold
_FOLD_SUBDIR = "20200102_20200331"
_PRED_FILE = f"predictions_{_FOLD_SUBDIR}.csv"


def _run_forecaster(df, features, output_dir, model_seed=42):
    model = Model(
        model_type="lightgbm",
        task_type="regression",
        loss_function="rmse",
        random_seed=model_seed,
        use_gpu=False,
        n_estimators=10
    )
    forecaster = WalkingForwardForecaster(
        model_wrapper=model,
        start_date="2020-01-01",
        end_date="2020-03-31",
        max_training_window_years=1,
        testing_window="Quarter",
        embargo_days=1,
    )
    forecaster.forecast(
        full_df=df,
        candidate_features=features,
        base_output_dir=output_dir,
        as_percentages=False,
    )


def _load_predictions(output_dir):
    """Find the first predictions CSV under output_dir (any subfolder)."""
    pred_files = sorted(Path(output_dir).rglob("predictions_*.csv"))
    assert pred_files, f"No prediction files found under {output_dir}"
    return pd.read_csv(pred_files[0])


class TestCheatFeatureDetection:
    """
    A feature that is identical to the target (future-peek) must produce
    near-zero MSE. Confirms the pipeline is sensitive enough to expose leakage.
    """

    def test_cheat_model_mse_far_below_honest(self, dummy_df, clean_test_dirs):
        """Cheating model MSE must be < 10 % of honest model MSE."""
        _run_forecaster(dummy_df, ["feature_1", "feature_2"], "Test_Results_Honest")
        _run_forecaster(
            dummy_df,
            ["feature_1", "feature_2", "feature_3_cheat"],
            "Test_Results_Cheat",
        )

        honest_preds = _load_predictions("Test_Results_Honest")
        cheat_preds = _load_predictions("Test_Results_Cheat")

        mse_honest = mean_squared_error(honest_preds["Actual"], honest_preds["Prediction"])
        mse_cheat = mean_squared_error(cheat_preds["Actual"], cheat_preds["Prediction"])

        assert mse_cheat < mse_honest * 0.20, (
            f"LEAKAGE DETECTION FAILED – cheating model MSE ({mse_cheat:.6f}) "
            f"is not < 20 % of honest model MSE ({mse_honest:.6f}). "
            "The pipeline may not be correctly exposing target-peek features."
        )
