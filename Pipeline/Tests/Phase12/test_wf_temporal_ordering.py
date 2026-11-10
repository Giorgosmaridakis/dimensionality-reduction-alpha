"""
Phase 12 – Test 6: Walk-Forward Temporal Ordering & Embargo Enforcement
========================================================================

Verifies two things in a single walk-forward run:
  1. Inner-CV folds never use future dates for training.
  2. The embargo gap between train end and validation start is ≥ embargo_days.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_wf_temporal_ordering.py -v
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
import modeling.tuning as tuning_package
from walking_forward import WalkingForwardForecaster
pytest_plugins = ["Tests.Phase12.test_config"]


# ---------------------------------------------------------------------------
# Mock model – records every fit() call so we can inspect fold boundaries
# ---------------------------------------------------------------------------

class _RecordingModel(Model):
    def __init__(self):
        self.model_type = "lightgbm"
        self.task_type = "regression"
        self.loss_function = "rmse"
        self.use_gpu = False
        self.params = {"n_estimators": 10}
        self.best_params_ = {"n_estimators": 10}
        self.cat_features = []
        self.fit_records: list[dict] = []

    def fit(self, X, y, qids=None, eval_set=None, **kwargs):
        self.fit_records.append({
            "train_dates": X.index.unique().sort_values().tolist(),
            "val_dates": (
                eval_set[0].index.unique().sort_values().tolist()
                if eval_set is not None else []
            ),
            "is_inner_cv": eval_set is not None,
        })

    def predict(self, X, qids=None):
        return np.zeros(len(X))


class TestWalkForwardTemporalOrdering:
    """
    Inner CV folds must be temporally ordered with an enforced embargo gap.
    """

    EMBARGO_DAYS = 1

    @staticmethod
    def _patch_recording_inner_cv_model(monkeypatch):
        fit_records = []

        class _RecordingTempModel:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, X, y, qids=None, eval_set=None, **kwargs):
                fit_records.append({
                    "train_dates": X.index.unique().sort_values().tolist(),
                    "val_dates": (
                        eval_set[0].index.unique().sort_values().tolist()
                        if eval_set is not None else []
                    ),
                    "is_inner_cv": eval_set is not None,
                })

            def predict(self, X, qids=None):
                return np.zeros(len(X))

        monkeypatch.setattr(tuning_package, "Model", _RecordingTempModel)
        return fit_records

    def test_inner_cv_no_lookahead(self, dummy_df, clean_test_dirs, monkeypatch):
        """Train dates must end strictly before validation dates begin."""
        fit_records = self._patch_recording_inner_cv_model(monkeypatch)
        model = _RecordingModel()
        forecaster = WalkingForwardForecaster(
            model_wrapper=model,
            start_date="2020-01-01",
            end_date="2020-12-31",
            max_training_window_years=1,
            testing_window="Quarter",
            embargo_days=self.EMBARGO_DAYS,
        )

        forecaster.forecast(
            full_df=dummy_df,
            candidate_features=["feature_1", "feature_2"],
            base_output_dir="Test_Results_1",
            use_standardization=True,
            param_grid={"learning_rate": [0.01, 0.05]},
            as_percentages=False,
        )

        cv_fits = [r for r in fit_records if r["is_inner_cv"]]
        assert len(cv_fits) > 0, "No inner-CV fits were recorded – check param_grid wiring."

        for i, rec in enumerate(cv_fits):
            t_max = pd.Timestamp(rec["train_dates"][-1])
            v_min = pd.Timestamp(rec["val_dates"][0])
            assert t_max < v_min, (
                f"LOOKAHEAD in CV fold {i}: "
                f"train ends {t_max.date()} but val starts {v_min.date()}"
            )

    def test_inner_cv_embargo_respected(self, dummy_df, clean_test_dirs, monkeypatch):
        """The calendar gap between last train date and first val date must be ≥ embargo_days."""
        fit_records = self._patch_recording_inner_cv_model(monkeypatch)
        model = _RecordingModel()
        forecaster = WalkingForwardForecaster(
            model_wrapper=model,
            start_date="2020-01-01",
            end_date="2020-12-31",
            max_training_window_years=1,
            testing_window="Quarter",
            embargo_days=self.EMBARGO_DAYS,
        )

        forecaster.forecast(
            full_df=dummy_df,
            candidate_features=["feature_1", "feature_2"],
            base_output_dir="Test_Results_1",
            use_standardization=True,
            param_grid={"learning_rate": [0.01, 0.05]},
            as_percentages=False,
        )

        cv_fits = [r for r in fit_records if r["is_inner_cv"]]
        assert len(cv_fits) > 0, "No inner-CV fits were recorded – check param_grid wiring."
        for i, rec in enumerate(cv_fits):
            t_max = pd.Timestamp(rec["train_dates"][-1])
            v_min = pd.Timestamp(rec["val_dates"][0])
            gap_days = (v_min - t_max).days
            assert gap_days >= self.EMBARGO_DAYS, (
                f"EMBARGO VIOLATED in CV fold {i}: "
                f"gap={gap_days} day(s), expected ≥ {self.EMBARGO_DAYS}"
            )
