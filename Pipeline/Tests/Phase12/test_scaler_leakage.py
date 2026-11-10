"""
Phase 12 – Test 2: Scaler Leakage
==================================
FeatureScaler must be fitted exclusively on training data.
If the scaler is fit on train+test combined, the test-set mean and std bleed
into the training statistics, inflating in-sample performance.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_scaler_leakage.py -v
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

from features.scaling import FeatureScaler
pytest_plugins = ["Tests.Phase12.test_config"]


class TestScalerLeakage:
    """
    StandardScaler (and our FeatureScaler) must be fit exclusively on the
    training set.  If the scaler is fit on train+test combined, the test-set
    mean and std bleed into the training statistics, inflating IS performance.
    """

    def test_scaler_stats_match_train_only(self, split_data):
        """
        Fit FeatureScaler on train, then verify its stored mean/std match the
        train statistics – NOT the global (train+test) statistics.
        """
        train_df, test_df, feature_cols = split_data

        scaler = FeatureScaler()
        scaler.fit_transform_standardize(train_df, feature_cols=feature_cols)

        combined = pd.concat([train_df, test_df])

        for col in feature_cols:
            train_mean = train_df[col].mean()
            train_std = train_df[col].std()
            global_mean = combined[col].mean()

            stored_mean = scaler.scalers[col]["mean"]
            stored_std = scaler.scalers[col]["std"]

            assert abs(stored_mean - train_mean) < 1e-9, (
                f"LEAKAGE – Feature '{col}': scaler mean ({stored_mean:.4f}) "
                f"does not match train mean ({train_mean:.4f}). "
                f"Global mean is {global_mean:.4f}."
            )
            assert abs(stored_std - train_std) < 1e-9, (
                f"LEAKAGE – Feature '{col}': scaler std ({stored_std:.4f}) "
                f"does not match train std ({train_std:.4f})."
            )

    def test_transform_uses_train_params_not_test(self, split_data):
        """
        Transforming the test set with a train-fitted scaler must produce a
        different result than fitting a new scaler directly on the test set.
        If they are identical, the scaler was (incorrectly) fit on the test set.
        """
        train_df, test_df, feature_cols = split_data

        # Correct: fit on train, apply to test
        scaler_train = FeatureScaler()
        scaler_train.fit_transform_standardize(train_df, feature_cols=feature_cols)
        test_correct = scaler_train.transform_standardize(test_df.copy(), feature_cols=feature_cols)

        # Wrong: fit on test (simulates leakage)
        scaler_leak = FeatureScaler()
        scaler_leak.fit_transform_standardize(test_df, feature_cols=feature_cols)
        test_leaked = scaler_leak.transform_standardize(test_df.copy(), feature_cols=feature_cols)

        for col in feature_cols:
            assert not np.allclose(test_correct[col].values, test_leaked[col].values), (
                f"WARNING – Feature '{col}': scaler trained on train and scaler trained "
                f"on test give identical results. This may indicate test-set leakage into "
                f"the scaler."
            )
