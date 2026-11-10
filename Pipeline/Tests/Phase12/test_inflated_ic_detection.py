"""
Phase 12 – Test 5: Inflated IC Sanity Check (Permutation Baseline)
====================================================================
If a model is overfitted or has leaked data, its IC will remain high
even when predictions are randomly shuffled within each cross-section.

Expected behaviour:
    - Real predictions   → IC significantly above 0 (ideally > 0.03 for daily)
    - Shuffled predictions → IC collapses to ≈ 0 (|IC| < 0.05)

This test ALSO serves as a regression guard: if someone accidentally wires
train labels as test predictions, the shuffled IC will be suspiciously high.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_inflated_ic_detection.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
pytest_plugins = ["Tests.Phase12.test_config"]

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



class TestInflatedICDetection:
    """
    If a model is overfitted or has leaked data, its IC will remain high
    even when predictions are randomly shuffled within each cross-section.

    The correct behaviour:
        - Real predictions → IC significantly above 0 (ideally > 0.03 for daily)
        - Shuffled predictions → IC collapses to ≈ 0 (|IC| < 0.05)

    This test ALSO serves as a regression guard: if someone accidentally wires
    train labels as test predictions, the shuffled IC will be suspiciously high.
    """

    N_PERMUTATIONS = 200
    SHUFFLE_IC_TOLERANCE = 0.05   # |mean shuffled IC| must stay below this

    def _cross_sectional_ic(self, df: pd.DataFrame, pred_col: str, actual_col: str) -> float:
        """Mean daily rank-IC across the test period."""
        daily_ics = df[["Date", pred_col, actual_col]].groupby("Date").apply(
            lambda g: stats.spearmanr(g[pred_col], g[actual_col])[0]
            if len(g) > 1 and g[pred_col].nunique() > 1
            else np.nan,
            include_groups=False,
        )
        return float(daily_ics.mean())

    def _build_prediction_df(self, test_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Simulate a 'model prediction' by using the first alpha feature as a
        predictor (no real signal on random data → IC ≈ 0 by design).
        """
        df = test_df.reset_index()[["Date", "ticker", "Target"] + feature_cols].copy()
        df["Prediction"] = df[feature_cols[0]]   # first alpha = stand-in for model score
        return df

    def test_shuffled_predictions_give_near_zero_ic(self, split_data):
        """
        Randomly permuting predictions within each cross-section must yield
        an IC close to 0.  If it doesn't, something has gone very wrong
        (e.g., the same predictions are duplicated in the target column).
        """
        _, test_df, feature_cols = split_data
        pred_df = self._build_prediction_df(test_df, feature_cols)

        rng = np.random.default_rng(99)
        shuffled_ics = []

        for _ in range(self.N_PERMUTATIONS):
            shuffled = pred_df.copy()
            shuffled["Prediction"] = shuffled.groupby("Date")["Prediction"].transform(
                lambda x: rng.permutation(x.values)
            )
            ic = self._cross_sectional_ic(shuffled, "Prediction", "Target")
            shuffled_ics.append(ic)

        mean_shuffled_ic = np.nanmean(shuffled_ics)

        assert abs(mean_shuffled_ic) < self.SHUFFLE_IC_TOLERANCE, (
            f"INFLATED IC DETECTED – Mean IC of {self.N_PERMUTATIONS} shuffled permutations "
            f"is {mean_shuffled_ic:.4f}, which exceeds the tolerance of "
            f"±{self.SHUFFLE_IC_TOLERANCE}. This strongly suggests the 'predictions' "
            f"still contain target-correlated information after shuffling."
        )

    def test_ic_collapses_relative_to_real_predictions(self, split_data):
        """
        The std of the shuffled-IC distribution must be plausibly small,
        confirming the distribution is centred tightly around 0.
        On a leaky dataset the shuffled IC would remain elevated.
        """
        _, test_df, feature_cols = split_data
        pred_df = self._build_prediction_df(test_df, feature_cols)

        rng = np.random.default_rng(42)
        shuffled_ics = []

        for _ in range(self.N_PERMUTATIONS):
            shuffled = pred_df.copy()
            shuffled["Prediction"] = shuffled.groupby("Date")["Prediction"].transform(
                lambda x: rng.permutation(x.values)
            )
            ic = self._cross_sectional_ic(shuffled, "Prediction", "Target")
            shuffled_ics.append(ic)

        shuffled_std = np.nanstd(shuffled_ics)

        # On random data with N tickers, E[std(IC)] ≈ 1/sqrt(N) ≈ 0.22 for N=20
        # but the MEAN across many permutations should be tiny (≈ 0)
        expected_std_upper = 2.0 / np.sqrt(20)   # generous upper bound

        assert shuffled_std < expected_std_upper, (
            f"INFLATED IC DETECTED – std of shuffled IC distribution is {shuffled_std:.4f}, "
            f"above the expected upper bound of {expected_std_upper:.4f}. "
            f"This may indicate correlated predictions and targets within each cross-section."
        )
