"""
Phase 12 – Test 4: Target-Peek Leakage
========================================
Features must be causally prior to the target.
A strong contemporaneous or forward correlation between a raw alpha feature
and the *next-period* target signals that the feature is somehow constructed
from future prices.

We use a conservative |rank-IC| > 0.5 threshold — real alpha ICs are
typically < 0.05, so anything above 0.5 is an almost certain sign of leakage.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_target_peek_leakage.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import scipy.stats as stats

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest_plugins = ["Tests.Phase12.test_config"]



class TestTargetPeekLeakage:
    """
    Features must be causally prior to the target.
    A strong contemporaneous or forward correlation between a raw alpha feature
    and the *next-period* target signals that the feature is somehow constructed
    from future prices.

    We use a conservative |rank-IC| > 0.5 threshold — real alpha ICs are
    typically < 0.05, so anything above 0.5 is an almost certain sign of leakage.
    """

    LEAKAGE_IC_THRESHOLD = 0.5

    def test_feature_does_not_encode_future_target(self, panel_df):
        """
        Shift the Target by –1 (one step forward) and check that no raw feature
        has an absurdly high rank-IC with that future target.
        """
        df = panel_df.copy()
        feature_cols = [c for c in df.columns if c.startswith("alpha")]

        # future_target[t] = Target[t+1] – leaked if a feature at t encodes it
        df["future_target"] = df.groupby("ticker")["Target"].shift(-1)
        df = df.dropna(subset=["future_target"])

        leaky_features = []
        for col in feature_cols:
            ic, _ = stats.spearmanr(df[col], df["future_target"])
            if abs(ic) > self.LEAKAGE_IC_THRESHOLD:
                leaky_features.append((col, ic))

        assert len(leaky_features) == 0, (
            f"TARGET-PEEK LEAKAGE DETECTED – {len(leaky_features)} feature(s) have "
            f"|IC| > {self.LEAKAGE_IC_THRESHOLD} with the next-period target:\n"
            + "\n".join(f"  {col}: IC={ic:.4f}" for col, ic in leaky_features)
        )

    def test_target_not_in_feature_set(self, split_data):
        """
        The literal target column (or any derivative of it) must not appear
        among the features passed to the model.
        """
        _, _, feature_cols = split_data
        forbidden = {"Target", "target", "Actual_Return", "actual_return",
                     "forward_return", "next_return"}
        leaked = set(feature_cols) & forbidden
        assert len(leaked) == 0, (
            f"TARGET-PEEK LEAKAGE – The following target-derived columns are in the "
            f"feature list: {leaked}"
        )
