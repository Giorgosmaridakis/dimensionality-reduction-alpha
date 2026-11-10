"""
Phase 6: Standardization Assumption
Requirement: Confirm that the input data was normalized before entering the network, as Neural Networks are sensitive to scale.
"""

from __future__ import annotations
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase6.test_config import ae_vanilla_context

def verify_standardization(ctx):
    """
    Internal helper to check mean/std of scaled features in context.

    Zero-variance columns (all values identical in the sample) are intentionally
    skipped. The FeatureScaler correctly maps them to 0.0 rather than NaN, and
    this edge case is already covered by the Phase 4 zero-std test.
    The Phase 6 config uses only 200 rows, which makes degenerate columns more
    likely in a small numeric sample.
    """
    df = ctx["df_scaled"]
    cols = ctx["alpha_cols"]
    skipped = []

    for col in cols:
        raw_std = df[col].std()

        # Skip columns that were constant before scaling — FeatureScaler
        # returns 0.0 for them, which is correct but cannot satisfy 0.9 < std < 1.1.
        if np.isclose(raw_std, 0.0):
            skipped.append(col)
            continue

        mean = df[col].mean()
        assert np.abs(mean) < 0.1, f"Mean too high for {col}: {mean:.4f}"
        assert 0.9 < raw_std < 1.1, f"Std out of range for {col}: {raw_std:.4f}"

    if skipped:
        print(
            f"\n[WARN] Skipped zero-variance columns (constant in 200-row sample): "
            f"{skipped}. This is expected for degenerate alphas."
        )


def test_standardization_assumption_ae(ae_vanilla_context):
    """Verifies standardization for the Vanilla Autoencoder input pipe."""
    verify_standardization(ae_vanilla_context)
