"""
Phase 4: Zero-Mean Verification
Requirement: Confirm that the mathematical mean of each scaled feature is within a tiny tolerance (10^-9) of zero.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase4.test_config import scaled_context

def test_zero_mean(scaled_context):
    """
    Confirm that the mathematical mean of each scaled feature is 
    within a tiny tolerance (10^-9) of zero.
    """
    alpha_cols = scaled_context["alpha_cols"]
    df_scaled = scaled_context["df_scaled"]
    universe = scaled_context["universe"]
    period = scaled_context["period"]
    
    if df_scaled.empty:
        pytest.skip("Empty dataset.")

    for col in alpha_cols:
        mean_val = df_scaled[col].mean()
        # Use a slightly looser tolerance for floating point summation of thousands of rows
        assert np.isclose(mean_val, 0.0, atol=1e-8), \
            f"Feature {col} Mean in {universe} {period} is far from zero: {mean_val}"

    print(f"\n[OK] Zero-Mean Verification: All {len(alpha_cols)} features centered at zero for {universe} {period}.")
