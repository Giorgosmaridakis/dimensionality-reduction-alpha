"""
Phase 4: Unit-Variance Verification
Requirement: Confirm that the standard deviation is exactly 1.0.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase4.test_config import scaled_context

def test_unit_variance(scaled_context):
    """
    Confirm that the standard deviation of each scaled feature is 1.0.
    """
    alpha_cols = scaled_context["alpha_cols"]
    df_scaled = scaled_context["df_scaled"]
    universe = scaled_context["universe"]
    period = scaled_context["period"]
    
    if df_scaled.empty:
        pytest.skip("Empty dataset.")

    for col in alpha_cols:
        std_val = df_scaled[col].std()
        
        # We only check unit variance for columns that had non-zero variance at start
        # DatasetBuilder + FeatureScaler logic handles std=0 by setting std=1 temporarily (no scaling done)
        # So it should be either close to 1.0 or close to 0.0 (if originally constant)
        if std_val > 1e-5:
            assert np.isclose(std_val, 1.0, atol=1e-8), \
                f"Feature {col} Std in {universe} {period} is not 1.0: {std_val}"

    print(f"\n[OK] Unit-Variance Verification: Features have unit variance for {universe} {period}.")
