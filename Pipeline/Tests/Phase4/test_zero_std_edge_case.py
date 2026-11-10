"""
Phase 4: Zero-Std Edge Case
Requirement: If a feature has zero variance (all values identical), ensure the scaler returns 0.0 rather than NaN or infinity.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from features.scaling import FeatureScaler

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_zero_std_edge_case():
    """
    If a feature has zero variance (all values identical), 
    ensure the scaler returns 0.0 rather than NaN or infinity.
    """
    scaler = FeatureScaler()
    
    # Create DF with a constant column (must start with 'alpha' to be auto-detected)
    df_constant = pd.DataFrame({
        'alpha_const': [100.0, 100.0, 100.0, 100.0, 100.0]
    })
    
    # fit_transform (parameters auto-detected)
    df_out = scaler.fit_transform_standardize(df_constant)
    
    # Check for NaN/Inf
    assert not df_out['alpha_const'].isna().any(), "Scaler produced NaNs for constant column!"
    assert np.isfinite(df_out['alpha_const']).all(), "Scaler produced infinite values for constant column!"
    
    # Verify values are 0.0
    # math: (100 - mean(100)) / 1.0 = 0.0 / 1.0 = 0.0
    assert (df_out['alpha_const'] == 0.0).all(), f"Constant column did not result in zeros: {df_out['alpha_const'].values}"

    print(f"\n[OK] Zero-Std Edge Case: Constant features correctly handled and returned as 0.0.")
