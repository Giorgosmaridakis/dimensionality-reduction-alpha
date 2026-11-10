"""
Phase 6: Handling NaNs
Requirement: Ensure the extractor fills missing values with the feature's mean or zero to prevent the entire network output from becoming NaN.
"""

from __future__ import annotations
import pytest
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase6.test_config import ae_vanilla_context

def verify_nan_mitigation(ctx):
    """Internal helper to inject NaNs and verify non-NaN output."""
    extractor = ctx["extractor"]
    df_raw = ctx["df_scaled"].copy()
    cols = ctx["alpha_cols"]
    
    # Inject synthetic NaNs into real data copy
    df_raw.iloc[0, df_raw.columns.get_loc(cols[0])] = np.nan
    df_raw.iloc[2, df_raw.columns.get_loc(cols[1])] = np.nan
    
    df_out = extractor.transform_autoencoder(df_raw)
    comp_prefix = "ae_component_"
        
    out_cols = [c for c in df_out.columns if c.startswith(comp_prefix)]
    for c in out_cols:
        assert not df_out[c].isna().any(), f"NaN detected in {c} after processing {ctx['type']}"

def test_handling_nans_ae(ae_vanilla_context):
    """Verifies NaN mitigation for Vanilla Autoencoder."""
    verify_nan_mitigation(ae_vanilla_context)
