"""
Phase 5: Reconstruction Path
Requirement: Verify that the decomposer can transform new test data correctly and check reconstruction_mse for all three methods.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase5.test_config import decomposed_context_pca, decomposed_context_fa

def test_pca_reconstruction(decomposed_context_pca):
    """Verifies PCA reconstruction MSE and transform logic."""
    decom = decomposed_context_pca["decom"]
    df_raw = decomposed_context_pca["df_scaled"]
    alpha_cols = decomposed_context_pca["alpha_cols"]
    
    # 1. Check MSE exists
    assert hasattr(decom, 'pca_reconstruction_mse'), "PCA reconstruction MSE not found."
    assert decom.pca_reconstruction_mse >= 0
    
    # 2. Verify transform
    df_transformed = decom.transform_pca(df_raw, keep_original_features=True)
    pca_cols = [c for c in df_transformed.columns if c.startswith('pca_component_')]
    assert len(pca_cols) == decomposed_context_pca["n_components"]
    
    print(f"\n[OK] PCA Reconstruction verified (MSE: {decom.pca_reconstruction_mse:.6f})")

def test_fa_reconstruction(decomposed_context_fa):
    """Verifies FA reconstruction MSE and transform logic."""
    decom = decomposed_context_fa["decom"]
    df_raw = decomposed_context_fa["df_scaled"]
    
    assert hasattr(decom, 'fa_reconstruction_mse'), "FA reconstruction MSE not found."
    assert decom.fa_reconstruction_mse >= 0
    
    df_transformed = decom.transform_fa(df_raw, keep_original_features=True)
    fa_cols = [c for c in df_transformed.columns if c.startswith('fa_component_')]
    assert len(fa_cols) == decomposed_context_fa["n_components"]
    
    print(f"\n[OK] FA Reconstruction verified (MSE: {decom.fa_reconstruction_mse:.6f})")
