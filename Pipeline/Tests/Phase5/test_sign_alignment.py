"""
Phase 5: Sign-Deterministic Alignment
Requirement: Ensure the "direction" of components remains stable for PCA and FA.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from features.decomposition import FeatureDecomposer

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase5.test_config import get_base_context, TEST_PARAMS

@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_pca_sign_alignment(universe, period):
    """
    Verify that PCA sign alignment logic works.
    Fits a reference, then fits a second PCA and check dot products.
    """
    ctx = get_base_context(universe, period)
    df = ctx["df_scaled"]
    alpha_cols = ctx["alpha_cols"]
    
    # 1. Fit reference PCA
    decom_ref = FeatureDecomposer()
    decom_ref.fit_transform_pca(df, alpha_cols, n_components=5)
    
    # 2. Fit aligned PCA using the same data (should result in same signs)
    decom_aligned = FeatureDecomposer()
    decom_aligned.fit_transform_pca(df, alpha_cols, n_components=5, align_signs_with=decom_ref)
    
    # 3. Check dot products of components
    # The dot product of aligned components with ref components should be exactly 1.0 (positive)
    for i in range(5):
        dot = np.dot(decom_aligned.pca.components_[i], decom_ref.pca.components_[i])
        assert dot > 0.99, f"PCA Sign Alignment failed at component {i+1} (dot={dot})"
    
    print(f"\n[OK] PCA Sign Alignment verified for {universe} {period}.")

@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_fa_sign_alignment(universe, period):
    """
    Verify that FA sign alignment logic works.
    """
    ctx = get_base_context(universe, period)
    df = ctx["df_scaled"]
    alpha_cols = ctx["alpha_cols"]
    
    # 1. Fit reference FA
    decom_ref = FeatureDecomposer()
    decom_ref.fit_transform_fa(df, alpha_cols, n_components=5)
    
    # 2. Fit aligned FA
    decom_aligned = FeatureDecomposer()
    decom_aligned.fit_transform_fa(df, alpha_cols, n_components=5, align_signs_with=decom_ref)
    
    # 3. Check dot products
    for i in range(5):
        dot = np.dot(decom_aligned.fa.components_[i], decom_ref.fa.components_[i])
        assert dot > 0.99, f"FA Sign Alignment failed at component {i+1} (dot={dot})"

    print(f"\n[OK] FA Sign Alignment verified for {universe} {period}.")
