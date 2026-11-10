"""
Phase 5: Orthogonality Verification
Requirement: Prove that the resulting components have zero correlation with each other (specifically for PCA).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase5.test_config import decomposed_context_pca

def test_pca_orthogonality(decomposed_context_pca):
    """
    Prove that the resulting PCA components have zero correlation with each other.
    """
    df = decomposed_context_pca["df_transformed"]
    pca_cols = [c for c in df.columns if c.startswith('pca_component_')]
    
    if len(pca_cols) < 2:
        pytest.skip("Not enough components to check orthogonality.")
        
    # Calculate correlation matrix
    corr_matrix = df[pca_cols].corr().values
    
    # Check off-diagonal elements
    for i in range(len(pca_cols)):
        for j in range(i + 1, len(pca_cols)):
            # PCA components should be orthogonal (and thus uncorrelated if centered)
            # sklearn's PCA handles centering.
            assert np.isclose(corr_matrix[i, j], 0.0, atol=1e-8), \
                f"PCA components {pca_cols[i]} and {pca_cols[j]} are not orthogonal (corr={corr_matrix[i, j]})"

    print(f"\n[OK] PCA Orthogonality: All {len(pca_cols)} components verified as uncorrelated.")
