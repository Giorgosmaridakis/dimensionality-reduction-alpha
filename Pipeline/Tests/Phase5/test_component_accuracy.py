"""
Phase 5: Component Accuracy
Requirement: Ensure the resulting DataFrame contains exactly K components for PCA, FA.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase5.test_config import decomposed_context_pca, decomposed_context_fa

def test_pca_component_accuracy(decomposed_context_pca):
    """Verifies K components for PCA."""
    df = decomposed_context_pca["df_transformed"]
    k = decomposed_context_pca["n_components"]
    cols = [c for c in df.columns if c.startswith('pca_component_')]
    assert len(cols) == k, f"PCA expected {k} components, got {len(cols)}"

def test_fa_component_accuracy(decomposed_context_fa):
    """Verifies K components for Factor Analysis."""
    df = decomposed_context_fa["df_transformed"]
    k = decomposed_context_fa["n_components"]
    cols = [c for c in df.columns if c.startswith('fa_component_')]
    assert len(cols) == k, f"FA expected {k} components, got {len(cols)}"

