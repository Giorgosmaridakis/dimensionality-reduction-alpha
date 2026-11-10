"""
Phase 7: Perfect Correlation Test
Requirement: Create a feature that is a 100% copy of the Target; the function must return an IC of 1.0.
"""

from __future__ import annotations
import pytest
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

from Tests.Phase7.test_config import real_ic_context
from evaluation.metrics import calculate_ic_and_pvalue

def test_perfect_correlation_ic_is_one(real_ic_context):
    """Verifies that a feature perfectly matching the target yields IC = 1.0."""
    ctx = real_ic_context
    df = ctx["df_real"].copy()
    target_col = ctx["target_col"]
    
    # Construct a perfect signal
    df['perfect_alpha'] = df[target_col]
    
    X = df[['perfect_alpha']]
    y = df[target_col]
    
    # Test for Regression (Pearson)
    results_reg = calculate_ic_and_pvalue(X, y, task_type='regression')
    ic_reg = results_reg.loc[results_reg['Feature'] == 'perfect_alpha', 'IC'].values[0]
    assert np.isclose(ic_reg, 1.0), f"Pearson IC for perfect signal should be 1.0, got {ic_reg}"
    
    # Test for Ranking (Spearman)
    # Set index to Date for group-by
    X_indexed = X.copy()
    X_indexed.index = df.index
    results_rank = calculate_ic_and_pvalue(X_indexed, y, task_type='ranking')
    ic_rank = results_rank.loc[results_rank['Feature'] == 'perfect_alpha', 'IC'].values[0]
    assert np.isclose(ic_rank, 1.0), f"Spearman IC for perfect signal should be 1.0, got {ic_rank}"
