"""
Phase 7: Rank vs Linear Logic
Requirement: Verify that for 'ranking' tasks, a Spearman (rank-based) correlation is used, and for 'regression', a Pearson correlation is used.
"""

from __future__ import annotations
import pytest
import sys
import numpy as np
import scipy.stats as stats
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

from Tests.Phase7.test_config import real_ic_context
from evaluation.metrics import calculate_ic_and_pvalue

def test_rank_vs_linear_logic_distinctness(real_ic_context):
    """Verifies that Spearman and Pearson correlations are correctly switched."""
    ctx = real_ic_context
    df = ctx["df_real"].copy()
    alpha_cols = ctx["alpha_cols"]
    target_col = ctx["target_col"]
    
    # We'll use one specific alpha
    test_feature = alpha_cols[0]
    X = df[[test_feature]]
    y = df[target_col]
    
    # Manually calculate Pearson using scipy
    mask = ~X[test_feature].isna() & ~y.isna()
    expected_pearson, _ = stats.pearsonr(X.loc[mask, test_feature], y.loc[mask])
    
    # Get results from metrics.py
    results_reg = calculate_ic_and_pvalue(X, y, task_type='regression')
    actual_pearson = results_reg.loc[results_reg['Feature'] == test_feature, 'IC'].values[0]
    
    # Use equal_nan=True because both being NaN (due to constant input) is technically 'consistent'
    assert np.isclose(actual_pearson, expected_pearson, equal_nan=True), \
        f"Pearson calculation mismatch. Expected {expected_pearson}, got {actual_pearson}"

    # Verify that Spearman returns a DIFFERENT value (as they are mathematically distinct)
    # This check is only meaningful if the values are not NaN
    results_rank = calculate_ic_and_pvalue(X, y, task_type='ranking')
    actual_spearman = results_rank.loc[results_rank['Feature'] == test_feature, 'IC'].values[0]
    
    if not np.isnan(actual_pearson) and not np.isnan(actual_spearman):
        # They should not be identical unless the sample is perfectly linear and ordered
        assert not np.isclose(actual_pearson, actual_spearman), \
            "Pearson and Spearman should typically yield distinct values for real data."
