"""
Phase 7: Group-By Consistency
Requirement: Assert that the IC is calculated cross-sectionally (per date) before being averaged, rather than across the entire pool at once.
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

def test_groupby_consistency_cross_sectional(real_ic_context):
    """
    Verifies that 'ranking' IC is calculated per-date. 
    If we have only 2 dates, the result should be (IC_date1 + IC_date2) / 2.
    """
    ctx = real_ic_context
    df = ctx["df_real"].copy()
    alpha_cols = ctx["alpha_cols"]
    target_col = ctx["target_col"]
    
    # Take data for exactly 2 dates to make manual verification easy
    all_dates = df.index.unique().sort_values()
    if len(all_dates) < 2:
        pytest.skip("Not enough unique dates in sample for cross-sectional test.")
        
    date1, date2 = all_dates[0], all_dates[1]
    df_subset = df.loc[[date1, date2]].copy()
    
    test_feature = alpha_cols[0]
    X = df_subset[[test_feature]]
    y = df_subset[target_col]
    
    # Get the official result (ranking mode)
    results = calculate_ic_and_pvalue(X, y, task_type='ranking')
    actual_ic = results.loc[results['Feature'] == test_feature, 'IC'].values[0]
    
    # Calculate manually per date
    import scipy.stats as stats
    
    def get_date_ic(d):
        grp = df_subset.loc[[d]]
        feat_vals = grp[test_feature]
        targ_vals = grp[target_col]
        # Avoid ConstantInputWarning from scipy.stats.spearmanr
        if len(feat_vals.unique()) <= 1 or len(targ_vals.unique()) <= 1:
            return np.nan
        ic, _ = stats.spearmanr(feat_vals, targ_vals)
        return ic
    
    # Ensure each date has at least 2 tickers for a valid test
    if len(df_subset.loc[[date1]]) < 2 or len(df_subset.loc[[date2]]) < 2:
        pytest.skip("Sampled dates do not have enough tickers for cross-sectional IC calculation.")

    ic1 = get_date_ic(date1)
    ic2 = get_date_ic(date2)
    expected_ic = (ic1 + ic2) / 2
    
    # Use equal_nan=True because if both are NaN, they are technically consistent
    assert np.isclose(actual_ic, expected_ic, equal_nan=True), \
        f"Cross-sectional average mismatch. Expected {expected_ic}, got {actual_ic}"
