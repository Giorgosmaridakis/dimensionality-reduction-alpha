"""
Information Coefficient helpers.

These are used for diagnostics, not portfolio execution. The backtester uses
the saved prediction ranks/scores directly.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

def calculate_ic_and_pvalue(X, y, task_type='regression'):
    """
    Calculates the Information Coefficient and p-value.
    - If task_type == 'ranking': Calculates Cross-Sectional Spearman Rank IC (grouped by Date).
    - If task_type == 'regression': Calculates Standard Pearson IC across all continuous data.
    """
    if task_type == 'ranking':
        # --- RANKING: CROSS-SECTIONAL SPEARMAN IC ---
        temp_df = X.copy()
        temp_df['Target'] = y
        feature_stats = {col: {'ic_sum': 0, 'pval_sum': 0, 'count': 0} for col in X.columns}
        
        # We assume X index is Date
        for date, group in temp_df.groupby(level=0):
            if len(group) > 1:
                target_vals = group['Target']
                for col in X.columns:
                    feature_vals = group[col]
                    valid_mask = ~feature_vals.isna() & ~target_vals.isna()
                    f_valid = feature_vals.loc[valid_mask]
                    t_valid = target_vals.loc[valid_mask]
                    
                    if len(f_valid) > 1 and len(f_valid.unique()) > 1 and len(t_valid.unique()) > 1:
                        ic, p_val = stats.spearmanr(f_valid, t_valid)
                        if not np.isnan(ic):
                            feature_stats[col]['ic_sum'] += ic
                            feature_stats[col]['pval_sum'] += p_val
                            feature_stats[col]['count'] += 1

        results = []
        for col, stats_dict in feature_stats.items():
            count = stats_dict['count']
            if count > 0:
                results.append({'Feature': col, 'IC': stats_dict['ic_sum'] / count, 'P_Value': stats_dict['pval_sum'] / count})
            else:
                results.append({'Feature': col, 'IC': np.nan, 'P_Value': np.nan})
                
    else:
        # --- REGRESSION: STANDARD PEARSON IC ---
        results = []
        for col in X.columns:
            valid_mask = ~X[col].isna() & ~y.isna()
            feature_valid = X.loc[valid_mask, col]
            target_valid = y.loc[valid_mask]
            
            if len(feature_valid) > 1:
                ic, p_val = stats.pearsonr(feature_valid, target_valid)
                results.append({'Feature': col, 'IC': ic, 'P_Value': p_val})
            else:
                results.append({'Feature': col, 'IC': np.nan, 'P_Value': np.nan})
                
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['Abs_IC'] = df_results['IC'].abs()
        df_results = df_results.sort_values(by='Abs_IC', ascending=False).drop(columns=['Abs_IC'])
    return df_results
