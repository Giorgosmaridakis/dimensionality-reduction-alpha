"""
Phase 11: Risk Metrics
Requirement: Verify calculation of Sharpe Ratio, Max Drawdown, and Turnover.
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from Backtester.backtester import Backtester
from Tests.Phase11.test_config import backtest_data

def test_phase11_risk_metrics_sanity(backtest_data, tmp_path):
    """
    Verify that generated risk metrics (Sharpe, MDD) are within reasonable bounds 
    and present in the OOS metrics file.
    """
    prices_df, predictions = backtest_data
    bt = Backtester(output_folder=str(tmp_path))
    
    bt.run_backtest(prices_df, predictions, strategy='long_short')
    
    metrics_path = tmp_path / 'long_short_final_oos_metrics.csv'
    assert metrics_path.exists()
    
    metrics_df = pd.read_csv(metrics_path)
    
    sharpe = metrics_df['Sharpe Ratio'].values[0]
    mdd = metrics_df['Max Drawdown [%]'].values[0]
    
    # Check for basic sanity (metrics exist and are numbers)
    assert not np.isnan(sharpe), "Sharpe Ratio should be a number."
    assert mdd >= 0, "Max Drawdown as a percentage should be non-negative in vbt output."
    
    # Check Turnover
    time_series_path = tmp_path / 'long_short_time_series.csv'
    ts_df = pd.read_csv(time_series_path)
    assert 'Turnover' in ts_df.columns
    assert ts_df['Turnover'].mean() >= 0
