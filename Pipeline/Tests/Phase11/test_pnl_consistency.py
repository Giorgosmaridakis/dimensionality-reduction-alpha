"""
Phase 11: PnL Consistency
Requirement: Verify that daily PnL calculation matches the (Weight * Daily Return) logic.
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from Backtester.backtester import Backtester
from Tests.Phase11.test_config import backtest_data

def test_phase11_pnl_consistency(backtest_data, tmp_path):
    """
    Manually calculate PnL for 1 day and compare with Backtester output.
    """
    
    prices_df, predictions = backtest_data
    bt = Backtester(output_folder=str(tmp_path))
    
    # Run backtest
    bt.run_backtest(prices_df, predictions, strategy='long')
    
    # Load daily time series
    ts_path = tmp_path / 'long_time_series.csv'
    ts_df = pd.read_csv(ts_path)
    
    # Load generated positions
    pos_path = tmp_path / 'vbt_generated_positions.csv'
    pos_df = pd.read_csv(pos_path, index_col=0)
    pos_df.index = pd.to_datetime(pos_df.index)
    
    # Calculate returns
    returns_df = prices_df.pct_change().dropna()
    
    # Pick a day (e.g., the second day of the backtest)
    test_date = returns_df.index[1]
    prev_date = returns_df.index[0]
    
    # Portfolio value on prev_date
    prev_val = ts_df[ts_df['Date'].str.contains(str(prev_date.date()))]['Cumulative PnL(%)'].values[0]
    # (Simplified check: verify PnL% vs manual Weight * Return)
    
    # Backtester uses TargetPercent at 'Date', which executes on next available close?
    # In vectorbt from_orders with targetpercent, it usually applies the target at the start of the next period.
    # However, let's just check if 'Total PnL(%)' is roughly consistent with simple arithmetic.
    
    actual_pnl_pct = ts_df[ts_df['Date'].str.contains(str(test_date.date()))]['Total PnL(%)'].values[0]
    
    weights = pos_df.loc[prev_date] # Weights set on prev_date
    rets = returns_df.loc[test_date] # Returns realized on test_date
    
    expected_pnl_pct = (weights * rets).sum() * 100
    
    # Note: VectorBT might have slight differences due to transaction costs or rounding
    assert np.isclose(actual_pnl_pct, expected_pnl_pct, atol=0.1), \
        f"Daily PnL% {actual_pnl_pct} deviates from expected {expected_pnl_pct}"
