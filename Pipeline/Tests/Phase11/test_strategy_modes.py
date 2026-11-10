"""
Phase 11: Strategy Modes
Requirement: Ensure 'long', 'short', and 'long_short' strategies produce distinct and correct position profiles.
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from Backtester.backtester import Backtester
from Tests.Phase11.test_config import backtest_data

def test_phase11_strategy_distinctness(backtest_data, tmp_path):
    """
    Verify that 'long' only has positive weights and 'short' only has negative weights.
    """
    prices_df, predictions = backtest_data
    
    # Test LONG
    long_dir = tmp_path / "long"
    bt_long = Backtester(output_folder=str(long_dir))
    bt_long.run_backtest(prices_df, predictions, strategy='long')
    long_pos = pd.read_csv(long_dir / 'vbt_generated_positions.csv', index_col=0)
    assert (long_pos >= 0).all().all(), "Long strategy should not have negative weights."
    
    # Test SHORT
    short_dir = tmp_path / "short"
    bt_short = Backtester(output_folder=str(short_dir))
    bt_short.run_backtest(prices_df, predictions, strategy='short')
    short_pos = pd.read_csv(short_dir / 'vbt_generated_positions.csv', index_col=0)
    assert (short_pos <= 0).all().all(), "Short strategy should not have positive weights."
    
    # Verify they produce different cumulative returns
    metrics_long = pd.read_csv(long_dir / 'long_final_oos_metrics.csv')
    metrics_short = pd.read_csv(short_dir / 'short_final_oos_metrics.csv')
    
    ret_long = metrics_long['Annualized Return'].values[0]
    ret_short = metrics_short['Annualized Return'].values[0]
    assert ret_long != ret_short, "Long and Short strategies should result in different performance."
