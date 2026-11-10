"""
Phase 11: Rank-to-Weight Mapping
Requirement: Verify that the size_df correctly implements the Strategy (e.g., Top 10% Long, Bottom 10% Short).
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
import os
from Backtester.backtester import Backtester
from Tests.Phase11.test_config import backtest_data

def test_phase11_weight_mapping_long_short(backtest_data, tmp_path):
    """
    Verify that for a 'long_short' strategy, the top 10% assets get positive weights 
    and bottom 10% get negative weights, summing to roughly 0 (net neutral).
    """
    prices_df, predictions = backtest_data
    output_dir = tmp_path / "backtest_results"
    
    bt = Backtester(output_folder=str(output_dir))
    
    # We want to inspect the size_df created internally
    # Since run_backtest exports it, we can read it back or mock vectorbt
    bt.run_backtest(prices_df, predictions, strategy='long_short')
    
    vbt_pos_path = output_dir / 'vbt_generated_positions.csv'
    assert vbt_pos_path.exists()
    
    size_df = pd.read_csv(vbt_pos_path, index_col=0)
    
    # Signals dated t execute on the next available adjusted close, so the
    # first exported row is expected to be flat after the one-bar shift.
    nonzero_rows = size_df.loc[size_df.abs().sum(axis=1) > 0]
    assert not nonzero_rows.empty, "Long-short strategy should create non-zero shifted target weights."

    first_trade_row = nonzero_rows.iloc[0]
    positive_weights = first_trade_row[first_trade_row > 0]
    negative_weights = first_trade_row[first_trade_row < 0]
    
    assert len(positive_weights) > 0, "Top 10% should have positive weights."
    assert len(negative_weights) > 0, "Bottom 10% should have negative weights."
    
    # Net exposure for long_short should be close to 0
    net_exposure = first_trade_row.sum()
    assert np.isclose(net_exposure, 0, atol=1e-5), f"Long-Short should be net neutral. Got {net_exposure}"
    
    # Gross exposure should be 1.0 (0.5 long + 0.5 short)
    gross_exposure = first_trade_row.abs().sum()
    assert np.isclose(gross_exposure, 1.0, atol=1e-5), f"Gross exposure should be 1.0. Got {gross_exposure}"
