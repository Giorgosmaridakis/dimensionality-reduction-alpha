"""
Phase 3: Zero Leakage
Requirement: Verify that no part of the calculation of the 'Target' uses any information that would not have been available at the end of day T.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase3.test_config import merged_context

def test_zero_leakage(merged_context):
    """
    Verify that the Target is not the current same-day close-to-close return.

    With the current close-only execution convention, Target_T should be the
    adjusted-close return from T+1 to T+2, not the return ending at T.
    """
    universe = merged_context["universe"]
    period = merged_context["period"]
    final_df = merged_context["final_df"]
    dl = merged_context["dl"]
    
    if final_df.empty:
        pytest.skip("Empty dataset.")

    # 1. Load adjusted close, which is the source used by DatasetBuilder.
    adjusted_close = dl.get_adjusted_close().sort_index()
    same_day_returns = adjusted_close.pct_change()

    # 2. Sample rows
    sample_size = min(100, len(final_df))
    samples = final_df.sample(sample_size)
    
    leaks = []
    for _, row in samples.iterrows():
        date = row['Date']
        ticker = row['ticker']
        target_val = row['Target']
        
        # Check current same-day return ending at the signal date. Equal values
        # can happen by coincidence, especially when both returns are 0.0, so
        # only flag leakage when the correct future target is different.
        try:
            current_day_return = same_day_returns.loc[date, ticker]
            ticker_prices = adjusted_close[ticker].dropna()
            future_dates = ticker_prices.index[ticker_prices.index > date]
            if len(future_dates) < 2:
                continue

            entry_date = future_dates[0]
            exit_date = future_dates[1]
            expected_future_return = ticker_prices.loc[exit_date] / ticker_prices.loc[entry_date] - 1

            if (
                np.isclose(target_val, current_day_return, atol=1e-8)
                and not np.isclose(expected_future_return, current_day_return, atol=1e-8)
                and not np.isnan(target_val)
            ):
                leaks.append(f"Leakage detected @ {date} for {ticker}: Target matches CURRENT day return ({target_val})")
        except KeyError:
            continue

    if leaks:
        pytest.fail(f"Zero Leakage failed! {len(leaks)} cases where Target == Current Return.\n" + "\n".join(leaks[:5]))
        
    print(f"\n[OK] Zero Leakage verified for {universe} {period}: No same-day return alignment found.")
