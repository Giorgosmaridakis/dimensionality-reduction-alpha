"""
Phase 3: Temporal Causality
Requirement: Prove that for every signal row at time T, the Target column
contains the tradable adjusted-close return from T+1 to T+2.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase3.test_config import merged_context

def test_temporal_causality(merged_context):
    """
    Prove that every row at time T is aligned to the return earned after
    next-close execution:

        Target_T = Adjusted_Close_{T+2} / Adjusted_Close_{T+1} - 1
    """
    universe = merged_context["universe"]
    period = merged_context["period"]
    final_df = merged_context["final_df"]
    dl = merged_context["dl"]
    
    if final_df.empty:
        pytest.skip("Empty dataset.")

    # 1. Load adjusted close, which is the source used by DatasetBuilder.
    adjusted_close = dl.get_adjusted_close().sort_index()

    # 2. Sample 100 rows to verify causality
    sample_size = min(100, len(final_df))
    samples = final_df.sample(sample_size)
    
    mismatches = []
    for _, row in samples.iterrows():
        date = row['Date']
        ticker = row['ticker']
        target_val = row['Target']
        
        if ticker not in adjusted_close.columns:
            mismatches.append(f"{ticker} not found in Adjusted_Close.csv")
            continue

        ticker_prices = adjusted_close[ticker].dropna()
        future_dates = ticker_prices.index[ticker_prices.index > date]
        
        if len(future_dates) < 2:
            mismatches.append(f"Not enough future adjusted-close dates for {ticker} at {date}")
            continue
            
        entry_date = future_dates[0]
        exit_date = future_dates[1]
        expected_return = ticker_prices.loc[exit_date] / ticker_prices.loc[entry_date] - 1
        
        if not np.isclose(target_val, expected_return, atol=1e-8):
            mismatches.append(
                f"Causality Break @ {date} for {ticker}: Got {target_val}, "
                f"Expected {expected_return} ({entry_date} -> {exit_date})"
            )

    if mismatches:
        pytest.fail(f"Temporal Causality failed with {len(mismatches)} errors!\n" + "\n".join(mismatches[:5]))
        
    print(f"\n[OK] Temporal Causality verified for {universe} {period}: {sample_size} samples checked.")
