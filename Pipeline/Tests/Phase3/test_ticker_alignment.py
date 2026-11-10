"""
Phase 3: Ticker Alignment
Requirement: Ensure that the merge did not accidentally shift tickers, causing
features for one company to be aligned with another company's adjusted-close
target return.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase3.test_config import merged_context

def test_ticker_alignment(merged_context):
    """
    Ensure that the merge didn't accidentally shift tickers, causing 
    features for one company to be aligned with the returns of another.
    """
    universe = merged_context["universe"]
    period = merged_context["period"]
    final_df = merged_context["final_df"]
    dl = merged_context["dl"]
    
    if final_df.empty:
        pytest.skip("Empty dataset.")

    # 1. Pick 3 distinct tickers to audit
    all_tickers = final_df['ticker'].unique()
    audit_tickers = np.random.choice(all_tickers, size=min(3, len(all_tickers)), replace=False)
    
    # 2. Load the adjusted-close source used to build the target.
    adjusted_close = dl.get_adjusted_close().sort_index()
    
    # 3. Perform the audit for each selected ticker
    for ticker in audit_tickers:
        ticker_df = final_df[final_df['ticker'] == ticker]
        if ticker_df.empty: continue
        
        # INCREASED ROBUSTNESS: Pick 5 random rows per ticker instead of 1
        sample_size = min(5, len(ticker_df))
        samples = ticker_df.sample(sample_size)
        
        for _, sample_row in samples.iterrows():
            date = sample_row['Date']
            target_val = sample_row['Target']
            
            # Find values for this ticker in the pre-loaded source data.
            if ticker in adjusted_close.columns:
                ticker_prices = adjusted_close[ticker].dropna()
                future_dates = ticker_prices.index[ticker_prices.index > date]
                
                if len(future_dates) < 2:
                    continue # Skip if no tradable future window for this row.
                    
                entry_date = future_dates[0]
                exit_date = future_dates[1]
                expected_val = ticker_prices.loc[exit_date] / ticker_prices.loc[entry_date] - 1
                assert np.isclose(target_val, expected_val, atol=1e-8), \
                    f"Ticker Alignment mismatch for {ticker} at {date}! Master table has wrong identity."
            else:
                pytest.fail(f"Audit Ticker {ticker} not found in adjusted-close source!")

    print(f"\n[OK] Ticker Alignment verified for {universe} {period} across {len(audit_tickers)} tickers.")
