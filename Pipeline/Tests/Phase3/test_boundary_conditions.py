"""
Phase 3: Boundary Conditions
Requirement: Check what happens at the very end of the dataset--ensure dates
without enough future adjusted-close observations are handled correctly.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase3.test_config import merged_context

def test_boundary_conditions(merged_context):
    """
    Check what happens at the very end of the dataset.
    Ensure dates without a T+1 entry close and T+2 exit close are dropped.
    """
    universe = merged_context["universe"]
    period = merged_context["period"]
    final_df = merged_context["final_df"]
    dl = merged_context["dl"]
    
    if final_df.empty:
        pytest.skip("Empty dataset.")

    # 1. Find the absolute last date in the adjusted-close source.
    adjusted_close = dl.get_adjusted_close().sort_index()
    absolute_last_date = adjusted_close.index.max()
    
    # 2. Find the last date in our merged panel
    last_date_in_final = final_df['Date'].max()
    
    print(f"\n[INFO] {universe} {period}: Absolute last return date: {absolute_last_date}")
    print(f"[INFO] {universe} {period}: Last date in merged panel: {last_date_in_final}")
    
    # The last date in the panel must leave room for T+1 entry and T+2 exit.
    eligible_signal_dates = adjusted_close.index[:-2]
    assert last_date_in_final <= eligible_signal_dates.max(), \
        f"Boundary Condition failure: {last_date_in_final} should not have a tradable T+1 -> T+2 target!"

    print(f"[OK] Boundary Conditions verified: The final date correctly lacks a future target and was likely dropped.")
