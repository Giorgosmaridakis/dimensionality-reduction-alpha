"""
Phase 2: NaN Persistence
Requirement: Verify dropna logic works as intended.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase2.test_config import panel_context

def test_nan_persistence(panel_context):
    """
    Verify dropna logic:
    1. dropna=True should remove rows with NaNs.
    2. dropna=False should keep them.
    Also print how many NaNs are removed for each universe and timeframe.
    """
    universe = panel_context["universe"]
    period = panel_context["period"]
    df_raw = panel_context["df_raw"]
    df_clean = panel_context["df_clean"]

    if df_raw.empty:
        pytest.skip("No data found.")

    nans_removed = len(df_raw) - len(df_clean)
    
    print(f"\n[INFO] {universe} {period}: Total potential rows: {len(df_raw)}")
    print(f"[INFO] {universe} {period}: NaNs removed: {nans_removed}")
    print(f"[INFO] {universe} {period}: Final panel rows: {len(df_clean)}")

    # Check that df_clean has no NaNs
    assert not df_clean.isna().any().any(), "Dropped-NA panel still contains NaNs!"
    
    # Check that if there were NaNs in df_raw, length must be different
    has_nans = df_raw.isna().any().any()
    if has_nans:
        assert len(df_clean) < len(df_raw), "NaNs were found but not removed!"
    else:
        assert len(df_clean) == len(df_raw), "No NaNs found, but lengths differ!"
