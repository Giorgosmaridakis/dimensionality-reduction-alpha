from __future__ import annotations
import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder

# Phase 7 Test Configuration Matrix
# ================================
TEST_PARAMS = [
    ("ETFs", "2006-2022")
]

def get_base_context(universe, period):
    """Loads a real project panel with target column for IC testing."""
    dl = DataLoader(universe=universe, time_period=period)
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder = DatasetBuilder(dropna=True)
    
    # Load first 2 alphas for speed
    all_files = dl.get_alpha_files()
    subset_keys = list(all_files.keys())[:2]
    old_get_alphas = dl.get_alpha_files
    dl.get_alpha_files = lambda: {k: all_files[k] for k in subset_keys}
    
    panel_df = builder.prepare_stacked_data(dl)
    dl.get_alpha_files = old_get_alphas # Restore

    if panel_df.empty:
        pytest.skip(f"No alpha data found for {universe} {period}")
        
    # Merge targets
    final_df = builder.merge_features_and_targets(panel_df, dl)
    
    if final_df.empty:
        pytest.skip(f"Could not merge targets for {universe} {period}")
        
    alpha_cols = [c for c in final_df.columns if c.startswith('alpha')]
    
    # Sort by Date and Ticker to ensure cross-sectional variety in the sampled head
    final_df = final_df.sort_values(by=[builder.index_date_name, builder.ticker_name])
    
    # Set index to Date as expected by ranking IC functions
    final_df = final_df.set_index(builder.index_date_name)
    
    # Sample 500 rows to ensure multiple dates for cross-sectional group-by
    df_sampled = final_df.head(500).copy()
    
    return {
        "universe": universe,
        "period": period,
        "df_real": df_sampled,
        "alpha_cols": alpha_cols,
        "target_col": "Target"
    }

@pytest.fixture(params=TEST_PARAMS, scope="session")
def real_ic_context(request):
    """Fixture for real data IC calculation context."""
    return get_base_context(*request.param)
