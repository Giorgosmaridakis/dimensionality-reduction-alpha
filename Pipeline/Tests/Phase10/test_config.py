from __future__ import annotations
import pytest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder

@pytest.fixture(scope="module")
def simulation_context():
    """
    Fixture that provides real financial data spanning multiple years 
    for walking-forward orchestration testing.
    """
    loader = DataLoader(universe='ETFs', time_period='2006-2022')
    if not loader.universe_path.exists():
         pytest.skip("Data path not found for Phase 10 testing.")
         
    builder = DatasetBuilder(dropna=True)
    
    # Load a subset of alphas
    all_files = loader.get_alpha_files()
    if not all_files:
         pytest.skip("No alpha data found for Phase 10 simulation context.")
    
    subset_keys = list(all_files.keys())[:3]
    old_get_alphas = loader.get_alpha_files
    loader.get_alpha_files = lambda: {k: all_files[k] for k in subset_keys}
    
    try:
        panel_df = builder.prepare_stacked_data(loader)
        if panel_df.empty:
             pytest.skip("Panel construction failed in Phase 10 config.")
             
        final_df = builder.merge_features_and_targets(panel_df, loader)
    finally:
        loader.get_alpha_files = old_get_alphas
    
    # Ensure DatetimeIndex
    if builder.index_date_name in final_df.columns:
        final_df = final_df.set_index(builder.index_date_name)
    final_df.index = pd.to_datetime(final_df.index)
    final_df = final_df.sort_index()
    
    alpha_cols = [c for c in final_df.columns if c.startswith('alpha')]
    
    # Check for enough years
    unique_years = final_df.index.year.unique()
    if len(unique_years) < 2:
         pytest.skip(f"Not enough years in dataset ({len(unique_years)}) for walking-forward test.")
         
    return final_df, alpha_cols
