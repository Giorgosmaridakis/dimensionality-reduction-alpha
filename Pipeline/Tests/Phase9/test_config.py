from __future__ import annotations
import pytest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix path to allow importing from Pipeline
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder

@pytest.fixture(scope="module")
def model_data():
    """
    Fixture that provides real financial data for GBDT model testing.
    Returns (X, y, qids, alpha_cols).
    """
    loader = DataLoader(universe='ETFs', time_period='2006-2022')
    if not loader.universe_path.exists():
         pytest.skip("Data path not found for Phase 9 testing.")
         
    builder = DatasetBuilder(dropna=True)
    
    # Temporarily limit alphas for speed in tests
    all_files = loader.get_alpha_files()
    if not all_files:
         pytest.skip("No alpha files found for Phase 9 real data testing.")
         
    subset_keys = list(all_files.keys())[:3]
    old_get_alphas = loader.get_alpha_files
    loader.get_alpha_files = lambda: {k: all_files[k] for k in subset_keys}
    
    try:
        panel_df = builder.prepare_stacked_data(loader)
        if panel_df.empty:
             pytest.skip("Panel construction failed in Phase 9 config.")
             
        final_df = builder.merge_features_and_targets(panel_df, loader)
    finally:
        loader.get_alpha_files = old_get_alphas # Restore
    
    alpha_cols = [c for c in final_df.columns if c.startswith('alpha')]
    
    # Ensure Date is index
    if builder.index_date_name in final_df.columns:
        final_df = final_df.set_index(builder.index_date_name)
    
    # Sort for stability
    final_df = final_df.sort_index()
    
    # Sample 1000 rows
    df_sampled = final_df.head(1000).copy()
    
    X = df_sampled[alpha_cols]
    y = df_sampled["Target"]
    # qids = ticker column
    qids = df_sampled[builder.ticker_name]
    
    return X, y, qids, alpha_cols
