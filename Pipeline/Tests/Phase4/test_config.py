"""
Phase 4: Statistical Normalization (Standardization)
Requirement: Ensure features have zero mean and unit variance.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from features.scaling import FeatureScaler

# Phase 4 Test Configuration Matrix
# ================================
TEST_PARAMS = [
    ("ETFs", "2006-2022"),
    ("ETFs", "2023-2025"),
    ("S&P500", "2020-2024"),
    ("S&P500", "2025-End")
]

@pytest.fixture(params=TEST_PARAMS, scope="session")
def scaled_context(request):
    """
    Builds the final merged dataset and applies Statistical Normalization (Z-score).
    Caches the result per (universe, period) for the entire session.
    """
    universe, period = request.param
    dl = DataLoader(universe=universe, time_period=period)
    
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder = DatasetBuilder(dropna=True)
    scaler = FeatureScaler()
    
    # 1. Step 2 logic: Panel stacking
    panel_df = builder.prepare_stacked_data(dl)
    if panel_df.empty:
        pytest.skip(f"No alpha data found for {universe} {period}")
        
    # 2. Step 3 logic: Target merging
    final_df = builder.merge_features_and_targets(panel_df, dl)
    
    # 3. Step 4 logic: Scaling
    alpha_cols = [c for c in final_df.columns if c.startswith('alpha')]
    df_scaled = scaler.fit_transform_standardize(final_df)
    
    return {
        "universe": universe,
        "period": period,
        "dl": dl,
        "alpha_cols": alpha_cols,
        "df_scaled": df_scaled,
        "scaler": scaler,
        "df_raw": final_df
    }
