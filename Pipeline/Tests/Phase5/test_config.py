import pytest
import sys
from pathlib import Path
from functools import lru_cache

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from features.scaling import FeatureScaler
from features.decomposition import FeatureDecomposer

# Phase 5 Test Configuration Matrix
# ================================
TEST_PARAMS = [
    ("ETFs", "2006-2022"),
    ("ETFs", "2023-2025"),
    ("S&P500", "2020-2024"),
    ("S&P500", "2025-End")
]

@lru_cache(maxsize=None)
def get_base_context(universe, period):
    dl = DataLoader(universe=universe, time_period=period)
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder = DatasetBuilder(dropna=True)
    scaler = FeatureScaler()
    
    panel_df = builder.prepare_stacked_data(dl)
    if panel_df.empty:
        pytest.skip(f"No alpha data found for {universe} {period}")
        
    final_df = builder.merge_features_and_targets(panel_df, dl)
    
    alpha_cols = [c for c in final_df.columns if c.startswith('alpha')]
    df_scaled = scaler.fit_transform_standardize(final_df)
    
    return {
        "universe": universe,
        "period": period,
        "dl": dl,
        "alpha_cols": alpha_cols,
        "df_scaled": df_scaled
    }

@pytest.fixture(params=TEST_PARAMS, scope="session")
def decomposed_context_pca(request):
    """Fixture for PCA decomposition."""
    ctx = get_base_context(*request.param)
    decom = FeatureDecomposer()
    n_comp = 5
    df_pca = decom.fit_transform_pca(ctx["df_scaled"], ctx["alpha_cols"], n_components=n_comp)
    
    return {
        **ctx,
        "decom": decom,
        "df_transformed": df_pca,
        "n_components": n_comp,
        "type": "pca"
    }

@pytest.fixture(params=TEST_PARAMS, scope="session")
def decomposed_context_fa(request):
    """Fixture for Factor Analysis decomposition."""
    ctx = get_base_context(*request.param)
    decom = FeatureDecomposer()
    n_comp = 5
    df_fa = decom.fit_transform_fa(ctx["df_scaled"], ctx["alpha_cols"], n_components=n_comp)
    
    return {
        **ctx,
        "decom": decom,
        "df_transformed": df_fa,
        "n_components": n_comp,
        "type": "fa"
    }
