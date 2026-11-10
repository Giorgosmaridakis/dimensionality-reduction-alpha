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
from features.scaling import FeatureScaler
from features.neural_extractors import NeuralFeatureExtractor

# Phase 6 Test Configuration Matrix
# ================================
TEST_PARAMS = [
    ("ETFs", "2006-2022")
]

def get_base_context(universe, period):
    """Loads and scales real project data."""
    dl = DataLoader(universe=universe, time_period=period)
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder = DatasetBuilder(dropna=True)
    scaler = FeatureScaler()
    
    # Sample only a subset of alpha files for speed
    all_files = dl.get_alpha_files()
    subset_keys = list(all_files.keys())[:2]
    old_get_alphas = dl.get_alpha_files
    dl.get_alpha_files = lambda: {k: all_files[k] for k in subset_keys}
    
    panel_df = builder.prepare_stacked_data(dl)
    dl.get_alpha_files = old_get_alphas # Restore

    if panel_df.empty:
        pytest.skip(f"No alpha data found for {universe} {period}")

    panel_df = builder.merge_features_and_targets(panel_df, dl, target_col="Target")
    if panel_df.empty or "Target" not in panel_df.columns:
        pytest.skip(f"No target returns found for {universe} {period}")
        
    alpha_cols = [c for c in panel_df.columns if c.startswith('alpha')]
    
    # Increase sample size to 200 for better statistical stability in training
    df_sampled = panel_df.head(200).copy()
    
    # Second, standardize THAT specific sample
    # This ensures mean approx 0 and std approx 1 for the data exactly as fed to the NN
    df_scaled = scaler.fit_transform_standardize(df_sampled)
    
    return {
        "universe": universe,
        "period": period,
        "alpha_cols": alpha_cols,
        "df_scaled": df_scaled
    }

@pytest.fixture(params=TEST_PARAMS, scope="session")
def ae_vanilla_context(request):
    """Fixture for Vanilla Autoencoder Non-Linear Extraction context."""
    ctx = get_base_context(*request.param)
    extractor = NeuralFeatureExtractor()
    
    latent_dim = 5
    # Fit and transform
    df_transformed = extractor.fit_transform_autoencoder(
        ctx["df_scaled"], 
        ctx["alpha_cols"], 
        latent_dim=latent_dim, 
        target=ctx["df_scaled"]["Target"],
        max_epochs=50,
        patience=5,
        learning_rate=1e-2 # Slightly higher for quick convergence
    )
    
    return {
        **ctx,
        "extractor": extractor,
        "model": extractor.autoencoder,
        "history": extractor.ae_history,
        "df_transformed": df_transformed,
        "latent_dim": latent_dim,
        "type": "gkx_conditional_ae"
    }
