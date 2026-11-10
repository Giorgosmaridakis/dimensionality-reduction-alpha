from __future__ import annotations
import pytest
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

@pytest.fixture
def tuning_data():
    """Provides a real project dataset for testing HyperparameterTuner."""
    universe = "ETFs"
    period = "2006-2022"
    
    dl = DataLoader(universe=universe, time_period=period)
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder = DatasetBuilder(dropna=True)
    
    # Load first 3 alphas for speed and variety
    all_files = dl.get_alpha_files()
    if not all_files:
        pytest.skip(f"No alpha data found for {universe} {period}")
        
    subset_keys = list(all_files.keys())[:3]
    old_get_alphas = dl.get_alpha_files
    dl.get_alpha_files = lambda: {k: all_files[k] for k in subset_keys}
    
    try:
        panel_df = builder.prepare_stacked_data(dl)
        if panel_df.empty:
             pytest.skip(f"Empty panel for {universe} {period}")
             
        final_df = builder.merge_features_and_targets(panel_df, dl)
    finally:
        dl.get_alpha_files = old_get_alphas # Restore

    if final_df.empty:
        pytest.skip(f"Could not merge targets for {universe} {period}")
        
    alpha_cols = [c for c in final_df.columns if c.startswith('alpha')]
    
    # Sort by Date to ensure TimeSeriesSplit works correctly
    # Sorting by Date then Ticker provides stable cross-sectional slices
    final_df = final_df.sort_values(by=[builder.index_date_name, builder.ticker_name])
    
    # Set index to Date as expected by HyperparameterTuner
    final_df = final_df.set_index(builder.index_date_name)
    
    # Sample 1500 rows to ensure enough dates for 3 splits + embargo (20+ dates)
    # 1500 rows cover about 3 weeks of ETF data if ~100 tickers/day
    df_sampled = final_df.head(1500).copy()
    
    X = df_sampled[alpha_cols]
    y = df_sampled["Target"]
    # qids should be the Date index itself (as a Series)
    qids = pd.Series(df_sampled.index, index=df_sampled.index)
    
    return X, y, qids

@pytest.fixture
def mock_preprocessor():
    """A mock preprocessor to verify flag routing."""
    class MockPreprocessor:
        def __init__(self):
            self.called_methods = []
            self.pca_col_names = ["pca1", "pca2"]
            self.fa_col_names = ["fa1"]
            self.kpca_col_names = ["kpca1"]
            self.ae_col_names = ["ae1"]
            
        def fit_transform_standardize(self, X, features):
            self.called_methods.append("standardize")
            return X
            
        def transform_standardize(self, X, features):
            return X
            
        def fit_transform_pca(self, X, features, n_components):
            self.called_methods.append("pca")
            return pd.DataFrame(np.random.randn(len(X), 2), index=X.index, columns=self.pca_col_names)
            
        def transform_pca(self, X):
            return pd.DataFrame(np.random.randn(len(X), 2), index=X.index, columns=self.pca_col_names)
            
        def fit_transform_fa(self, X, features, n_components):
            self.called_methods.append("fa")
            return pd.DataFrame(np.random.randn(len(X), 1), index=X.index, columns=self.fa_col_names)

        def transform_fa(self, X):
            return pd.DataFrame(np.random.randn(len(X), 1), index=X.index, columns=self.fa_col_names)
            
    return MockPreprocessor()
