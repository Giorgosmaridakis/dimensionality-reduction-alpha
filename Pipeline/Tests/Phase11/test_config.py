from __future__ import annotations
import pytest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add Pipeline root to sys.path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import DataLoader

@pytest.fixture(scope="module")
def backtest_data():
    """
    Fixture that provides real price data and synthetic predictions 
    for backtester verification.
    """
    loader = DataLoader(universe='ETFs', time_period='2006-2022')
    if not loader.universe_path.exists():
         pytest.skip("Data path not found for Phase 11 testing.")
         
    # Load Returns/Price data
    returns_path = loader.get_returns()
    if not returns_path:
        pytest.skip("Returns.csv missing for Phase 11 testing.")
        
    # We use the returns file to reconstruct a wide Close price frame for vectorbt
    returns_df = pd.read_csv(returns_path[0])
    # Assume first column is Date
    date_col = returns_df.columns[0]
    returns_df[date_col] = pd.to_datetime(returns_df[date_col])
    returns_df = returns_df.set_index(date_col)
    
    # Pick top 20 assets for faster testing
    returns_df = returns_df.iloc[:, :20].dropna()
    
    # Reconstruct prices from returns (Price_t = Price_0 * Exp(CumSum(LogReturns)))
    # For testing, we just care about price relatives
    prices_df = (1 + returns_df).cumprod()
    
    # Create synthetic predictions
    # We'll create a 'Predicted_Rank' based on the returns_df shifted by 1 day (mocking "perfect" signal)
    # to ensure the backtester has enough variance to trade.
    pred_data = []
    for date, row in returns_df.iterrows():
        for ticker, val in row.items():
            pred_data.append({
                'Date': date,
                'Ticker': ticker,
                'Prediction': val + np.random.normal(0, 0.001) # Add slight noise
            })
    
    predictions = pd.DataFrame(pred_data)
    
    return prices_df, predictions
