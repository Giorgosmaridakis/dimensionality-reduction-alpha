"""
Phase 4: Out-of-Sample Integrity
Requirement: Fit the scaler on a 'Train' period and apply it to a 'Test' period. Verify that the 'Test' data is scaled using the mean and std of the 'Train' data, not its own.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from features.scaling import FeatureScaler

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase4.test_config import scaled_context

def test_oos_integrity(scaled_context):
    """
    Fit the scaler on 'Train' data and apply it to 'Test' data. 
    Verify that the 'Test' data is scaled using the mean and std of 'Train'.
    """
    alpha_cols = scaled_context["alpha_cols"]
    df_raw = scaled_context["df_raw"]
    universe = scaled_context["universe"]
    period = scaled_context["period"]
    
    if df_raw.empty or len(df_raw) < 10:
        pytest.skip("Not enough data for split.")

    # 1. Split data (First 70% Train, last 30% Test)
    split_idx = int(len(df_raw) * 0.7)
    df_train = df_raw.iloc[:split_idx].copy()
    df_test = df_raw.iloc[split_idx:].copy()
    
    # 2. Fit and Transform Train
    new_scaler = FeatureScaler()
    df_train_scaled = new_scaler.fit_transform_standardize(df_train)
    
    # 3. Transform Test (Out-of-Sample)
    df_test_scaled = new_scaler.transform_standardize(df_test)
    
    # 4. Manual Verification for a random sample
    col = alpha_cols[0]
    train_mean = df_train[col].mean()
    train_std = df_train[col].std()
    if train_std == 0 or np.isnan(train_std): train_std = 1.0
    
    raw_val = df_test[col].iloc[0]
    expected_scaled_val = (raw_val - train_mean) / train_std
    actual_scaled_val = df_test_scaled[col].iloc[0]
    
    assert np.isclose(actual_scaled_val, expected_scaled_val, atol=1e-8), \
        f"OOS Integrity breach: Test data not scaled with Train parameters in {universe} {period}!"

    print(f"\n[OK] OS-OOS Integrity: Test data transformation verified using Train parameters.")
