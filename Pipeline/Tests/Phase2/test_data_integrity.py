"""
Phase 2: Data Integrity
Requirement: Pick random samples and verify against source CSVs.
"""

import pytest
import numpy as np
import pandas as pd
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase2.test_config import panel_context

def test_data_integrity(panel_context):
    """
    Rigorously verifies a sample of the panel data against original source CSVs.
    """
    universe = panel_context["universe"]
    period = panel_context["period"]
    df_clean = panel_context["df_clean"]
    dl = panel_context["dl"]

    if df_clean.empty:
        pytest.skip("Empty panel.")

    # 1. Selection: Pick a few rows
    sample_size = min(500, max(20, int(len(df_clean) * 0.05)))
    samples = df_clean.sample(n=sample_size)
    
    # 2. Pick alpha columns to audit
    alpha_cols = [c for c in df_clean.columns if c.startswith('alpha')]
    if not alpha_cols:
        pytest.fail("No alpha columns found!")
    
    audit_alphas = random.sample(alpha_cols, k=min(3, len(alpha_cols)))
    
    # 3. Pre-load source CSVs once
    alpha_files = dl.get_alpha_files()
    source_data = {}
    for alpha in audit_alphas:
        df_src = pd.read_csv(alpha_files[alpha][0], index_col=0)
        df_src.index = pd.to_datetime(df_src.index)
        source_data[alpha] = df_src

    # 4. Perform the audit
    mismatches = []
    for _, row in samples.iterrows():
        ticker = row['ticker']
        date = row['Date']
        
        for alpha in audit_alphas:
            panel_val = row[alpha]
            source_val = source_data[alpha].loc[date, ticker]
            
            if not np.isclose(panel_val, source_val, atol=1e-8, equal_nan=True):
                mismatches.append(f"{alpha} Mismatch: Ticker={ticker}, Date={date}, Panel={panel_val}, Source={source_val}")

    if mismatches:
        pytest.fail(f"Integrity Audit Failed with {len(mismatches)} errors!\n" + "\n".join(mismatches[:5]))
    
    print(f"\n[OK] Data Integrity: {sample_size} samples matched source CSVs for {universe} {period}.")
