"""
Phase 2: Column Mapping
Requirement: Ensure every alpha file corresponds to exactly one column.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase2.test_config import panel_context

def test_column_mapping(panel_context):
    """
    Verifies that every alpha file found on disk exists as a column in the panel.
    """
    universe = panel_context["universe"]
    period = panel_context["period"]
    df_clean = panel_context["df_clean"]
    dl = panel_context["dl"]

    if df_clean.empty:
        pytest.skip("Empty panel.")

    alpha_files = dl.get_alpha_files()
    
    for alpha_name in alpha_files.keys():
        assert alpha_name in df_clean.columns, f"Alpha file {alpha_name} is missing from the panel!"
        
    print(f"\n[OK] Column Mapping: All {len(alpha_files)} files correctly mapped to panel columns for {universe} {period}.")
