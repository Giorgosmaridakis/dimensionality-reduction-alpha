"""
Phase 5: Structural Integrity
Requirement: Ensure the transformed DataFrame is structurally correct:
  - Is non-empty (has rows)
  - Has no duplicate (Date, ticker) pairs
  - Contains a valid 'Date' column
  - Max date does not exceed the period's declared end year
  - Row count is identical to the original scaled DataFrame (PCA/FA must not drop or add rows)
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase5.test_config import decomposed_context_pca, decomposed_context_fa

# Directory where debug CSVs are written
DEBUG_OUTPUT_DIR = Path(__file__).parent / "debug_outputs"
DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_end_year(period: str) -> int:
    """
    Derives the expected end year from the period label string.
    e.g. '2006-2022' -> 2022, '2025-End' -> current year + 1 (open-ended).
    """
    end_token = period.split("-")[-1]
    if end_token.isdigit():
        return int(end_token)
    # "End" means the period is still open; use a far-future year as the bound
    return pd.Timestamp.now().year + 1


def _assert_structural_integrity(ctx, label: str):
    """Core assertions shared by PCA and FA contexts."""
    df: pd.DataFrame = ctx["df_transformed"]
    df_original: pd.DataFrame = ctx["df_scaled"]
    period: str = ctx["period"]

    # --- 1. Non-empty ---
    assert len(df) > 0, f"[{label}] DataFrame is empty for period '{period}'"

    # --- 2. Date column exists ---
    assert "Date" in df.columns, (
        f"[{label}] 'Date' column missing from transformed DataFrame for period '{period}'. "
        f"Columns found: {list(df.columns)}"
    )

    # --- 3. Date column is datetime ---
    df["Date"] = pd.to_datetime(df["Date"])
    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), (
        f"[{label}] 'Date' column is not datetime for period '{period}'"
    )

    # --- 4. Max date does not exceed the declared end year ---
    expected_end_year = _parse_end_year(period)
    actual_max_date = df["Date"].max()
    assert actual_max_date.year <= expected_end_year, (
        f"[{label}] Max date {actual_max_date.date()} exceeds expected end year "
        f"{expected_end_year} for period '{period}'"
    )

    # --- 5. No duplicate (Date, ticker) pairs ---
    if "ticker" in df.columns:
        duplicates = df.duplicated(subset=["Date", "ticker"]).sum()
        assert duplicates == 0, (
            f"[{label}] Found {duplicates} duplicate (Date, ticker) rows for period '{period}'"
        )
    else:
        # If no ticker column, just check for duplicate dates
        duplicates = df.duplicated(subset=["Date"]).sum()
        assert duplicates == 0, (
            f"[{label}] Found {duplicates} duplicate Date rows for period '{period}'"
        )

    # --- 6. Row count is preserved vs. the original scaled DataFrame ---
    assert len(df) == len(df_original), (
        f"[{label}] Row count mismatch for period '{period}': "
        f"original has {len(df_original)} rows, transformed has {len(df)} rows. "
        f"Decomposition must not drop or add rows."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pca_structural_integrity(decomposed_context_pca):
    """
    Verifies that the PCA-transformed DataFrame is structurally correct:
    non-empty, has valid dates, max date within period bounds, no duplicates.
    Also exports the DataFrame to a CSV for manual inspection.
    """
    _assert_structural_integrity(decomposed_context_pca, label="PCA")

    # --- Export for manual inspection ---
    df: pd.DataFrame = decomposed_context_pca["df_transformed"]
    universe = decomposed_context_pca["universe"].replace("&", "and")
    period   = decomposed_context_pca["period"]
    out_path = DEBUG_OUTPUT_DIR / f"pca_{universe}_{period}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[DEBUG] PCA output saved to: {out_path}")


def test_fa_structural_integrity(decomposed_context_fa):
    """
    Verifies that the FA-transformed DataFrame is structurally correct:
    non-empty, has valid dates, max date within period bounds, no duplicates.
    Also exports the DataFrame to a CSV for manual inspection.
    """
    _assert_structural_integrity(decomposed_context_fa, label="FA")

    # --- Export for manual inspection ---
    df: pd.DataFrame = decomposed_context_fa["df_transformed"]
    universe = decomposed_context_fa["universe"].replace("&", "and")
    period   = decomposed_context_fa["period"]
    out_path = DEBUG_OUTPUT_DIR / f"fa_{universe}_{period}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[DEBUG] FA output saved to: {out_path}")
