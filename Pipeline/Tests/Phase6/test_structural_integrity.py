"""
Phase 6: Structural Integrity
Requirement: Ensure the transformed DataFrame produced by the Neural Extractors
(GKX autoencoder / FactorVAE) is structurally correct:
  - Is non-empty (has rows)
  - Has no duplicate (Date, ticker) pairs
  - Contains a valid 'Date' column
  - Max date does not exceed the period's declared end year
  - Row count is identical to the original scaled DataFrame (neural extractors must not drop or add rows)
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase6.test_config import ae_vanilla_context

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
    """Core assertions shared by neural feature-extraction contexts."""
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
        duplicates = df.duplicated(subset=["Date"]).sum()
        assert duplicates == 0, (
            f"[{label}] Found {duplicates} duplicate Date rows for period '{period}'"
        )

    # --- 6. Row count is preserved vs. the original scaled DataFrame ---
    assert len(df) == len(df_original), (
        f"[{label}] Row count mismatch for period '{period}': "
        f"original has {len(df_original)} rows, transformed has {len(df)} rows. "
        f"Neural extractor must not drop or add rows."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ae_structural_integrity(ae_vanilla_context):
    """
    Verifies that the Autoencoder-transformed DataFrame is structurally correct:
    non-empty, valid Date column, max date within period bounds, no duplicates,
    and row count matches the original scaled DataFrame.
    Also exports the DataFrame to a CSV for manual inspection.
    """
    _assert_structural_integrity(ae_vanilla_context, label="AE")

    # --- Export for manual inspection ---
    df: pd.DataFrame = ae_vanilla_context["df_transformed"]
    universe = ae_vanilla_context["universe"].replace("&", "and")
    period   = ae_vanilla_context["period"]
    out_path = DEBUG_OUTPUT_DIR / f"ae_{universe}_{period}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[DEBUG] AE output saved to: {out_path}")

