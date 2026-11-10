import re
import pytest
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import DataLoader
from Tests.Phase1.test_config import EXPECTED_ALPHA_COUNTS, TEST_PARAMS

"""
Phase 1: Real-World File Discovery
Requirement: Verify every universe folder contains the exact signals expected.
"""

# Strict naming rule:
#   - Must start with exactly "alpha"
#   - Followed by exactly 3 digits
#   - The numeric value must be between 001 and 101 (inclusive)
#   - Must end with exactly ".csv"
#   - No parentheses, spaces, suffixes, or any other characters allowed
_ALPHA_FILENAME_RE = re.compile(r'^alpha(\d{3})\.csv$')
_ALPHA_NUMBER_MIN = 1
_ALPHA_NUMBER_MAX = 101


@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_alpha_file_counts(universe, period):
    """
    Scans the physical disk and compares found counts to discovery_config.py.
    """
    expected = EXPECTED_ALPHA_COUNTS[universe][period]
    dl = DataLoader(universe=universe, time_period=period)
    
    # Check if the data path even exists before running
    if not dl.universe_path.exists():
        pytest.skip(f"Dataset path {dl.universe_path} not found. Skipping...")
        
    discovered = dl.get_alpha_files()
    found_count = len(discovered)
    
    assert found_count == expected, (
        f"Mismatch in {universe} ({period}).\n"
        f"Config expected: {expected}\n"
        f"Found on Disk: {found_count}"
    )
    print(f"\n[OK] {universe} {period}: Found {found_count} signals.")


@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_alpha_filename_format(universe, period):
    """
    Validates that every alpha file on disk follows the strict naming convention:
      - Exact format: alpha<NNN>.csv  (e.g. alpha001.csv, alpha101.csv)
      - Exactly 3 digits — no more, no fewer
      - Numeric value between 001 and 101 inclusive
      - No parentheses, spaces, suffixes, or any extra characters
      - File extension must be exactly '.csv' (lowercase)

    Examples of VALID names  : alpha001.csv, alpha057.csv, alpha101.csv
    Examples of INVALID names: alpha1.csv, alpha(001).csv, alpha001 (1).csv,
                                Alpha001.csv, alpha001.CSV, alpha102.csv
    """
    dl = DataLoader(universe=universe, time_period=period)

    if not dl.universe_path.exists():
        pytest.skip(f"Dataset path {dl.universe_path} not found. Skipping...")

    # Only scan inside directories named EXACTLY 'zuras'.
    # Using rglob("alpha*.csv") would also crawl 'zuras_statistics/' and other
    # look-alike folders, producing false positives.
    period_path = dl.universe_path / period
    if not period_path.exists():
        pytest.skip(f"Period path {period_path} not found. Skipping...")

    exact_zuras_dirs = [
        d for d in period_path.rglob("*")
        if d.is_dir() and d.name == "zuras"
    ]

    all_csv_files = []
    for zuras_dir in exact_zuras_dirs:
        all_csv_files.extend(zuras_dir.glob("alpha*.csv"))

    if not all_csv_files:
        pytest.skip(f"No alpha*.csv files found in any exact 'zuras/' folder under {period_path}.")

    bad_names = []
    for file_path in all_csv_files:
        filename = file_path.name
        match = _ALPHA_FILENAME_RE.match(filename)

        if not match:
            bad_names.append(f"  BAD FORMAT : {file_path}")
            continue

        number = int(match.group(1))
        if not (_ALPHA_NUMBER_MIN <= number <= _ALPHA_NUMBER_MAX):
            bad_names.append(
                f"  OUT OF RANGE ({number:03d} not in "
                f"{_ALPHA_NUMBER_MIN:03d}–{_ALPHA_NUMBER_MAX:03d}): {file_path}"
            )

    assert not bad_names, (
        f"\n[{universe} | {period}] {len(bad_names)} file(s) violate the naming convention "
        f"'alpha<001-101>.csv':\n" + "\n".join(bad_names)
    )
    print(
        f"\n[OK] Filename Format: All {len(all_csv_files)} alpha files in "
        f"{universe} {period} follow the correct naming convention."
    )
