# Phase 1 Test Configuration Matrix
# ================================
# This file serves as the 'Source of Truth' for all Phase 1 tests.
# You can update the counts and parameters here to match your current data state.


# The list of (universe, period) pairs to iterate over in all tests.
# This ensures that ALL tests in Phase 1 run across the entire dataset.
TEST_PARAMS = [
    ("ETFs", "2006-2022"),
    ("ETFs", "2023-2025"),
    ("S&P500", "2020-2024"),
    ("S&P500", "2025-End")
]




# File Discovery
EXPECTED_ALPHA_COUNTS = {
    "ETFs": {
        "2006-2022": 82,  
        "2023-2025": 82
    },
    "S&P500": {
        "2020-2024": 82,
        "2025-End": 82
    }
}


