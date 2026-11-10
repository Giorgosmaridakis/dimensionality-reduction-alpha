"""Walk-forward experiment package."""

import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from .forecaster import WalkingForwardForecaster

__all__ = ["WalkingForwardForecaster"]
