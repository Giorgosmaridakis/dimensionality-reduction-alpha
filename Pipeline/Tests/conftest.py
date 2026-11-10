"""Shared pytest fixtures for the pipeline test suite."""

from __future__ import annotations

import re
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# vectorbt imports numba-decorated helpers during test collection. On some
# Python/package layouts Numba cannot create cache locators for site-packages
# files, so keep test imports deterministic by disabling JIT compilation.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


@pytest.fixture
def tmp_path(request) -> Path:
    """Repo-local replacement for pytest tmp_path.

    The default pytest temp factory creates directories with permissions that
    can be unreadable in the Windows sandbox used for this project. Keeping test
    scratch files under ``Pipeline/temp`` makes the tests deterministic and
    accessible while preserving the normal ``tmp_path`` interface.
    """
    root = Path(__file__).resolve().parents[1] / "temp" / "pytest_tmp_paths"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    path = root / f"{safe_name}_{uuid.uuid4().hex[:8]}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
