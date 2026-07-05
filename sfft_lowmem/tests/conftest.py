"""Shared setup for the fork's test tiers.

Puts the fork engine (``sfft_lowmem/``), the validation harness
(``validation/``) and the repo root (for ``phrosty.pipeline``) on sys.path, and
registers the ``gpu`` marker.  GPU tests are skipped when ``SKIP_GPU_TESTS`` is
set (same convention as upstream phrosty) or when no CUDA device is visible.
"""
import os
import pathlib
import sys

import numpy as np
import pytest

_TESTS_DIR = pathlib.Path(__file__).parent.resolve()
_FORK_DIR = _TESTS_DIR.parent            # sfft_lowmem/
_REPO_ROOT = _FORK_DIR.parent
_VALIDATION_DIR = _REPO_ROOT / "validation"

for _p in (str(_FORK_DIR), str(_VALIDATION_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Upstream sfft still references the removed numpy alias np.in1d; restore it
# (same shim as validation/run_backend.py).
if not hasattr(np, "in1d"):
    np.in1d = np.isin


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: needs a CUDA device (skipped when SKIP_GPU_TESTS is set or no device is found)")


def _gpu_available():
    if os.getenv("SKIP_GPU_TESTS", ""):
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device visible / SKIP_GPU_TESTS set")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
