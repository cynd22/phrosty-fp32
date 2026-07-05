"""Tier 0 — tests for the validation harness itself (validation/compare.py).

The standing rule delegates PASS/FAIL judgment to compare.py, which makes it
load-bearing: a bug there could wave a broken core through validation.  These
tests feed it synthetic stock/fork pairs engineered to pass or trip each
criterion and assert the exit code (0 = PASS, 1 = FAIL, per its contract).

Run via subprocess so the tests exercise the real CLI surface (argument
parsing included), exactly as the standing-rule procedure invokes it.
"""
import os
import pathlib
import subprocess
import sys

import numpy as np

_COMPARE = pathlib.Path(__file__).parent.parent.parent / "validation" / "compare.py"
_SKY = 10.0
_N = 512
_CORE_HALF = 50   # small frames -> shrink the excluded core box accordingly


def _run_compare(tmp_path, stock, fork):
    sp = tmp_path / "stock.npy"
    fp = tmp_path / "fork.npy"
    np.save(sp, stock)
    np.save(fp, fork)
    return subprocess.run(
        [sys.executable, str(_COMPARE), "--stock", str(sp), "--fork", str(fp),
         "--sky-rms", str(_SKY), "--core-half", str(_CORE_HALF)],
        capture_output=True, text=True, timeout=120)


def _stock_field(seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.5 * _SKY, (_N, _N))


def test_identical_fork_passes(tmp_path):
    stock = _stock_field()
    r = _run_compare(tmp_path, stock, stock.copy())
    assert r.returncode == 0, r.stdout + r.stderr


def test_small_white_perturbation_passes(tmp_path):
    # ~0.1% of sky, white: comfortably inside all three tolerances.
    stock = _stock_field()
    rng = np.random.default_rng(1)
    fork = stock + rng.normal(0.0, 1e-3 * _SKY, stock.shape)
    r = _run_compare(tmp_path, stock, fork)
    assert r.returncode == 0, r.stdout + r.stderr


def test_bgrms_inflation_fails(tmp_path):
    # criterion (1): 20% background-rms inflation >> 5% tolerance.
    stock = _stock_field()
    r = _run_compare(tmp_path, stock, stock * 1.2)
    assert r.returncode == 1, r.stdout + r.stderr


def test_structured_artifact_fails(tmp_path):
    # criteria (2)+(3): a spatially-correlated additive artifact at ~50% of
    # sky — the signature of the historical c64 break (structured background,
    # ACF divergence, large pairwise residual).
    stock = _stock_field()
    n = np.arange(_N)
    stripes = 0.5 * _SKY * np.sin(2 * np.pi * n / 16.0)
    fork = stock + stripes[None, :]
    r = _run_compare(tmp_path, stock, fork)
    assert r.returncode == 1, r.stdout + r.stderr


def test_shape_mismatch_fails(tmp_path):
    stock = _stock_field()
    r = _run_compare(tmp_path, stock, stock[: _N // 2, :].copy())
    assert r.returncode == 1, r.stdout + r.stderr
