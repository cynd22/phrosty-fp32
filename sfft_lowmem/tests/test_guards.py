"""Tier 0 — guard-rail tests.  No GPU work; run everywhere, on every change.

These exist because every guard here protects against a *silent* failure mode
that was found the hard way (see docs/SFFT_FORK_VALIDATION.md):

* ``KerPolyOrder >= 5`` overflows the fixed ``FIloc[16]`` buffer in the banded
  subtract kernel -> silent difference-image corruption, not a crash.
* An unrecognized ``SFFT_FFT_PRECISION`` must NOT fall through to c64 — c64 on
  the solve-feeding DFTs is the validated-broken path that produced structured
  garbage backgrounds at >= 2560^2.
* ``SFFT_BACKEND`` routing decides whether the user gets the fork at all; a
  typo must not silently change which engine runs (stock f64 is the default,
  and stays the default — the fork is strictly opt-in).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

_FORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# PCCP argument guards (fire before any GPU call, so numpy arrays suffice)
# ---------------------------------------------------------------------------

def _dummy_images(n=64):
    a = np.zeros((n, n), dtype=np.float32)
    return a, a.copy(), a.copy(), a.copy()


def test_pccp_rejects_kerpolyorder_gt4():
    from sfft_lowmem_rfft import PureCupy_Customized_Packet_F32_rfft
    ref, sci, mref, msci = _dummy_images()
    with pytest.raises(AssertionError, match="FIloc"):
        PureCupy_Customized_Packet_F32_rfft.PCCP(
            ref, sci, mref, msci, ForceConv="REF", GKerHW=4, KerPolyOrder=5)


def test_pccp_rejects_gkerhw_gt31():
    from sfft_lowmem_rfft import PureCupy_Customized_Packet_F32_rfft
    ref, sci, mref, msci = _dummy_images()
    with pytest.raises(AssertionError, match="powl"):
        PureCupy_Customized_Packet_F32_rfft.PCCP(
            ref, sci, mref, msci, ForceConv="REF", GKerHW=32, KerPolyOrder=2)


def test_configure_rejects_fij_gt16():
    # The config-space guard (independent of PCCP): KerPolyOrder=5 -> Fij=21.
    from sfft_lowmem_core import SingleSFFTConfigure_Cupy_F32
    with pytest.raises(ValueError, match="Fij=21"):
        SingleSFFTConfigure_Cupy_F32(NX=256, NY=256, KerHW=3, KerPolyOrder=5,
                                     BGPolyOrder=2, ConstPhotRatio=True,
                                     VERBOSE_LEVEL=0)


# ---------------------------------------------------------------------------
# SFFT_FFT_PRECISION is read at import time in each core module; an
# unrecognized value must raise, never silently downgrade to c64.
# ---------------------------------------------------------------------------

_CORES = ["sfft_lowmem_core", "sfft_lowmem_core_4088", "sfft_lowmem_core_rfft"]


def _import_core_subprocess(module, precision):
    env = dict(os.environ)
    env["SFFT_FFT_PRECISION"] = precision
    env["PYTHONPATH"] = _FORK_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c",
         f"import numpy as np\n"
         f"np.in1d = getattr(np, 'in1d', np.isin)\n"
         f"import {module}"],
        env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("module", _CORES)
def test_bad_fft_precision_fails_loud(module):
    r = _import_core_subprocess(module, "c96")
    assert r.returncode != 0
    assert "SFFT_FFT_PRECISION" in r.stderr


@pytest.mark.parametrize("module", _CORES)
@pytest.mark.parametrize("precision", ["c128", "c64"])
def test_valid_fft_precision_imports(module, precision):
    # c64 must stay importable: it is the documented escape hatch used to
    # reproduce the broken path in controlled A/B runs.
    r = _import_core_subprocess(module, precision)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# SFFT_BACKEND routing in phrosty.pipeline (fork is opt-in; default is stock
# float64 — "they run fp64 as default" is load-bearing for upstream parity).
# ---------------------------------------------------------------------------

def _resolve(monkeypatch, value):
    import phrosty.pipeline as pl
    if value is None:
        monkeypatch.delenv("SFFT_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SFFT_BACKEND", value)
    return pl._resolve_sfft_backend(), pl

def test_backend_default_is_stock(monkeypatch):
    flow, pl = _resolve(monkeypatch, None)
    assert flow is pl.SpaceSFFT_CupyFlow


def test_backend_rfft_selected(monkeypatch):
    flow, _ = _resolve(monkeypatch, "rfft")
    assert flow.__name__ == "SpaceSFFT_CupyFlow_LowMem_rfft"


@pytest.mark.parametrize("value", [" rfft ", "rfft\n", "RFFT"])
def test_backend_tolerates_whitespace_and_case(monkeypatch, value):
    # Regression for the strip() fix: trailing whitespace used to silently
    # route to stock.
    flow, _ = _resolve(monkeypatch, value)
    assert flow.__name__ == "SpaceSFFT_CupyFlow_LowMem_rfft"


def test_backend_unknown_falls_back_to_stock(monkeypatch):
    # Documented (if debatable) behavior: unknown values -> stock f64.
    # If this ever gains a warning, extend this test to assert it.
    flow, pl = _resolve(monkeypatch, "bogus-backend")
    assert flow is pl.SpaceSFFT_CupyFlow
