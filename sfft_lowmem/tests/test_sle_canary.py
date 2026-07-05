"""Tier 0 — CANARY tests for the SkyLevel_Estimator failure modes (sfft #40).

These are NOT tests of the fork: the fork faithfully inherits stock sfft's
``create_score_image``, which divides by ``SkyLevel_Estimator.SLE(...)[1]``
with no check of the return value.  On low-overlap fields (a dominant
near-constant plateau), SLE either returns its -1.0 failure sentinel (exactly
constant plateau) or silently mode-collapses onto the plateau's numerical
noise (noisy plateau) — and the caller divides by it either way.  Documented
end-to-end in https://github.com/Roman-Supernova-PIT/sfft/issues/40.

These tests ASSERT THE CURRENT (BROKEN) BEHAVIOR, so that:
  * the failure mode is pinned as an executable characterization, not prose;
  * the moment a guard lands (upstream PR #38 territory, or a fork-side guard
    in create_score_image), these tests FAIL — the signal to invert them into
    guard tests and drop the canary.

Pure CPU (SLE is numpy code); no photometry claims involved — this is an
unchecked-error-return characterization, nothing more.
"""
import numpy as np
import pytest

from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator

_N = 600
_SIG = 13.3   # sky-ish scale, matches the issue's demonstration


def _healthy_field():
    rng = np.random.default_rng(3)
    return rng.normal(50.0, _SIG, (_N, _N))


def _plateau_field(plateau_frac=0.85, plateau_noise=0.0):
    """Field dominated by a near-constant plateau (the no-overlap footprint
    signature): ``plateau_frac`` of rows are the plateau, the rest real noise."""
    rng = np.random.default_rng(4)
    a = rng.normal(50.0, _SIG, (_N, _N))
    cut = int(plateau_frac * _N)
    if plateau_noise > 0.0:
        a[:cut, :] = rng.normal(0.0, plateau_noise, (cut, _N))
    else:
        a[:cut, :] = 0.0
    return a


def test_sle_sane_on_healthy_field():
    # Baseline: on a normal noise field SLE returns a usable sigma.
    sig = SkyLevel_Estimator.SLE(PixA_obj=_healthy_field())[1]
    assert 0.8 * _SIG < sig < 1.2 * _SIG


def test_sle_canary_constant_plateau_returns_unusable_sigma():
    # CANARY (issue #40, "starving" branch): exactly-constant dominant
    # plateau.  SLE does not raise; it hands back a sigma no caller could
    # divide by safely (the -1.0 sentinel, or a collapsed near-zero value).
    # If this test ever FAILS, a guard was added somewhere — replace the
    # canaries with real guard tests.
    sig = SkyLevel_Estimator.SLE(PixA_obj=_plateau_field(plateau_noise=0.0))[1]
    assert sig <= 0.0 or sig < 1e-6 * _SIG


def test_sle_canary_noisy_plateau_mode_collapses():
    # CANARY (issue #40, mode-collapse branch): plateau carrying ~1e-9-scale
    # numerical residue.  SLE converges ONTO THE PLATEAU and returns its
    # noise as the field's sky sigma — orders of magnitude below the real
    # noise scale, with no error of any kind.
    sig = SkyLevel_Estimator.SLE(PixA_obj=_plateau_field(plateau_noise=1e-9))[1]
    assert sig == pytest.approx(0.0, abs=1e-3 * _SIG)
    # ...which is exactly what create_score_image would then divide by.
