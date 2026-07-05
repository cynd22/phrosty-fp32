"""Tier 1 — GPU mechanism tests at small N (seconds, not the standing rule).

Each test encodes a specific claim the fork's comments/docs make, so a future
refactor cannot silently break the mechanism the claim rests on:

* the chunked FillLS_OMG is *bit-exact* vs the stock kernel (the LEVER-2 claim
  in sfft_lowmem_core_rfft.py);
* the row-banded half-Kab subtract is invariant to NBAND (banding is a memory
  layout choice, not an arithmetic one);
* the rfft half-spectrum post-processing identity holds to ~1e-13 (the LEVER-1
  claim in sfft_lowmem_rfft.py);
* the c128 fork core reproduces the stock float64 core's Solution and raw
  difference on identical input (small-N smoke of the headline equivalence).

WARNING — these do NOT replace the STANDING RULE.  Small-N tests are exactly
what missed the c64 conditioning bug (it only manifests at >= 2560^2 on real
data; see docs/SFFT_FORK_VALIDATION.md).  These catch *wiring* regressions
fast; any change to the core numerics still requires the full stock-vs-fork
run at >= 2560^2 via validation/.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_N = 512
_SKY = 10.0
_GKERHW = 4
_KERPOLY = 2

_cache = {}


def _gm(nt, maxthread):
    return ((nt - 1) // maxthread + 1, min(nt, maxthread))


def _synthetic_pair():
    """Well-conditioned sky-subtracted pair: same truth scene seen through two
    slightly different PSFs, independent noise.  (Synthetic is fine here: only
    Solution and the RAW difference are compared — no decorrelation, which is
    the stage that misbehaves on idealized scenes.)"""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(20172782)
    truth = np.zeros((_N, _N))
    nsrc = 200
    ys = rng.integers(20, _N - 20, nsrc)
    xs = rng.integers(20, _N - 20, nsrc)
    flux = rng.lognormal(mean=8.0, sigma=1.0, size=nsrc)
    np.add.at(truth, (ys, xs), flux)
    ref = gaussian_filter(truth, 1.4) + rng.normal(0., _SKY, truth.shape)
    sci = 1.03 * gaussian_filter(truth, 1.9) + rng.normal(0., _SKY, truth.shape)
    return ref, sci


def _setup():
    if _cache:
        return _cache
    import cupy as cp
    ref, sci = _synthetic_pair()
    _cache["REF"] = cp.ascontiguousarray(cp.array(ref, dtype=cp.float64))
    _cache["SCI"] = cp.ascontiguousarray(cp.array(sci, dtype=cp.float64))
    return _cache


# ---------------------------------------------------------------------------
# LEVER 1 claim: half-spectrum post-processing is exact for Hermitian
# multipliers (transforms of real kernels), at even N.
# ---------------------------------------------------------------------------

def test_rfft_postproc_hermitian_identity():
    import cupy as cp
    rng = np.random.default_rng(7)
    x = cp.array(rng.normal(0., 1., (_N, _N)))
    kern = cp.array(rng.normal(0., 1., (_N, _N)))   # real -> Hermitian transform
    mult = cp.fft.fft2(kern)
    ref = cp.fft.ifft2(cp.fft.fft2(x) * mult).real
    got = cp.fft.irfft2(cp.fft.rfft2(x) * mult[:, : _N // 2 + 1], s=(_N, _N))
    scale = float(cp.max(cp.abs(ref)))
    assert float(cp.max(cp.abs(got - ref))) < 1e-12 * scale


# ---------------------------------------------------------------------------
# LEVER 2 claim: chunked FillLS_OMG is bit-exact vs the stock kernel.
# ---------------------------------------------------------------------------

def test_chunked_omg_bit_exact_vs_stock_kernel():
    import cupy as cp
    from sfft_lowmem_core import SingleSFFTConfigure_Cupy_F32
    from sfft_lowmem_core_rfft import _OMG_CHUNK_MOD

    n = 128
    cfg = SingleSFFTConfigure_Cupy_F32(NX=n, NY=n, KerHW=3, KerPolyOrder=_KERPOLY,
                                       BGPolyOrder=2, ConstPhotRatio=True,
                                       VERBOSE_LEVEL=0)
    params, modules = cfg
    N0, N1 = params["N0"], params["N1"]
    w0, w1, L0, L1 = params["w0"], params["w1"], params["L0"], params["L1"]
    Fij, Fab = params["Fij"], params["Fab"]
    Fijab, NEQ, FOMG = params["Fijab"], params["NEQ"], params["FOMG"]
    SCALE, maxthread = params["SCALE"], params["MaxThreadPerB"]

    SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)],
                           dtype=np.int32)
    SREF_ijab_GPU = cp.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)],
                             dtype=cp.int32)
    REF_ab_GPU = cp.array([(a - w0, b - w1) for a in range(L0) for b in range(L1)],
                          dtype=cp.int32)

    rng = np.random.default_rng(11)
    fi = rng.normal(size=(Fij, N0, N1)) + 1j * rng.normal(size=(Fij, N0, N1))
    SPixA_FIij_GPU = cp.array(fi, dtype=cp.complex128)

    PreOMG = cp.empty((FOMG, N0, N1), dtype=cp.float64)
    for k in range(FOMG):
        a, b = int(SREF_iji0j0[k, 0]), int(SREF_iji0j0[k, 1])
        plane = cp.fft.fft2(SPixA_FIij_GPU[a] * cp.conj(SPixA_FIij_GPU[b])) * SCALE
        PreOMG[k] = plane.real * SCALE

    bpg, tpb = _gm(Fijab, maxthread)
    grid = (bpg, bpg)
    block = (tpb, tpb, 1)

    LH_ref = cp.zeros((NEQ, NEQ), dtype=cp.float64)
    modules["FillLS_OMG"].get_function("kmain")(
        args=(SREF_ijab_GPU, REF_ab_GPU, PreOMG, LH_ref), block=block, grid=grid)

    LH_chunk = cp.zeros((NEQ, NEQ), dtype=cp.float64)
    kchunk = _OMG_CHUNK_MOD.get_function("kchunk")
    for i8j8 in range(Fij):
        PreBlk = cp.ascontiguousarray(PreOMG[i8j8 * Fij:(i8j8 + 1) * Fij])
        kchunk(args=(SREF_ijab_GPU, REF_ab_GPU, PreBlk, LH_chunk,
                     np.int32(Fijab), np.int32(Fij), np.int32(N0), np.int32(N1),
                     np.int32(i8j8), np.int32(NEQ)),
               block=block, grid=grid)

    # The claim is BIT-exact, so assert equality, not closeness.
    assert bool(cp.array_equal(LH_ref, LH_chunk))


# ---------------------------------------------------------------------------
# Banding claim: NBAND splits rows, never arithmetic -> bitwise-identical DIFF.
# ---------------------------------------------------------------------------

def test_fdiff_banding_invariance():
    import cupy as cp
    from sfft_lowmem_rfft import PureCupy_Customized_Packet_F32_rfft

    c = _setup()
    diffs = {}
    old = os.environ.get("SFFT_FDIFF_NBAND")
    try:
        for nband in ("1", "4"):
            os.environ["SFFT_FDIFF_NBAND"] = nband
            _, diff = PureCupy_Customized_Packet_F32_rfft.PCCP(
                c["REF"], c["SCI"], c["REF"].copy(), c["SCI"].copy(),
                ForceConv="REF", GKerHW=_GKERHW, KerPolyOrder=_KERPOLY)
            diffs[nband] = diff
    finally:
        if old is None:
            os.environ.pop("SFFT_FDIFF_NBAND", None)
        else:
            os.environ["SFFT_FDIFF_NBAND"] = old
    assert bool(cp.array_equal(diffs["1"], diffs["4"]))


# ---------------------------------------------------------------------------
# Headline equivalence, small-N smoke: c128 fork core vs stock float64 core on
# identical input — Solution and RAW difference (no decorrelation stage).
# ---------------------------------------------------------------------------

def test_fork_matches_stock_solution_and_rawdiff():
    import cupy as cp
    from sfft.PureCupyCustomizedPacket import PureCupy_Customized_Packet
    from sfft_lowmem_rfft import PureCupy_Customized_Packet_F32_rfft

    c = _setup()
    kwargs = dict(ForceConv="REF", GKerHW=_GKERHW, KerPolyOrder=_KERPOLY,
                  BGPolyOrder=2, ConstPhotRatio=True)

    out_stock = PureCupy_Customized_Packet.PCCP(
        c["REF"], c["SCI"], c["REF"].copy(), c["SCI"].copy(), **kwargs)
    sol_stock, diff_stock = out_stock[0], out_stock[1]

    out_fork = PureCupy_Customized_Packet_F32_rfft.PCCP(
        c["REF"], c["SCI"], c["REF"].copy(), c["SCI"].copy(), **kwargs)
    sol_fork, diff_fork = out_fork[0], out_fork[1]

    # Solution: <= 5e-4 of scale (validation doc: <= 1e-4 at all even sizes
    # tested on real data; 5x margin for the synthetic scene).
    sol_scale = float(cp.max(cp.abs(sol_stock)))
    sol_err = float(cp.max(cp.abs(sol_fork.astype(cp.float64) - sol_stock)))
    assert sol_err < 5e-4 * sol_scale

    # Raw difference: background-level agreement in units of sky.
    d = cp.asnumpy(diff_fork).astype(np.float64) - cp.asnumpy(diff_stock)
    pair_rms = float(np.sqrt(np.mean(d ** 2)))
    assert pair_rms < 0.01 * _SKY

    # And neither run produced a degenerate difference.
    s_stock = float(cp.asnumpy(diff_stock).std())
    s_fork = float(cp.asnumpy(diff_fork).astype(np.float64).std())
    assert s_stock > 0.
    assert abs(s_fork - s_stock) / s_stock < 0.02
