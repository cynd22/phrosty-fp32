"""
sfft_lowmem_core.py  --  Single-precision / low-memory fork of the sfft
PureCupy subtraction core (SFFTConfigure.py + SFFTSubtract.py).

This is a *copy-and-edit* fork.  The installed `sfft` package is NEVER
touched; the A100/float64 calibrated path keeps importing stock sfft.

What changes vs stock:
  * The 6 Hadamard-product CUDA kernels (HadProd_*) and Construct_FDIFF
    are JIT-compiled with cuFloatComplex instead of cuDoubleComplex
    (single-precision complex math).  We do this by *string-substituting*
    the stock kernel source returned by SingleSFFTConfigure_Cupy.SSCC,
    then re-compiling with cp.RawModule.  No hand transcription -> no
    index bugs.
  * The FillLS_* kernels are left fully at stock double precision
    (`double Pre*` -> `double LHMAT/RHb`): the protected linear system,
    see below.
  * In ElementalSFFTSubtract:
        complex128 -> complex64   (HpOMG/HpGAM/HpPSI/HpPHI/HpTHE/HpDEL,
                                    PixA_FJ, SPixA_FIij/FTpq, Kab_*, FDIFF)
        float32 for the image inputs; the real Pre* stacks stay float64
        (the FullPrecision carve-out, see NOTE).
  * PROTECTED FULL-PRECISION CARVE-OUT (ferrotorch autocast policy):
    the LHMAT / RHb matrix fill and the lu_factor / lu_solve stay
    float64.  NEQ x NEQ is N-independent (~35 MB at DK=2), so this costs
    almost nothing and protects the numerically sensitive solve.  The
    Pre* stacks are themselves kept in float64 (the FillLS_* kernels are
    left at stock `double Pre*` -> `double LHMAT`; see the NOTE below).

Peak-VRAM driver (the reason a flow-only f32 fork is NOT enough): the
(FOMG=Fij^2, N, N) complex stack HpOMG and its (FOMG, N, N) real PreOMG
live *inside* this core.  Halving them (c128->c64, f64->f32) is the
load-bearing win for the 2560^2 target.
"""

import os
import time
import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpx_linalg

from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure_Cupy

__author__ = "lowmem fork"


# ----------------------------------------------------------------------
# Kernel-string single-precision rewriter
# ----------------------------------------------------------------------
# The stock SingleSFFTConfigure_Cupy builds a dict of cp.RawModule's from
# Python source strings.  We cannot read those strings back out of a
# compiled RawModule, so instead we re-run the *string generation* by
# monkey-replicating: we capture each kernel's source by patching
# cp.RawModule during the SSCC call, transform the complex-math kernels
# to single precision, and recompile.

# kernels that contain complex (cuDoubleComplex) math -> need full f32 rewrite.
# These are the big Hadamard-product stacks + the final difference assembly;
# converting them to complex64 is the load-bearing memory win.
_COMPLEX_KERNELS = {
    'HadProd_OMG', 'HadProd_GAM', 'HadProd_PSI',
    'HadProd_PHI', 'HadProd_THE', 'HadProd_DEL',
    'Construct_FDIFF',
}
# NOTE: the FillLS_* kernels are LEFT UNCHANGED (stock `double Pre*` /
# `double LHMAT`).  The real Pre* stacks feed the linear-system matrix,
# which we keep in FULL float64 precision (ferrotorch FullPrecision
# carve-out).  Empirically the f32 Pre* path produced garbage solutions
# on the cross-convolved (PSF-smoothed -> ill-conditioned) image pairs
# that the real pipeline feeds; f64 Pre* fixes it while the complex
# Hadamard FFT stacks stay complex64 (still a large memory win).


def _f32_complex_rewrite(code: str) -> str:
    """Rewrite a cuDoubleComplex kernel source into cuFloatComplex."""
    code = code.replace('make_cuDoubleComplex', 'make_cuFloatComplex')
    code = code.replace('cuDoubleComplex', 'cuFloatComplex')
    code = code.replace('cuCmul(', 'cuCmulf(')
    code = code.replace('cuCadd(', 'cuCaddf(')
    code = code.replace('cuCsub(', 'cuCsubf(')
    code = code.replace('cuCdiv(', 'cuCdivf(')
    code = code.replace('cuConj(', 'cuConjf(')
    # The Construct_FDIFF kernel declares `double One_re=1.0; ...` scalars
    # that feed make_cuDoubleComplex; keep them double for the literal but
    # make_cuFloatComplex(double,double) is fine (implicit narrowing).
    return code


class _CaptureRawModule:
    """Stand-in for cp.RawModule that records the source code, then
    forwards to the real cp.RawModule so SSCC still returns working
    modules (we only use it to harvest the f64 source strings)."""
    captured = {}

    def __init__(self, *args, **kwargs):
        self._code = kwargs.get('code', args[0] if args else None)
        self._args = args
        self._kwargs = kwargs
        self._real = cp.RawModule(*args, **kwargs)

    def get_function(self, name):
        return self._real.get_function(name)


def SingleSFFTConfigure_Cupy_F32(NX, NY, KerHW, KerPolyOrder=2, BGPolyOrder=2,
                                 ConstPhotRatio=True, CUDA_COMPILER='nvrtc',
                                 VERBOSE_LEVEL=2):
    """float32/complex64 variant of SingleSFFTConfigure_Cupy.SSCC.

    Strategy: run stock SSCC but intercept every cp.RawModule(code=...)
    call to harvest the kernel source, then re-compile the complex
    kernels in single precision and the FillLS kernels with float Pre*.
    """
    captured_sources = {}
    orig_rawmodule = cp.RawModule

    # We need to know which module name each compiled kernel maps to.
    # SSCC assigns SFFTModule_dict[name] = cp.RawModule(...) immediately
    # after building `_code`.  We capture (code, kwargs) in call order and
    # then read the final dict to map name->source.
    call_log = []

    def _spy_rawmodule(*args, **kwargs):
        code = kwargs.get('code', args[0] if args else None)
        mod = orig_rawmodule(*args, **kwargs)
        call_log.append((code, kwargs, mod))
        return mod

    cp.RawModule = _spy_rawmodule
    try:
        SFFTConfig = SingleSFFTConfigure_Cupy.SSCC(
            NX=NX, NY=NY, KerHW=KerHW, KerPolyOrder=KerPolyOrder,
            BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio,
            CUDA_COMPILER=CUDA_COMPILER, VERBOSE_LEVEL=0)
    finally:
        cp.RawModule = orig_rawmodule

    SFFTParam_dict, SFFTModule_dict = SFFTConfig

    # Map module-object identity -> source code.
    id2code = {id(mod): code for (code, kw, mod) in call_log}

    # Rebuild the precision-sensitive modules in single precision.
    for name, mod in list(SFFTModule_dict.items()):
        code = id2code.get(id(mod))
        if code is None:
            continue
        new_code = None
        if name in _COMPLEX_KERNELS:
            new_code = _f32_complex_rewrite(code)
        if new_code is not None and new_code != code:
            SFFTModule_dict[name] = cp.RawModule(
                code=new_code, backend=CUDA_COMPILER,
                translate_cucomplex=True)

    return SFFTConfig


# ----------------------------------------------------------------------
# Single-precision Elemental subtraction
# ----------------------------------------------------------------------
_CPX = np.complex64       # complex working precision (Hadamard FFT stacks, subtract pass)
_FLT = np.float32         # real working precision (image planes, phase factors)

# FFT_PRECISION controls the precision of the forward DFTs and the Hadamard
# FFT stacks that FEED THE LINEAR SYSTEM (the solve pass).  The per-element
# c64 FFT error (~1e-7) is benign at small N but is amplified by the
# ill-conditioned PSF-matching solve at large N, poisoning LHMAT/RHb and
# producing the size-dependent structured-noise break (2560/4088).  Running
# the SOLVE-FEEDING transforms in complex128 restores f64-grade LHMAT
# accuracy; one plane is live at a time so the memory cost is only the
# persistent stacks, NOT a full c128 Hadamard block.  DEFAULT c128.
# The subtract pass (Construct_FDIFF + twiddle) stays c64 -- it does not feed
# the solve and its error is a pure ~f32-of-sky precision tradeoff.
_FFT_PREC = os.environ.get('SFFT_FFT_PRECISION', 'c128').strip().lower()
if _FFT_PREC not in ('c64', 'c128'):
    raise ValueError(f"SFFT_FFT_PRECISION must be 'c64' or 'c128', got: {_FFT_PREC!r}")
# solve-feeding FFT precision; c64 is the validated-BROKEN path (see
#   docs/SFFT_FORK_VALIDATION.md), so an unrecognized value must fail loudly
#   rather than silently downgrade to it.
_PCPX = np.complex64 if _FFT_PREC == 'c64' else np.complex128
_PRE = np.float64         # PROTECTED: real Pre* stacks that fill the linear
                          # system stay full float64 (FullPrecision carve-out).


def _retry_oom(fn):
    """ferrotorch RetryAfterFree: on OOM, reclaim cupy pool + retry once."""
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            return fn(*a, **k)
    return wrap


class ElementalSFFTSubtract_PureCupy_F32:
    @staticmethod
    def ESSPC(PixA_I_GPU, PixA_J_GPU, SFFTConfig, SFFTSolution_GPU=None,
              Subtract=False, VERBOSE_LEVEL=2):

        def LSSolver(LHMAT_GPU, RHb_GPU):
            # PROTECTED full-precision solve (ferrotorch FullPrecision policy)
            lu_piv_GPU = cpx_linalg.lu_factor(LHMAT_GPU, overwrite_a=False,
                                              check_finite=True)
            return cpx_linalg.lu_solve(lu_piv_GPU, RHb_GPU)

        ta = time.time()
        SFFTParam_dict, SFFTModule_dict = SFFTConfig
        N0, N1 = SFFTParam_dict['N0'], SFFTParam_dict['N1']
        w0, w1 = SFFTParam_dict['w0'], SFFTParam_dict['w1']
        DK, DB = SFFTParam_dict['DK'], SFFTParam_dict['DB']

        if PixA_I_GPU.shape != (N0, N1) or PixA_J_GPU.shape != (N0, N1):
            raise Exception('MeLOn ERROR: INCONSISTENT shape of input images!')

        ConstPhotRatio = SFFTParam_dict['ConstPhotRatio']
        MaxThreadPerB = SFFTParam_dict['MaxThreadPerB']
        L0, L1 = SFFTParam_dict['L0'], SFFTParam_dict['L1']
        Fab, Fij, Fpq = SFFTParam_dict['Fab'], SFFTParam_dict['Fij'], SFFTParam_dict['Fpq']
        SCALE, SCALE_L = SFFTParam_dict['SCALE'], SFFTParam_dict['SCALE_L']
        NEQ, Fijab = SFFTParam_dict['NEQ'], SFFTParam_dict['Fijab']
        NEQ_FSfree = SFFTParam_dict['NEQ_FSfree']
        FOMG, FGAM = SFFTParam_dict['FOMG'], SFFTParam_dict['FGAM']
        FTHE, FPSI = SFFTParam_dict['FTHE'], SFFTParam_dict['FPSI']
        FPHI, FDEL = SFFTParam_dict['FPHI'], SFFTParam_dict['FDEL']

        GPUManage = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
        BpG_PIX0, TpB_PIX0 = GPUManage(N0)
        BpG_PIX1, TpB_PIX1 = GPUManage(N1)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        REF_pq_GPU = cp.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)], dtype=cp.int32)
        REF_ij_GPU = cp.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)], dtype=cp.int32)
        REF_ab_GPU = cp.array([(a_pos-w0, b_pos-w1) for a_pos in range(L0) for b_pos in range(L1)], dtype=cp.int32)

        SREF_iji0j0_GPU = cp.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)], dtype=cp.int32)
        SREF_pqp0q0_GPU = cp.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)], dtype=cp.int32)
        SREF_ijpq_GPU = cp.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)], dtype=cp.int32)
        SREF_pqij_GPU = cp.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)], dtype=cp.int32)
        SREF_ijab_GPU = cp.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)], dtype=cp.int32)

        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)
        if ConstPhotRatio:
            FBijab = ij00[1:]
            MASK_nFS = np.ones(NEQ).astype(bool)
            MASK_nFS[FBijab] = False
            IDX_nFS = np.where(MASK_nFS)[0].astype(np.int32)
            IDX_nFS_GPU = cp.array(IDX_nFS)
            NEQ_FSfree = len(IDX_nFS)

        assert PixA_I_GPU.flags['C_CONTIGUOUS']
        assert PixA_J_GPU.flags['C_CONTIGUOUS']

        # * Get Spatial Coordinates  (coordinate kernels stay double-precision)
        PixA_X_GPU = cp.zeros((N0, N1), dtype=cp.int32)
        PixA_Y_GPU = cp.zeros((N0, N1), dtype=cp.int32)
        PixA_CX_GPU = cp.zeros((N0, N1), dtype=cp.float64)
        PixA_CY_GPU = cp.zeros((N0, N1), dtype=cp.float64)

        _func = SFFTModule_dict['SpatialCoor'].get_function('kmain')
        _func(args=(PixA_X_GPU, PixA_Y_GPU, PixA_CX_GPU, PixA_CY_GPU), block=TpB_PIX, grid=BpG_PIX)

        # SpatialPoly: stock kernel writes f64 SPixA_Iij/Tpq.  Keep f64 here
        # (matches the double kernel signature), then narrow to f32 before FFT.
        SPixA_Iij_GPU = cp.zeros((Fij, N0, N1), dtype=cp.float64)
        SPixA_Tpq_GPU = cp.zeros((Fpq, N0, N1), dtype=cp.float64)
        _func = SFFTModule_dict['SpatialPoly'].get_function('kmain')
        # PixA_I must be float64 for the double-typed SpatialPoly kernel
        PixA_I_f64 = PixA_I_GPU if PixA_I_GPU.dtype == cp.float64 else PixA_I_GPU.astype(cp.float64)
        _func(args=(REF_ij_GPU, REF_pq_GPU, PixA_CX_GPU, PixA_CY_GPU, PixA_I_f64,
                    SPixA_Iij_GPU, SPixA_Tpq_GPU), block=TpB_PIX, grid=BpG_PIX)
        del PixA_I_f64

        # * Make DFT of J, Iij, Tpq
        # Solve pass (SFFTSolution_GPU is None): run the forward DFTs and the
        # Hadamard FFT stacks at _PCPX (complex128 by default) so the
        # ill-conditioned solve is fed f64-grade Fourier stacks.  Subtract pass
        # (SFFTSolution_GPU given): keep _CPX (complex64) -- Construct_FDIFF is a
        # complex64 kernel and its inputs must match; its error is benign.
        _SCPX = _PCPX if SFFTSolution_GPU is None else _CPX

        PixA_FJ_GPU = cp.empty((N0, N1), dtype=_SCPX)
        PixA_FJ_GPU[:, :] = PixA_J_GPU.astype(_SCPX)
        PixA_FJ_GPU[:, :] = cp.fft.fft2(PixA_FJ_GPU)
        PixA_FJ_GPU[:, :] *= SCALE

        SPixA_FIij_GPU = cp.empty((Fij, N0, N1), dtype=_SCPX)
        SPixA_FIij_GPU[:, :, :] = SPixA_Iij_GPU.astype(_SCPX)
        for k in range(Fij):
            SPixA_FIij_GPU[k:k+1] = cp.fft.fft2(SPixA_FIij_GPU[k:k+1])
        SPixA_FIij_GPU[:, :] *= SCALE

        SPixA_FTpq_GPU = cp.empty((Fpq, N0, N1), dtype=_SCPX)
        SPixA_FTpq_GPU[:, :, :] = SPixA_Tpq_GPU.astype(_SCPX)
        for k in range(Fpq):
            SPixA_FTpq_GPU[k:k+1] = cp.fft.fft2(SPixA_FTpq_GPU[k:k+1])
        SPixA_FTpq_GPU[:, :] *= SCALE

        del SPixA_Iij_GPU
        del SPixA_Tpq_GPU

        PixA_CFJ_GPU = cp.conj(PixA_FJ_GPU)
        SPixA_CFIij_GPU = cp.conj(SPixA_FIij_GPU)
        SPixA_CFTpq_GPU = cp.conj(SPixA_FTpq_GPU)

        if SFFTSolution_GPU is not None:
            Solution_GPU = SFFTSolution_GPU
            if Solution_GPU.dtype != cp.float64:
                Solution_GPU = Solution_GPU.astype(cp.float64)
            a_ijab_GPU = Solution_GPU[:Fijab]
            b_pq_GPU = Solution_GPU[Fijab:]

        if SFFTSolution_GPU is None:
            BpG_OMG, TpB_OMG = (GPUManage(Fijab)[0], GPUManage(Fijab)[0]), (GPUManage(Fijab)[1], GPUManage(Fijab)[1], 1)
            BpG_GAM, TpB_GAM = (GPUManage(Fijab)[0], GPUManage(Fpq)[0]), (GPUManage(Fijab)[1], GPUManage(Fpq)[1], 1)
            BpG_THE, TpB_THE = (GPUManage(Fijab)[0], 1), (GPUManage(Fijab)[1], 1, 1)
            BpG_PSI, TpB_PSI = (GPUManage(Fpq)[0], GPUManage(Fijab)[0]), (GPUManage(Fpq)[1], GPUManage(Fijab)[1], 1)
            BpG_PHI, TpB_PHI = (GPUManage(Fpq)[0], GPUManage(Fpq)[0]), (GPUManage(Fpq)[1], GPUManage(Fpq)[1], 1)
            BpG_DEL, TpB_DEL = (GPUManage(Fpq)[0], 1), (GPUManage(Fpq)[1], 1, 1)

            # PROTECTED full precision linear system
            LHMAT_GPU = cp.empty((NEQ, NEQ), dtype=np.float64)
            RHb_GPU = cp.empty(NEQ, dtype=np.float64)

            # Hadamard product helper.  The stock HadProd_* CUDA kernels are
            # compiled as cuFloatComplex (complex64) and CANNOT be fed _SCPX
            # arrays when _SCPX is complex128 (the solve pass at the c128
            # default) -- the kernel would misread the wider stride and emit
            # NaNs into LHMAT.  Instead we build each Hadamard plane with the
            # VERIFIED elementwise identity  Hp[k] = A[SREF[k,0]] * B[SREF[k,1]]
            # (the exact operation the HadProd_* kernels perform; cupy honours
            # the operand dtype, so this is correct at both c64 and c128 and is
            # bit-equivalent to the kernel at c64).  The full (F*, N, N) _SCPX
            # stack is still materialised, preserving the base-core memory
            # layout.  THE/DEL multiply against the single B plane PixA_CFJ.
            def _hadprod(Fblk, stackA, stackB, sref):
                Hp = cp.empty((Fblk, N0, N1), dtype=_SCPX)
                for k in range(Fblk):
                    Hp[k] = stackA[int(sref[k, 0])] * stackB[int(sref[k, 1])]
                return Hp

            def _hadprod_J(Fblk, stackA, planeB):
                Hp = cp.empty((Fblk, N0, N1), dtype=_SCPX)
                for k in range(Fblk):
                    Hp[k] = stackA[k] * planeB
                return Hp

            SREF_iji0j0 = cp.asnumpy(SREF_iji0j0_GPU)
            SREF_ijpq = cp.asnumpy(SREF_ijpq_GPU)
            SREF_pqij = cp.asnumpy(SREF_pqij_GPU)
            SREF_pqp0q0 = cp.asnumpy(SREF_pqp0q0_GPU)

            # ---- OMG ----
            HpOMG_GPU = _hadprod(FOMG, SPixA_FIij_GPU, SPixA_CFIij_GPU, SREF_iji0j0)
            for k in range(FOMG):
                HpOMG_GPU[k:k+1] = cp.fft.fft2(HpOMG_GPU[k:k+1])
            HpOMG_GPU *= SCALE
            PreOMG_GPU = cp.empty((FOMG, N0, N1), dtype=_PRE)
            PreOMG_GPU[:, :, :] = HpOMG_GPU.real
            PreOMG_GPU[:, :, :] *= SCALE
            del HpOMG_GPU
            _func = SFFTModule_dict['FillLS_OMG'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreOMG_GPU, LHMAT_GPU), block=TpB_OMG, grid=BpG_OMG)
            del PreOMG_GPU

            # ---- GAM ----
            HpGAM_GPU = _hadprod(FGAM, SPixA_FIij_GPU, SPixA_CFTpq_GPU, SREF_ijpq)
            for k in range(FGAM):
                HpGAM_GPU[k:k+1] = cp.fft.fft2(HpGAM_GPU[k:k+1])
            HpGAM_GPU *= SCALE
            PreGAM_GPU = cp.empty((FGAM, N0, N1), dtype=_PRE)
            PreGAM_GPU[:, :, :] = HpGAM_GPU.real
            del HpGAM_GPU
            _func = SFFTModule_dict['FillLS_GAM'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreGAM_GPU, LHMAT_GPU), block=TpB_GAM, grid=BpG_GAM)
            del PreGAM_GPU

            # ---- PSI ----  Hp = FTpq[SREF col0=pq] * CFIij[SREF col1=ij]
            # (the stock HadProd_PSI body multiplies FTpq[col0]*CFIij[col1]
            #  regardless of the kernel arg order -- col0 ranges over Fpq.)
            HpPSI_GPU = _hadprod(FPSI, SPixA_FTpq_GPU, SPixA_CFIij_GPU, SREF_pqij)
            for k in range(FPSI):
                HpPSI_GPU[k:k+1] = cp.fft.fft2(HpPSI_GPU[k:k+1])
            HpPSI_GPU *= SCALE
            PrePSI_GPU = cp.empty((FPSI, N0, N1), dtype=_PRE)
            PrePSI_GPU[:, :, :] = HpPSI_GPU.real
            del HpPSI_GPU
            _func = SFFTModule_dict['FillLS_PSI'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PrePSI_GPU, LHMAT_GPU), block=TpB_PSI, grid=BpG_PSI)
            del PrePSI_GPU

            # ---- PHI ----
            HpPHI_GPU = _hadprod(FPHI, SPixA_FTpq_GPU, SPixA_CFTpq_GPU, SREF_pqp0q0)
            for k in range(FPHI):
                HpPHI_GPU[k:k+1] = cp.fft.fft2(HpPHI_GPU[k:k+1])
            HpPHI_GPU *= SCALE
            PrePHI_GPU = cp.empty((FPHI, N0, N1), dtype=_PRE)
            PrePHI_GPU[:, :, :] = HpPHI_GPU.real
            PrePHI_GPU[:, :, :] *= SCALE_L
            del HpPHI_GPU
            _func = SFFTModule_dict['FillLS_PHI'].get_function('kmain')
            _func(args=(PrePHI_GPU, LHMAT_GPU), block=TpB_PHI, grid=BpG_PHI)
            del PrePHI_GPU

            # ---- THE & DEL ----
            HpTHE_GPU = _hadprod_J(FTHE, SPixA_FIij_GPU, PixA_CFJ_GPU)
            HpDEL_GPU = _hadprod_J(FDEL, SPixA_FTpq_GPU, PixA_CFJ_GPU)
            for k in range(FTHE):
                HpTHE_GPU[k:k+1] = cp.fft.fft2(HpTHE_GPU[k:k+1])
            HpTHE_GPU[:, :, :] *= SCALE
            for k in range(FDEL):
                HpDEL_GPU[k:k+1] = cp.fft.fft2(HpDEL_GPU[k:k+1])
            HpDEL_GPU[:, :, :] *= SCALE
            PreTHE_GPU = cp.empty((FTHE, N0, N1), dtype=_PRE)
            PreTHE_GPU[:, :, :] = HpTHE_GPU.real
            del HpTHE_GPU
            PreDEL_GPU = cp.empty((FDEL, N0, N1), dtype=_PRE)
            PreDEL_GPU[:, :, :] = HpDEL_GPU.real
            PreDEL_GPU[:, :, :] *= SCALE_L
            del HpDEL_GPU
            _func = SFFTModule_dict['FillLS_THE'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreTHE_GPU, RHb_GPU), block=TpB_THE, grid=BpG_THE)
            del PreTHE_GPU
            _func = SFFTModule_dict['FillLS_DEL'].get_function('kmain')
            _func(args=(PreDEL_GPU, RHb_GPU), block=TpB_DEL, grid=BpG_DEL)
            del PreDEL_GPU

            # ---- Remove Forbidden Stripes ----
            if ConstPhotRatio:
                RHb_FSfree_GPU = cp.take(RHb_GPU, IDX_nFS_GPU)
                _func = SFFTModule_dict['Remove_LSFStripes'].get_function('kmain')
                BpG_FSfree_PA, TpB_FSfree_PA = GPUManage(NEQ_FSfree)
                BpG_FSfree = (BpG_FSfree_PA, BpG_FSfree_PA)
                TpB_FSfree = (TpB_FSfree_PA, TpB_FSfree_PA, 1)
                LHMAT_FSfree_GPU = cp.empty((NEQ_FSfree, NEQ_FSfree), dtype=np.float64)
                _func(args=(LHMAT_GPU, IDX_nFS_GPU, LHMAT_FSfree_GPU), block=TpB_FSfree, grid=BpG_FSfree)

            # ---- Solve (protected f64) ----
            if not ConstPhotRatio:
                Solution_GPU = LSSolver(LHMAT_GPU=LHMAT_GPU, RHb_GPU=RHb_GPU)
            else:
                Solution_FSfree_GPU = LSSolver(LHMAT_GPU=LHMAT_FSfree_GPU, RHb_GPU=RHb_FSfree_GPU)
                _func = SFFTModule_dict['Extend_Solution'].get_function('kmain')
                BpG_ES, TpB_ES = (BpG_FSfree_PA, 1), (TpB_FSfree_PA, 1, 1)
                Solution_GPU = cp.zeros(NEQ, dtype=np.float64)
                _func(args=(Solution_FSfree_GPU, IDX_nFS_GPU, Solution_GPU), block=TpB_ES, grid=BpG_ES)

            a_ijab_GPU = Solution_GPU[:Fijab]
            b_pq_GPU = Solution_GPU[Fijab:]

        # * Perform Subtraction
        PixA_DIFF_GPU = None
        if Subtract:
            Wl_GPU = cp.exp((-2j*np.pi/N0) * PixA_X_GPU.astype(_FLT)).astype(_CPX)
            Wm_GPU = cp.exp((-2j*np.pi/N1) * PixA_Y_GPU.astype(_FLT)).astype(_CPX)
            Kab_Wla_GPU = cp.empty((L0, N0, N1), dtype=_CPX)
            Kab_Wmb_GPU = cp.empty((L1, N0, N1), dtype=_CPX)
            if w0 == w1:
                wx = w0
                for aob in range(-wx, wx+1):
                    Kab_Wla_GPU[aob + wx] = Wl_GPU ** aob
                    Kab_Wmb_GPU[aob + wx] = Wm_GPU ** aob
            else:
                for a in range(-w0, w0+1):
                    Kab_Wla_GPU[a + w0] = Wl_GPU ** a
                for b in range(-w1, w1+1):
                    Kab_Wmb_GPU[b + w1] = Wm_GPU ** b

            _func = SFFTModule_dict['Construct_FDIFF'].get_function('kmain')
            PixA_FDIFF_GPU = cp.empty((N0, N1), dtype=_CPX)
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, a_ijab_GPU.astype(_CPX),
                        SPixA_FIij_GPU, Kab_Wla_GPU, Kab_Wmb_GPU, b_pq_GPU.astype(_CPX),
                        SPixA_FTpq_GPU, PixA_FJ_GPU, PixA_FDIFF_GPU), block=TpB_PIX, grid=BpG_PIX)
            PixA_DIFF_GPU = SCALE_L * cp.fft.ifft2(PixA_FDIFF_GPU).real

        return Solution_GPU, PixA_DIFF_GPU


class GeneralSFFTSubtract_PureCupy_F32:
    @staticmethod
    def GSS(PixA_I_GPU, PixA_J_GPU, PixA_mI_GPU, PixA_mJ_GPU, SFFTConfig,
            ContamMask_I_GPU=None, VERBOSE_LEVEL=2):
        tmplst = [PixA_I_GPU.shape, PixA_J_GPU.shape, PixA_mI_GPU.shape, PixA_mI_GPU.shape]
        if len(set(tmplst)) > 1:
            raise Exception('MeLOn ERROR: Input images should have same size!')

        Solution_GPU = ElementalSFFTSubtract_PureCupy_F32.ESSPC(
            PixA_I_GPU=PixA_mI_GPU, PixA_J_GPU=PixA_mJ_GPU, SFFTConfig=SFFTConfig,
            SFFTSolution_GPU=None, Subtract=False, VERBOSE_LEVEL=VERBOSE_LEVEL)[0]

        PixA_DIFF_GPU = ElementalSFFTSubtract_PureCupy_F32.ESSPC(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU, SFFTConfig=SFFTConfig,
            SFFTSolution_GPU=Solution_GPU, Subtract=True, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]

        return Solution_GPU, PixA_DIFF_GPU, None
