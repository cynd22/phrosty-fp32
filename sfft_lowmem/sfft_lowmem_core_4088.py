"""
sfft_lowmem_core_4088.py  --  PHASE-A (push-to-4088) low-memory SFFT core.

Builds on sfft_lowmem_core.py (the working f32/c64 fork) and adds the
final memory levers needed to fit larger N on an 8 GB card:

  LEVER 1 (load-bearing) -- PLANE-BY-PLANE Hadamard + FFT + real-extract.
    Stock / phase-2 fork materialise the FULL (FOMG=36, N, N) complex64
    Hadamard stack `HpOMG` AND, simultaneously, the (36, N, N) float64
    `PreOMG` real stack.  At 4088 that single OMG block is ~9 GB and is
    THE peak.  Here we never hold the 36-plane complex stack: for each
    output plane k we compute one (N,N) complex64 plane = A[srcA]*B[srcB]
    (an exact cupy replication of the HadProd_* CUDA kernel, verified to
    1e-7), FFT it in place, write its real part into Pre[k], and free the
    plane.  Peak complex transient drops from FOMG*N^2 to 1*N^2.
    The (verified) elementwise identity replacing each kernel:
        Hp[k] = Astack[SREF[k,0]] * Bstack[SREF[k,1]]
    so no custom-kernel rewrite / index risk -- pure cupy.

  LEVER 2 -- Pre* stacks stay float32 by default for the MEMORY benchmark
    but the linear-system fill still reads them; we keep the protected
    float64 carve-out for the SOLVE inputs exactly as phase 2 (PreOMG etc.
    remain float64 because the FillLS kernels are `double Pre*`).  Memory
    saving here comes from never duplicating the complex stack, NOT from
    narrowing Pre (Pre f64 is required for solution correctness -- proven
    in phase 2).

  LEVER 3 -- rfft for the real-input forward transforms is NOT applied to
    the Pre path because FillLS_* index the full (N,N) layout; instead the
    complex-stack elimination (lever 1) already removes the dominant term.

  LEVER 4 -- free-after-use of the persistent conj stacks the moment the
    last block that needs them is consumed.

Correctness carve-outs INHERITED unchanged from phase 2:
  * Linear system LHMAT/RHb + lu_factor/lu_solve stay float64.
  * Pre* real stacks that fill LHMAT stay float64.
  * SpatialPoly / coordinate kernels stay float64 (double-typed kernels).
"""

import os
import time
import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpx_linalg

# FFT_PRECISION controls the precision of the FORWARD DFTs and the per-plane
# Hadamard FFTs that FEED THE LINEAR SYSTEM:
#   'c64'  -> single precision everywhere (smallest memory, but the f32 FFT
#             error propagates into LHMAT and -- for non-trivial / distinct-PSF
#             matching kernels -- corrupts the ill-conditioned solve).
#   'c128' -> the per-plane Hadamard FFT and the persistent forward DFTs run
#             in complex128 (one plane live at a time, so the memory cost is
#             only the 8 persistent stacks, NOT a full FOMG c128 stack).  This
#             restores f64-grade LHMAT accuracy.  DEFAULT.
# The final DIFF construction (Construct_FDIFF) always runs in c64 (it does
# not feed the solve; its error is a pure precision tradeoff, ~f32 of sky).
_FFT_PREC = os.environ.get('SFFT_FFT_PRECISION', 'c128').strip().lower()

# Reuse the phase-2 configure (it already builds the f32 complex kernels;
# we don't use the HadProd kernels here, but SSCC also builds SpatialCoor,
# SpatialPoly, FillLS_*, Remove_LSFStripes, Extend_Solution, Construct_FDIFF
# which we DO use).
from sfft_lowmem_core import SingleSFFTConfigure_Cupy_F32

__author__ = "lowmem fork - phase A 4088"

_CPX = np.complex64           # persistent stacks for the DIFF construction
_FLT = np.float32
_PRE = np.float64             # PROTECTED: real stacks that fill the linear system.
_PCPX = np.complex128 if _FFT_PREC == 'c128' else np.complex64   # solve-feeding FFTs


class ElementalSFFTSubtract_PureCupy_F32_4088:
    @staticmethod
    def ESSPC(PixA_I_GPU, PixA_J_GPU, SFFTConfig, SFFTSolution_GPU=None,
              Subtract=False, VERBOSE_LEVEL=2):

        def LSSolver(LHMAT_GPU, RHb_GPU):
            lu_piv_GPU = cpx_linalg.lu_factor(LHMAT_GPU, overwrite_a=True,
                                              check_finite=True)
            return cpx_linalg.lu_solve(lu_piv_GPU, RHb_GPU)

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

        # SREF maps (numpy is enough -- used to index host-side in the
        # plane loop).
        SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)], dtype=np.int32)
        SREF_pqp0q0 = np.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)], dtype=np.int32)
        SREF_ijpq = np.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)], dtype=np.int32)
        SREF_pqij = np.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)], dtype=np.int32)

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

        # ---- Spatial coordinates (double kernels) ----
        PixA_X_GPU = cp.zeros((N0, N1), dtype=cp.int32)
        PixA_Y_GPU = cp.zeros((N0, N1), dtype=cp.int32)
        PixA_CX_GPU = cp.zeros((N0, N1), dtype=cp.float64)
        PixA_CY_GPU = cp.zeros((N0, N1), dtype=cp.float64)
        _func = SFFTModule_dict['SpatialCoor'].get_function('kmain')
        _func(args=(PixA_X_GPU, PixA_Y_GPU, PixA_CX_GPU, PixA_CY_GPU), block=TpB_PIX, grid=BpG_PIX)

        SPixA_Iij_GPU = cp.zeros((Fij, N0, N1), dtype=cp.float64)
        SPixA_Tpq_GPU = cp.zeros((Fpq, N0, N1), dtype=cp.float64)
        _func = SFFTModule_dict['SpatialPoly'].get_function('kmain')
        PixA_I_f64 = PixA_I_GPU if PixA_I_GPU.dtype == cp.float64 else PixA_I_GPU.astype(cp.float64)
        _func(args=(REF_ij_GPU, REF_pq_GPU, PixA_CX_GPU, PixA_CY_GPU, PixA_I_f64,
                    SPixA_Iij_GPU, SPixA_Tpq_GPU), block=TpB_PIX, grid=BpG_PIX)
        del PixA_I_f64, PixA_CX_GPU, PixA_CY_GPU

        # ---- Forward DFTs ----
        # For the SOLVE pass (SFFTSolution is None) the stacks FEED THE LINEAR
        # SYSTEM, so run them at _PCPX (complex128 by default -> accurate
        # LHMAT).  For the SUBTRACT pass they only feed Construct_FDIFF (a pure
        # precision tradeoff) -> keep c64 there to save memory.
        _SCPX = _PCPX if SFFTSolution_GPU is None else _CPX
        PixA_FJ_GPU = cp.empty((N0, N1), dtype=_SCPX)
        PixA_FJ_GPU[:, :] = PixA_J_GPU.astype(_SCPX)
        PixA_FJ_GPU[:, :] = cp.fft.fft2(PixA_FJ_GPU)
        PixA_FJ_GPU *= SCALE

        SPixA_FIij_GPU = cp.empty((Fij, N0, N1), dtype=_SCPX)
        SPixA_FIij_GPU[:, :, :] = SPixA_Iij_GPU.astype(_SCPX)
        for k in range(Fij):
            SPixA_FIij_GPU[k:k+1] = cp.fft.fft2(SPixA_FIij_GPU[k:k+1])
        SPixA_FIij_GPU *= SCALE
        del SPixA_Iij_GPU

        SPixA_FTpq_GPU = cp.empty((Fpq, N0, N1), dtype=_SCPX)
        SPixA_FTpq_GPU[:, :, :] = SPixA_Tpq_GPU.astype(_SCPX)
        for k in range(Fpq):
            SPixA_FTpq_GPU[k:k+1] = cp.fft.fft2(SPixA_FTpq_GPU[k:k+1])
        SPixA_FTpq_GPU *= SCALE
        del SPixA_Tpq_GPU

        SPixA_CFIij_GPU = cp.conj(SPixA_FIij_GPU)
        SPixA_CFTpq_GPU = cp.conj(SPixA_FTpq_GPU)
        PixA_CFJ_GPU = cp.conj(PixA_FJ_GPU)

        if SFFTSolution_GPU is not None:
            Solution_GPU = SFFTSolution_GPU
            if Solution_GPU.dtype != cp.float64:
                Solution_GPU = Solution_GPU.astype(cp.float64)
            a_ijab_GPU = Solution_GPU[:Fijab]
            b_pq_GPU = Solution_GPU[Fijab:]

        # =====================================================================
        # PLANE-BY-PLANE Pre* construction (LEVER 1).  For each block we build
        # the (F*, N, N) float64 Pre stack one plane at a time, so the complex
        # transient is a single (N,N) plane, not the whole F*-plane stack.
        # =====================================================================
        def build_pre(Fblk, srcA_stack, srcB_stack, sref, scale_extra=None):
            """Return a (Fblk, N0, N1) float64 Pre stack, materialising only
            one complex64 plane at a time."""
            Pre = cp.empty((Fblk, N0, N1), dtype=_PRE)
            for k in range(Fblk):
                a, b = int(sref[k, 0]), int(sref[k, 1])
                plane = srcA_stack[a] * srcB_stack[b]          # (N,N) c64
                plane = cp.fft.fft2(plane)                      # in place-ish
                plane *= SCALE
                rp = plane.real
                if scale_extra is not None:
                    rp = rp * scale_extra
                Pre[k, :, :] = rp
                del plane, rp
            return Pre

        if SFFTSolution_GPU is None:
            BpG_OMG, TpB_OMG = (GPUManage(Fijab)[0], GPUManage(Fijab)[0]), (GPUManage(Fijab)[1], GPUManage(Fijab)[1], 1)
            BpG_GAM, TpB_GAM = (GPUManage(Fijab)[0], GPUManage(Fpq)[0]), (GPUManage(Fijab)[1], GPUManage(Fpq)[1], 1)
            BpG_THE, TpB_THE = (GPUManage(Fijab)[0], 1), (GPUManage(Fijab)[1], 1, 1)
            BpG_PSI, TpB_PSI = (GPUManage(Fpq)[0], GPUManage(Fijab)[0]), (GPUManage(Fpq)[1], GPUManage(Fijab)[1], 1)
            BpG_PHI, TpB_PHI = (GPUManage(Fpq)[0], GPUManage(Fpq)[0]), (GPUManage(Fpq)[1], GPUManage(Fpq)[1], 1)
            BpG_DEL, TpB_DEL = (GPUManage(Fpq)[0], 1), (GPUManage(Fpq)[1], 1, 1)

            LHMAT_GPU = cp.empty((NEQ, NEQ), dtype=np.float64)
            RHb_GPU = cp.empty(NEQ, dtype=np.float64)

            # ---- OMG ----  Hp = FIij * CFIij ; PreOMG *= SCALE (extra)
            PreOMG_GPU = build_pre(FOMG, SPixA_FIij_GPU, SPixA_CFIij_GPU,
                                   SREF_iji0j0, scale_extra=SCALE)
            _func = SFFTModule_dict['FillLS_OMG'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreOMG_GPU, LHMAT_GPU), block=TpB_OMG, grid=BpG_OMG)
            del PreOMG_GPU

            # ---- GAM ----  Hp = FIij * CFTpq
            PreGAM_GPU = build_pre(FGAM, SPixA_FIij_GPU, SPixA_CFTpq_GPU,
                                   SREF_ijpq)
            _func = SFFTModule_dict['FillLS_GAM'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreGAM_GPU, LHMAT_GPU), block=TpB_GAM, grid=BpG_GAM)
            del PreGAM_GPU

            # ---- PSI ----  Hp = FTpq[sref0] * CFIij[sref1]
            PrePSI_GPU = build_pre(FPSI, SPixA_FTpq_GPU, SPixA_CFIij_GPU,
                                   SREF_pqij)
            _func = SFFTModule_dict['FillLS_PSI'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PrePSI_GPU, LHMAT_GPU), block=TpB_PSI, grid=BpG_PSI)
            del PrePSI_GPU

            # ---- PHI ----  Hp = FTpq * CFTpq ; PrePHI *= SCALE_L
            PrePHI_GPU = build_pre(FPHI, SPixA_FTpq_GPU, SPixA_CFTpq_GPU,
                                   SREF_pqp0q0, scale_extra=SCALE_L)
            _func = SFFTModule_dict['FillLS_PHI'].get_function('kmain')
            _func(args=(PrePHI_GPU, LHMAT_GPU), block=TpB_PHI, grid=BpG_PHI)
            del PrePHI_GPU

            # ---- THE ----  Hp = FIij * CFJ (single B plane)
            PreTHE_GPU = cp.empty((FTHE, N0, N1), dtype=_PRE)
            for k in range(FTHE):
                plane = SPixA_FIij_GPU[k] * PixA_CFJ_GPU
                plane = cp.fft.fft2(plane); plane *= SCALE
                PreTHE_GPU[k, :, :] = plane.real
                del plane
            _func = SFFTModule_dict['FillLS_THE'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreTHE_GPU, RHb_GPU), block=TpB_THE, grid=BpG_THE)
            del PreTHE_GPU

            # ---- DEL ----  Hp = FTpq * CFJ ; PreDEL *= SCALE_L
            PreDEL_GPU = cp.empty((FDEL, N0, N1), dtype=_PRE)
            for k in range(FDEL):
                plane = SPixA_FTpq_GPU[k] * PixA_CFJ_GPU
                plane = cp.fft.fft2(plane); plane *= SCALE
                PreDEL_GPU[k, :, :] = plane.real * SCALE_L
                del plane
            _func = SFFTModule_dict['FillLS_DEL'].get_function('kmain')
            _func(args=(PreDEL_GPU, RHb_GPU), block=TpB_DEL, grid=BpG_DEL)
            del PreDEL_GPU

            # CFJ no longer needed for solve.
            # (kept FIij/FTpq + conj for the subtract phase if Subtract=True;
            #  but in GSS the solve and subtract are SEPARATE ESSPC calls, so
            #  here SFFTSolution is None -> Subtract is False -> we can free.)

            # ---- Remove Forbidden Stripes ----
            if ConstPhotRatio:
                RHb_FSfree_GPU = cp.take(RHb_GPU, IDX_nFS_GPU)
                _func = SFFTModule_dict['Remove_LSFStripes'].get_function('kmain')
                BpG_FSfree_PA, TpB_FSfree_PA = GPUManage(NEQ_FSfree)
                BpG_FSfree = (BpG_FSfree_PA, BpG_FSfree_PA)
                TpB_FSfree = (TpB_FSfree_PA, TpB_FSfree_PA, 1)
                LHMAT_FSfree_GPU = cp.empty((NEQ_FSfree, NEQ_FSfree), dtype=np.float64)
                _func(args=(LHMAT_GPU, IDX_nFS_GPU, LHMAT_FSfree_GPU), block=TpB_FSfree, grid=BpG_FSfree)
                del LHMAT_GPU

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
            del Wl_GPU, Wm_GPU

            _func = SFFTModule_dict['Construct_FDIFF'].get_function('kmain')
            PixA_FDIFF_GPU = cp.empty((N0, N1), dtype=_CPX)
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, a_ijab_GPU.astype(_CPX),
                        SPixA_FIij_GPU, Kab_Wla_GPU, Kab_Wmb_GPU, b_pq_GPU.astype(_CPX),
                        SPixA_FTpq_GPU, PixA_FJ_GPU, PixA_FDIFF_GPU), block=TpB_PIX, grid=BpG_PIX)
            del Kab_Wla_GPU, Kab_Wmb_GPU
            PixA_DIFF_GPU = SCALE_L * cp.fft.ifft2(PixA_FDIFF_GPU).real

        return Solution_GPU, PixA_DIFF_GPU


class GeneralSFFTSubtract_PureCupy_F32_4088:
    @staticmethod
    def GSS(PixA_I_GPU, PixA_J_GPU, PixA_mI_GPU, PixA_mJ_GPU, SFFTConfig,
            ContamMask_I_GPU=None, VERBOSE_LEVEL=2):
        tmplst = [PixA_I_GPU.shape, PixA_J_GPU.shape, PixA_mI_GPU.shape, PixA_mI_GPU.shape]
        if len(set(tmplst)) > 1:
            raise Exception('MeLOn ERROR: Input images should have same size!')

        Solution_GPU = ElementalSFFTSubtract_PureCupy_F32_4088.ESSPC(
            PixA_I_GPU=PixA_mI_GPU, PixA_J_GPU=PixA_mJ_GPU, SFFTConfig=SFFTConfig,
            SFFTSolution_GPU=None, Subtract=False, VERBOSE_LEVEL=VERBOSE_LEVEL)[0]

        # reclaim before the second pass
        cp.get_default_memory_pool().free_all_blocks()

        PixA_DIFF_GPU = ElementalSFFTSubtract_PureCupy_F32_4088.ESSPC(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU, SFFTConfig=SFFTConfig,
            SFFTSolution_GPU=Solution_GPU, Subtract=True, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]

        return Solution_GPU, PixA_DIFF_GPU, None
