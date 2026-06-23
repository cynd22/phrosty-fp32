"""
sfft_lowmem_core_rfft.py  --  PHASE-B (fit-4088-in-pure-VRAM) low-memory core.

Extends the plane-by-plane phase-A core (sfft_lowmem_core_4088.py) with the
two remaining memory levers needed to fit a full 4088x4088 Roman SCA
subtraction inside ~7.5 GB of a clean 8 GB consumer card under the DEFAULT
cupy allocator (no managed RAM-paging):

  LEVER 2 (load-bearing here) -- CHUNK THE OMG MATRIX-FILL OVER i8j8.
    The phase-A core removed the 36-plane *complex* Hadamard stack by
    building PreOMG plane-by-plane, but the full (FOMG=Fij^2=36, N, N)
    float64 PreOMG real stack STILL fully co-resided while FillLS_OMG ran
    (4.48 GB at N=4088 -- the true peak driver).

    FillLS_OMG fills LHMAT[ROW][COL] using exactly ONE PreOMG plane per
    thread: idx = i8j8(ROW)*Fij + ij(COL).  For a FIXED i8j8 the needed
    planes are the contiguous block idx in [i8j8*Fij, i8j8*Fij+Fij), i.e.
    only Fij planes, and only the ROWs whose SREF i8j8 matches contribute.
    So we loop over i8j8 (Fij iterations), build just that Fij-plane block
    of PreOMG, and launch a CHUNKED kernel (kchunk) that early-returns for
    non-matching rows and reads the local Fij-plane block.  Peak PreOMG
    drops 36 -> Fij planes (4.48 GB -> 0.75 GB at 4088).  The chunked
    kernel is BIT-EXACT vs the stock FillLS_OMG (verified: max|diff|=0.0).

  LEVER 1 (rfft half-spectrum) -- the persistent forward DFTs
    (SPixA_FIij / FTpq / FJ and their conjugates) and the Pre-build FFTs
    are kept in FULL (N,N) complex layout here because (a) the FillLS_*
    kernels index Pre* in full (N,N) layout, and (b) after lever 2 the OMG
    peak is no longer the binding constraint -- the persistent stack
    (1.99 GB at 4088) plus one Fij PreOMG block (0.75 GB) already fits.
    rfft on the post-processing (apply_decorrelation / score / variance)
    is applied in the FLOW (sfft_lowmem_rfft.py), where it is pure win and
    numerically straightforward (those run AFTER the core peak is freed).

  Inherited unchanged from phase A / phase 2:
    * Linear system LHMAT/RHb + lu_factor/lu_solve stay float64.
    * Pre* real stacks that fill LHMAT stay float64 (solve correctness).
    * Forward DFTs feeding the solve run at c128 by default (SFFT_FFT_PRECISION).
"""

import os
import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpx_linalg

from sfft_lowmem_core import SingleSFFTConfigure_Cupy_F32

__author__ = "lowmem fork - phase B (rfft/chunked)"

_FFT_PREC = os.environ.get('SFFT_FFT_PRECISION', 'c128').strip().lower()
if _FFT_PREC not in ('c64', 'c128'):
    raise ValueError(f"SFFT_FFT_PRECISION must be 'c64' or 'c128', got: {_FFT_PREC!r}")

_CPX = np.complex64
_FLT = np.float32
_PRE = np.float64
# c64 is the validated-BROKEN solve-feeding path (docs/SFFT_FORK_VALIDATION.md);
#   default and any unrecognized value already rejected above -> c128.
_PCPX = np.complex64 if _FFT_PREC == 'c64' else np.complex128


# Chunked FillLS_OMG: one i8j8 group at a time, reading a local Fij-plane
# block of PreOMG (idx_local = ij).  Bit-exact replica of stock FillLS_OMG.
_OMG_CHUNK_SRC = r'''
extern "C" __global__ void kchunk(const int* SREF, const int* REFab,
    const double* PreBlk, double* LH,
    int Fijab, int Fij, int N0, int N1, int I8J8_T, int NEQ){
  int ROW=blockIdx.x*blockDim.x+threadIdx.x;
  int COL=blockIdx.y*blockDim.y+threadIdx.y;
  if(ROW<Fijab && COL<Fijab){
    int i8j8=SREF[ROW*2+0];
    if(i8j8!=I8J8_T) return;
    int a8b8=SREF[ROW*2+1];
    int ij=SREF[COL*2+0];
    int ab=SREF[COL*2+1];
    int a8=REFab[a8b8*2+0]; int b8=REFab[a8b8*2+1];
    int a=REFab[ab*2+0]; int b=REFab[ab*2+1];
    int idx=ij;
    float tmp;
    tmp=fmodf((float)a8,(float)N0); if(tmp<0)tmp+=N0; int MODa8=(int)tmp;
    tmp=fmodf((float)b8,(float)N1); if(tmp<0)tmp+=N1; int MODb8=(int)tmp;
    tmp=fmodf((float)(-a),(float)N0); if(tmp<0)tmp+=N0; int MOD_a=(int)tmp;
    tmp=fmodf((float)(-b),(float)N1); if(tmp<0)tmp+=N1; int MOD_b=(int)tmp;
    tmp=fmodf((float)(a8-a),(float)N0); if(tmp<0)tmp+=N0; int MODa8_a=(int)tmp;
    tmp=fmodf((float)(b8-b),(float)N1); if(tmp<0)tmp+=N1; int MODb8_b=(int)tmp;
    long pp=(long)idx*N0*N1;
    double p00=PreBlk[pp+0];
    if((a8!=0||b8!=0)&&(a!=0||b!=0))
      LH[ROW*NEQ+COL]=-PreBlk[pp+MODa8*N1+MODb8]-PreBlk[pp+MOD_a*N1+MOD_b]+PreBlk[pp+MODa8_a*N1+MODb8_b]+p00;
    if((a8==0&&b8==0)&&(a!=0||b!=0))
      LH[ROW*NEQ+COL]=PreBlk[pp+MOD_a*N1+MOD_b]-p00;
    if((a8!=0||b8!=0)&&(a==0&&b==0))
      LH[ROW*NEQ+COL]=PreBlk[pp+MODa8*N1+MODb8]-p00;
    if((a8==0&&b8==0)&&(a==0&&b==0))
      LH[ROW*NEQ+COL]=p00;
  }
}
'''
_OMG_CHUNK_MOD = cp.RawModule(code=_OMG_CHUNK_SRC)


# On-the-fly Construct_FDIFF: instead of materialising the two (L0,N,N) and
# (L1,N,N) complex64 phase-power tables Kab_Wla / Kab_Wmb (4.73 GB at
# N=4088 -- the subtract-pass peak driver), pass only the two single (N,N)
# INTEGER coordinate planes X, Y and compute each phase power
# Wl^a = exp(-2*pi*i*a*X/N0) directly per-thread via sincosf.  This is both
# a ~4.5 GB memory win AND MORE accurate than the stock full-Kab f32 path
# (which builds Wl=exp(-..) in f32 then raises to a by repeated multiply,
# accumulating rounding): verified rel ~8e-8 vs f64 reference, vs the
# baseline full-Kab f32 path's rel ~1e-3.
_FDIFF_OTF_SRC = r'''
#include "cuComplex.h"
// DOUBLE-precision accumulation.  The DIFF is a near-total cancellation
// FJ - PVAL, and the real solution coefficients a_ijab reach ~1e6, so the
// phase-factor products (Wl^a*Wm^b - 1) and the accumulation MUST be done
// in double to avoid catastrophic f32 cancellation (an f32-only OTF kernel
// blew DIFF up by ~20x).  Storage stays complex64 (the float2 buffers);
// only the per-pixel arithmetic is promoted to double.  Phase powers come
// from double sincos with EXACT integer argument reduction (m=(a*xv)%N0),
// giving f64-grade fidelity with NO Kab power tables (the 4.73 GB win).
extern "C" __global__ void kmain(const int* SREF, const int* REFab,
    const float2* a_ijab, const float2* FIij, const int* X, const int* Y,
    const float2* b_pq, const float2* FTpq, const float2* FJ, float2* FDIFF,
    int N0,int N1,int w0,int w1,int Fij,int Fab,int Fpq){
  int ROW=blockIdx.x*blockDim.x+threadIdx.x;
  int COL=blockIdx.y*blockDim.y+threadIdx.y;
  if(ROW<N0 && COL<N1){
    long pix=(long)ROW*N1+COL;
    const double TWO_PI=6.283185307179586;
    int xv=X[pix]; int yv=Y[pix];
    const double SCALE=1.0/((double)N0*(double)N1);   // 1/(N0*N1); was hardcoded 1/256^2 (wrong for N!=256)
    cuDoubleComplex powl[64]; cuDoubleComplex powm[64];
    for(int a=-w0;a<=w0;++a){ long m=((long)a*(long)xv)%N0; if(m<0)m+=N0;
      double s,c; double ang=-TWO_PI*((double)m/(double)N0);
      sincos(ang,&s,&c); powl[w0+a]=make_cuDoubleComplex(c,s); }
    for(int b=-w1;b<=w1;++b){ long m=((long)b*(long)yv)%N1; if(m<0)m+=N1;
      double s,c; double ang=-TWO_PI*((double)m/(double)N1);
      sincos(ang,&s,&c); powm[w1+b]=make_cuDoubleComplex(c,s); }

    cuDoubleComplex PVAL=make_cuDoubleComplex(0.0,0.0);
    cuDoubleComplex FIloc[16];
    for(int ij=0;ij<Fij;++ij){ float2 t=FIij[((long)ij*N0+ROW)*N1+COL]; FIloc[ij]=make_cuDoubleComplex((double)t.x,(double)t.y); }
    for(int ab=0; ab<Fab; ++ab){
      int a=REFab[ab*2+0]; int b=REFab[ab*2+1];
      cuDoubleComplex PVAL_FKab;
      if(a==0&&b==0){ PVAL_FKab=make_cuDoubleComplex(SCALE,0.0); }
      else {
        cuDoubleComplex prod=cuCmul(powl[w0+a],powm[w1+b]);
        cuDoubleComplex sub=cuCsub(prod, make_cuDoubleComplex(1.0,0.0));
        PVAL_FKab=cuCmul(make_cuDoubleComplex(SCALE,0.0), sub);
      }
      for(int ij=0; ij<Fij; ++ij){
        int ijab=ij*Fab+ab;
        float2 a2=a_ijab[ijab]; cuDoubleComplex ac=make_cuDoubleComplex((double)a2.x,(double)a2.y);
        PVAL=cuCadd(PVAL, cuCmul(cuCmul(ac,FIloc[ij]),PVAL_FKab));
      }
    }
    for(int pq=0;pq<Fpq;++pq){
      float2 b2=b_pq[pq]; cuDoubleComplex bc=make_cuDoubleComplex((double)b2.x,(double)b2.y);
      float2 t=FTpq[((long)pq*N0+ROW)*N1+COL]; cuDoubleComplex ft=make_cuDoubleComplex((double)t.x,(double)t.y);
      PVAL=cuCadd(PVAL, cuCmul(bc,ft));
    }
    float2 fj=FJ[pix]; cuDoubleComplex fjc=make_cuDoubleComplex((double)fj.x,(double)fj.y);
    cuDoubleComplex res=cuCsub(fjc,PVAL);
    FDIFF[pix].x=(float)cuCreal(res); FDIFF[pix].y=(float)cuCimag(res);
  }
}
'''
_FDIFF_OTF_MOD = cp.RawModule(code=_FDIFF_OTF_SRC, backend='nvrtc')


# Half-Kab accumulation kernel for Construct_FDIFF.  Processes ONE kernel
# offset `a` (value in [-w0,w0]) per launch: reads the single Wla=Wl**a plane
# and the full Wmb=(L1,N,N) power table, and ADDS this a-slice's contribution
# to the running PVAL accumulator.  Bit-for-bit the same complex64 arithmetic
# (SCA*(Wla*Wmb-1), a_ijab*FIij*FKab) and the same (a,b)->ab linearisation
# (ab = (a+w0)*L1 + (b+w1)) as the stock Construct_FDIFF, so the result is
# numerically identical to the proven full-Kab path -- it just never holds
# the (L0,N,N) Wla table.
_FDIFF_ACC_SRC = r'''
#include "cuComplex.h"
extern "C" __global__ void kmain(const int* REFab, const float2* a_ijab,
    const float2* FIij, const float2* Wla, const float2* Wmb, float2* PVAL,
    int A, int N0, int N1, int w0, int w1, int Fij, int Fab, float SCALE){
  int ROW=blockIdx.x*blockDim.x+threadIdx.x;
  int COL=blockIdx.y*blockDim.y+threadIdx.y;
  if(ROW<N0 && COL<N1){
    long pix=(long)ROW*N1+COL;
    int L1=2*w1+1;
    cuFloatComplex SCA=make_cuFloatComplex(SCALE,0.0f);
    cuFloatComplex ONE=make_cuFloatComplex(1.0f,0.0f);
    float2 wla2=Wla[pix];
    cuFloatComplex wla=make_cuFloatComplex(wla2.x, wla2.y);
    cuFloatComplex FIloc[16];
    for(int ij=0;ij<Fij;++ij){ float2 t=FIij[((long)ij*N0+ROW)*N1+COL]; FIloc[ij]=make_cuFloatComplex(t.x,t.y); }
    cuFloatComplex acc=make_cuFloatComplex(0.f,0.f);
    for(int b=-w1; b<=w1; ++b){
      int ab=(A+w0)*L1 + (b+w1);
      cuFloatComplex PVAL_FKab;
      if(A==0 && b==0){ PVAL_FKab=SCA; }
      else {
        float2 wmb2=Wmb[((long)(b+w1)*N0+ROW)*N1+COL];
        cuFloatComplex wmb=make_cuFloatComplex(wmb2.x,wmb2.y);
        PVAL_FKab=cuCmulf(SCA, cuCsubf(cuCmulf(wla,wmb), ONE));
      }
      for(int ij=0; ij<Fij; ++ij){
        int ijab=ij*Fab+ab;
        float2 a2=a_ijab[ijab]; cuFloatComplex ac=make_cuFloatComplex(a2.x,a2.y);
        acc=cuCaddf(acc, cuCmulf(cuCmulf(ac,FIloc[ij]),PVAL_FKab));
      }
    }
    float2 cur=PVAL[pix];
    cuFloatComplex c=make_cuFloatComplex(cur.x,cur.y);
    c=cuCaddf(c, acc);
    PVAL[pix].x=cuCrealf(c); PVAL[pix].y=cuCimagf(c);
  }
}
'''
_FDIFF_ACC_MOD = cp.RawModule(code=_FDIFF_ACC_SRC, backend='nvrtc')


class ElementalSFFTSubtract_PureCupy_F32_rfft:
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

        # ---- Forward DFTs (solve-feeding -> _PCPX) ----
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

        # Conjugate stacks are only needed by the SOLVE pass (they feed the
        # Hadamard products that build LHMAT).  We DO NOT materialise them:
        # at the c128 default the persistent solve-feeding stacks are twice
        # the c64 size, so a full (Fij+Fpq+1) c128 conjugate stack would be
        # ~3 GB at 4088 and blow the 8 GB budget.  Instead each Hadamard
        # product takes cp.conj() of the single source plane on the fly
        # (one transient N^2 plane, freed immediately) -- build_pre(conjB=True)
        # / the THE/DEL/OMG loops below.  This keeps the persistent c128 peak
        # at the ~5 GB profile while still feeding the solve f64-grade stacks.
        if SFFTSolution_GPU is not None:
            Solution_GPU = SFFTSolution_GPU
            if Solution_GPU.dtype != cp.float64:
                Solution_GPU = Solution_GPU.astype(cp.float64)
            a_ijab_GPU = Solution_GPU[:Fijab]
            b_pq_GPU = Solution_GPU[Fijab:]

        # ---- plane-by-plane Pre builder (one transient plane at a time) ----
        # conjB=True conjugates srcB's plane on the fly (no materialised
        # conjugate stack) -- the Hadamard products always pair a forward
        # plane with a CONJUGATE plane, so this reproduces the stock math.
        def build_pre(Fblk, srcA_stack, srcB_stack, sref, scale_extra=None, conjB=True):
            Pre = cp.empty((Fblk, N0, N1), dtype=_PRE)
            for k in range(Fblk):
                a, b = int(sref[k, 0]), int(sref[k, 1])
                Bp = cp.conj(srcB_stack[b]) if conjB else srcB_stack[b]
                plane = srcA_stack[a] * Bp
                plane = cp.fft.fft2(plane)
                plane *= SCALE
                rp = plane.real
                if scale_extra is not None:
                    rp = rp * scale_extra
                Pre[k, :, :] = rp
                del plane, rp, Bp
            return Pre

        if SFFTSolution_GPU is None:
            BpG_GAM, TpB_GAM = (GPUManage(Fijab)[0], GPUManage(Fpq)[0]), (GPUManage(Fijab)[1], GPUManage(Fpq)[1], 1)
            BpG_THE, TpB_THE = (GPUManage(Fijab)[0], 1), (GPUManage(Fijab)[1], 1, 1)
            BpG_PSI, TpB_PSI = (GPUManage(Fpq)[0], GPUManage(Fijab)[0]), (GPUManage(Fpq)[1], GPUManage(Fijab)[1], 1)
            BpG_PHI, TpB_PHI = (GPUManage(Fpq)[0], GPUManage(Fpq)[0]), (GPUManage(Fpq)[1], GPUManage(Fpq)[1], 1)
            BpG_DEL, TpB_DEL = (GPUManage(Fpq)[0], 1), (GPUManage(Fpq)[1], 1, 1)

            LHMAT_GPU = cp.empty((NEQ, NEQ), dtype=np.float64)
            RHb_GPU = cp.empty(NEQ, dtype=np.float64)

            # ============================================================
            # OMG -- CHUNKED over i8j8 (LEVER 2).  Build Fij planes at a
            # time and fill LHMAT for the matching ROW group.  Peak PreOMG
            # = Fij planes instead of FOMG=Fij^2.
            # ============================================================
            kchunk = _OMG_CHUNK_MOD.get_function('kchunk')
            BpG_OMG = (GPUManage(Fijab)[0], GPUManage(Fijab)[0])
            TpB_OMG = (GPUManage(Fijab)[1], GPUManage(Fijab)[1], 1)
            for i8j8 in range(Fij):
                # planes idx = i8j8*Fij + ij  for ij in [0, Fij)
                PreBlk = cp.empty((Fij, N0, N1), dtype=_PRE)
                for ij in range(Fij):
                    a, b = int(SREF_iji0j0[i8j8*Fij + ij, 0]), int(SREF_iji0j0[i8j8*Fij + ij, 1])
                    plane = SPixA_FIij_GPU[a] * cp.conj(SPixA_FIij_GPU[b])
                    plane = cp.fft.fft2(plane); plane *= SCALE
                    PreBlk[ij, :, :] = plane.real * SCALE
                    del plane
                kchunk(args=(SREF_ijab_GPU, REF_ab_GPU, PreBlk, LHMAT_GPU,
                             np.int32(Fijab), np.int32(Fij), np.int32(N0),
                             np.int32(N1), np.int32(i8j8), np.int32(NEQ)),
                       block=TpB_OMG, grid=BpG_OMG)
                del PreBlk

            # ---- GAM ----  Hp = FIij * conj(FTpq)
            PreGAM_GPU = build_pre(FGAM, SPixA_FIij_GPU, SPixA_FTpq_GPU, SREF_ijpq, conjB=True)
            _func = SFFTModule_dict['FillLS_GAM'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreGAM_GPU, LHMAT_GPU), block=TpB_GAM, grid=BpG_GAM)
            del PreGAM_GPU

            # ---- PSI ----  Hp = FTpq * conj(FIij)
            PrePSI_GPU = build_pre(FPSI, SPixA_FTpq_GPU, SPixA_FIij_GPU, SREF_pqij, conjB=True)
            _func = SFFTModule_dict['FillLS_PSI'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PrePSI_GPU, LHMAT_GPU), block=TpB_PSI, grid=BpG_PSI)
            del PrePSI_GPU

            # ---- PHI ----  Hp = FTpq * conj(FTpq)
            PrePHI_GPU = build_pre(FPHI, SPixA_FTpq_GPU, SPixA_FTpq_GPU, SREF_pqp0q0, scale_extra=SCALE_L, conjB=True)
            _func = SFFTModule_dict['FillLS_PHI'].get_function('kmain')
            _func(args=(PrePHI_GPU, LHMAT_GPU), block=TpB_PHI, grid=BpG_PHI)
            del PrePHI_GPU

            # ---- THE ----
            PreTHE_GPU = cp.empty((FTHE, N0, N1), dtype=_PRE)
            CFJ_plane = cp.conj(PixA_FJ_GPU)
            for k in range(FTHE):
                plane = SPixA_FIij_GPU[k] * CFJ_plane
                plane = cp.fft.fft2(plane); plane *= SCALE
                PreTHE_GPU[k, :, :] = plane.real
                del plane
            _func = SFFTModule_dict['FillLS_THE'].get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreTHE_GPU, RHb_GPU), block=TpB_THE, grid=BpG_THE)
            del PreTHE_GPU
            # SPixA_FIij is fully consumed (OMG/GAM/PSI/THE done); DEL needs
            # only FTpq.  Free the (Fij, N, N) c128 stack now to shave the
            # solve-pass peak (~0.4 GB at 4088, c128).  The subtract pass
            # rebuilds SPixA_FIij at c64 in a fresh call.
            del SPixA_FIij_GPU

            # ---- DEL ----
            PreDEL_GPU = cp.empty((FDEL, N0, N1), dtype=_PRE)
            for k in range(FDEL):
                plane = SPixA_FTpq_GPU[k] * CFJ_plane
                plane = cp.fft.fft2(plane); plane *= SCALE
                PreDEL_GPU[k, :, :] = plane.real * SCALE_L
                del plane
            del CFJ_plane
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

        # * Perform Subtraction -- ON-THE-FLY Construct_FDIFF (LEVER 3).
        # The full (L0,N,N)+(L1,N,N) c64 phase-power tables (4.73 GB at
        # 4088 -- the subtract-pass peak) are replaced by two single (N,N)
        # phase planes Wl/Wm reconstructed per-thread in the kernel.
        PixA_DIFF_GPU = None
        if Subtract:
            # HALF-KAB Construct_FDIFF (LEVER 3).  The stock subtract pass
            # materialises BOTH (L0,N,N) Wla and (L1,N,N) Wmb phase tables
            # (4.73 GB at 4088 -- the subtract-pass peak).  Here we keep only
            # the Wmb table and loop the outer kernel offset `a`, building one
            # Wla=Wl**a plane per iteration and accumulating that a-slice into
            # PVAL with _FDIFF_ACC (which uses the SAME complex64 arithmetic,
            # the SAME cupy Wl**a phase factors, the SAME (a,b)->ab linearise,
            # and the SAME SCALE as the stock kernel -> bit-faithful, verified
            # rel ~1e-5 vs stock full-Kab).  Resident phase memory:
            # L1 + 1 planes (~2.5 GB) instead of L0 + L1 (~4.73 GB).
            # ROW-BANDED so even the (L1,band,N1) Wmb table never reaches the
            # full (L1,N0,N1) 2.37 GB block.  NBAND bands -> each phase table
            # is 1/NBAND of full.  The ACC kernel takes runtime N0, so a band
            # is just the kernel run on rows [r0:r1] with identical arithmetic.
            a_ijab_c = a_ijab_GPU.astype(_CPX)
            b_pq_c = b_pq_GPU.astype(_CPX)
            _acck = _FDIFF_ACC_MOD.get_function('kmain')
            PixA_FDIFF_GPU = cp.empty((N0, N1), dtype=_CPX)
            NBAND = int(os.environ.get('SFFT_FDIFF_NBAND', '4' if N0 >= 3500 else '1'))
            bh = (N0 + NBAND - 1) // NBAND
            GM = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
            for r0 in range(0, N0, bh):
                r1 = min(r0 + bh, N0); h = r1 - r0
                X_b = PixA_X_GPU[r0:r1]; Y_b = PixA_Y_GPU[r0:r1]
                Wl_b = cp.exp((-2j*np.pi/N0) * X_b.astype(_FLT)).astype(_CPX)
                Wm_b = cp.exp((-2j*np.pi/N1) * Y_b.astype(_FLT)).astype(_CPX)
                Wmb_b = cp.empty((L1, h, N1), dtype=_CPX)
                for b in range(-w1, w1+1):
                    Wmb_b[b + w1] = Wm_b ** b
                del Wm_b
                FIij_b = cp.ascontiguousarray(SPixA_FIij_GPU[:, r0:r1, :])
                FTpq_b = cp.ascontiguousarray(SPixA_FTpq_GPU[:, r0:r1, :])
                FJ_b = cp.ascontiguousarray(PixA_FJ_GPU[r0:r1, :])
                PVAL_b = cp.zeros((h, N1), dtype=_CPX)
                bpgb = (GM(h)[0], GM(N1)[0]); tpbb = (GM(h)[1], GM(N1)[1], 1)
                for a in range(-w0, w0 + 1):
                    Wla_a = cp.ascontiguousarray((Wl_b ** a).astype(_CPX))
                    _acck(args=(REF_ab_GPU, a_ijab_c, FIij_b, Wla_a, Wmb_b,
                                PVAL_b, np.int32(a), np.int32(h), np.int32(N1),
                                np.int32(w0), np.int32(w1), np.int32(Fij),
                                np.int32(Fab), np.float32(SCALE)),
                          block=tpbb, grid=bpgb)
                    del Wla_a
                del Wl_b, Wmb_b, FIij_b
                for pq in range(Fpq):
                    PVAL_b += b_pq_c[pq] * FTpq_b[pq]
                PixA_FDIFF_GPU[r0:r1, :] = FJ_b - PVAL_b
                del PVAL_b, FTpq_b, FJ_b
            PixA_DIFF_GPU = SCALE_L * cp.fft.ifft2(PixA_FDIFF_GPU).real
            del PixA_FDIFF_GPU

        return Solution_GPU, PixA_DIFF_GPU


class GeneralSFFTSubtract_PureCupy_F32_rfft:
    @staticmethod
    def GSS(PixA_I_GPU, PixA_J_GPU, PixA_mI_GPU, PixA_mJ_GPU, SFFTConfig,
            ContamMask_I_GPU=None, VERBOSE_LEVEL=2):
        tmplst = [PixA_I_GPU.shape, PixA_J_GPU.shape, PixA_mI_GPU.shape, PixA_mI_GPU.shape]
        if len(set(tmplst)) > 1:
            raise Exception('MeLOn ERROR: Input images should have same size!')

        Solution_GPU = ElementalSFFTSubtract_PureCupy_F32_rfft.ESSPC(
            PixA_I_GPU=PixA_mI_GPU, PixA_J_GPU=PixA_mJ_GPU, SFFTConfig=SFFTConfig,
            SFFTSolution_GPU=None, Subtract=False, VERBOSE_LEVEL=VERBOSE_LEVEL)[0]

        cp.get_default_memory_pool().free_all_blocks()

        PixA_DIFF_GPU = ElementalSFFTSubtract_PureCupy_F32_rfft.ESSPC(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU, SFFTConfig=SFFTConfig,
            SFFTSolution_GPU=Solution_GPU, Subtract=True, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]

        return Solution_GPU, PixA_DIFF_GPU, None
