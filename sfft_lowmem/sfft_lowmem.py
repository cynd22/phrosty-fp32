"""
sfft_lowmem.py  --  Single-precision / low-memory SpaceSFFT fork.

Top-level entry point for an 8 GB consumer GPU.  Defines:
  * PureCupy_Customized_Packet_F32.PCCP  -- relaxed-dtype (float32-ok)
        customized packet that drives the forked float32 core in
        sfft_lowmem_core.py.
  * SpaceSFFT_CupyFlow_LowMem            -- copy of sfft's
        SpaceSFFT_CupyFlow with: no float64-forcing of inputs, float32
        flow buffers, complex64 decorrelation, and free-after-use of the
        big intermediates.

This module copies source.  It does NOT monkeypatch the installed sfft;
the dtype is baked into JIT kernel strings (handled in the core fork).
The stock A100/float64 path keeps importing `sfft.SpaceSFFTCupyFlow`.

Memory policy (per PLAN):
  Phase 1 : kill flow-level float64 forcing  (this file)
  Phase 2 : forked float32/complex64 core     (sfft_lowmem_core.py)
  Phase 4 : free-after-use of large flow intermediates (this file)
The decorrelation FFTs in find/apply/score/variance run in numpy on the
CPU in stock sfft (np.fft) -- we keep that (it is off the GPU peak) but
narrow GPU-resident arrays to float32/complex64.
"""

import cupy as cp
import numpy as np

from sfft.utils.PureCupyFFTKits import PureCupy_FFTKits
from sfft.utils.PatternRotationCalculator import PatternRotation_Calculator
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator, KERNEL_CSZ, KERNEL_CSZ_INV
from sfft.utils.ResampKits import Cupy_ZoomRotate
from sfft.utils.ResampKits import Cupy_Resampling
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

from sfft_lowmem_core import (SingleSFFTConfigure_Cupy_F32,
                              GeneralSFFTSubtract_PureCupy_F32)

__author__ = "lowmem fork"

# real / complex working precision for the flow
_FLT = cp.float32
_CPX = cp.complex64


class PureCupy_Customized_Packet_F32:
    @staticmethod
    def PCCP(PixA_REF_GPU, PixA_SCI_GPU, PixA_mREF_GPU, PixA_mSCI_GPU,
             ForceConv, GKerHW, KerPolyOrder=2, BGPolyOrder=2,
             ConstPhotRatio=True, CUDA_DEVICE_4SUBTRACT='0', VERBOSE_LEVEL=2):
        """Relaxed-dtype PCCP: accepts float32 (or float64) and runs the
        forked single-precision core."""

        assert cp.sum(cp.isnan(PixA_mREF_GPU)) == 0, "masked REF contains NaNs!"
        assert cp.sum(cp.isnan(PixA_mSCI_GPU)) == 0, "masked SCI contains NaNs!"
        for arr in (PixA_REF_GPU, PixA_SCI_GPU, PixA_mREF_GPU, PixA_mSCI_GPU):
            assert arr.ndim == 2
            # RELAXED gate: float32 OR float64 accepted (was float64-only).
            assert arr.dtype in (cp.float32, cp.float64), \
                "PCCP expects float32 or float64!"

        assert ForceConv in ['REF', 'SCI']
        # Fixed-size CUDA kernel buffers cap the supported polynomial/kernel sizes:
        #   FIloc[16] -> Fij=(DK+1)(DK+2)/2 <= 16  => KerPolyOrder <= 4
        #   powl/powm[64] -> 2*GKerHW+1 <= 64       => GKerHW <= 31
        assert KerPolyOrder <= 4, "KerPolyOrder > 4 overflows the FIloc[16] kernel buffer"
        assert GKerHW <= 31, "GKerHW > 31 overflows the powl/powm[64] kernel buffers"
        ConvdSide = ForceConv
        KerHW = GKerHW

        device = cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT))
        device.use()

        NaNmask_REF_GPU = cp.isnan(PixA_REF_GPU)
        NaNmask_SCI_GPU = cp.isnan(PixA_SCI_GPU)
        NaNmask_GPU = None
        if NaNmask_REF_GPU.any() or NaNmask_SCI_GPU.any():
            NaNmask_GPU = cp.logical_or(NaNmask_REF_GPU, NaNmask_SCI_GPU)

        NX, NY = PixA_REF_GPU.shape
        SFFTConfig = SingleSFFTConfigure_Cupy_F32(
            NX=NX, NY=NY, KerHW=KerHW, KerPolyOrder=KerPolyOrder,
            BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio,
            VERBOSE_LEVEL=VERBOSE_LEVEL)

        if ConvdSide == 'REF':
            PixA_mI_GPU, PixA_mJ_GPU = PixA_mREF_GPU, PixA_mSCI_GPU
            if NaNmask_GPU is not None:
                PixA_I_GPU, PixA_J_GPU = PixA_REF_GPU.copy(), PixA_SCI_GPU.copy()
                PixA_I_GPU[NaNmask_GPU] = PixA_mI_GPU[NaNmask_GPU]
                PixA_J_GPU[NaNmask_GPU] = PixA_mJ_GPU[NaNmask_GPU]
            else:
                PixA_I_GPU, PixA_J_GPU = PixA_REF_GPU, PixA_SCI_GPU
        else:
            PixA_mI_GPU, PixA_mJ_GPU = PixA_mSCI_GPU, PixA_mREF_GPU
            if NaNmask_GPU is not None:
                PixA_I_GPU, PixA_J_GPU = PixA_SCI_GPU.copy(), PixA_REF_GPU.copy()
                PixA_I_GPU[NaNmask_GPU] = PixA_mI_GPU[NaNmask_GPU]
                PixA_J_GPU[NaNmask_GPU] = PixA_mJ_GPU[NaNmask_GPU]
            else:
                PixA_I_GPU, PixA_J_GPU = PixA_SCI_GPU, PixA_REF_GPU

        Solution_GPU, PixA_DIFF_GPU, _ = GeneralSFFTSubtract_PureCupy_F32.GSS(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU,
            PixA_mI_GPU=PixA_mI_GPU, PixA_mJ_GPU=PixA_mJ_GPU,
            SFFTConfig=SFFTConfig, ContamMask_I_GPU=None, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if NaNmask_GPU is not None:
            PixA_DIFF_GPU[NaNmask_GPU] = cp.nan
        if ConvdSide == 'SCI':
            PixA_DIFF_GPU *= -1.
        return Solution_GPU, PixA_DIFF_GPU


class SpaceSFFT_CupyFlow_LowMem:
    """float32/complex64 low-memory variant of sfft's SpaceSFFT_CupyFlow."""

    def __init__(self, hdr_target, hdr_object,
                 target_skyrms, object_skyrms,
                 PixA_target_GPU, PixA_object_GPU,
                 PixA_targetVar_GPU, PixA_objectVar_GPU,
                 PixA_target_DMASK_GPU, PixA_object_DMASK_GPU,
                 PSF_target_GPU, PSF_object_GPU,
                 sci_is_target=True,
                 GKerHW=9, KerPolyOrder=2, BGPolyOrder=0, ConstPhotRatio=True,
                 Consider_Matching_Kernel=False,
                 CUDA_COMPILER="nvrtc", CUDA_DEVICE_4SUBTRACT='0',
                 GAIN=1.0, RANDOM_SEED=10086):

        assert PixA_target_GPU.flags['C_CONTIGUOUS']
        assert PixA_object_GPU.flags['C_CONTIGUOUS']
        assert PixA_target_DMASK_GPU.flags['C_CONTIGUOUS']
        assert PixA_object_DMASK_GPU.flags['C_CONTIGUOUS']
        assert PSF_target_GPU.flags['C_CONTIGUOUS']
        assert PSF_object_GPU.flags['C_CONTIGUOUS']

        self.hdr_target = hdr_target
        self.hdr_object = hdr_object
        self.target_skyrms = target_skyrms
        self.object_skyrms = object_skyrms

        # ---- Phase 1: float32 flow ----
        # IMPORTANT: the stock sfft resampling / ZoomRotate CUDA kernels
        # (sfft.utils.ResampKits, line ~409) declare their image operands
        # as `double[...]` and are NOT recompiled here.  Feeding them
        # float32 makes the kernel read 8 bytes per 4-byte element ->
        # out-of-bounds (CUDA_ERROR_ILLEGAL_ADDRESS).  So we keep the
        # *resampler inputs* in float64 and narrow the *resampled outputs*
        # to float32 immediately after (see resampling_image_mask_psf).
        # This costs only a few transient full-frame f64 arrays, NOT the
        # big (Fij^2, N, N) core stacks (those stay f32 -> the real win).
        def _f64(a):
            return a if a.dtype == cp.float64 else a.astype(cp.float64)

        self.PixA_target_GPU = _f64(PixA_target_GPU)
        self.PixA_object_GPU = _f64(PixA_object_GPU)
        self.PixA_targetVar_GPU = _f64(PixA_targetVar_GPU)
        self.PixA_objectVar_GPU = _f64(PixA_objectVar_GPU)
        self.PixA_target_DMASK_GPU = _f64(PixA_target_DMASK_GPU)
        self.PixA_object_DMASK_GPU = _f64(PixA_object_DMASK_GPU)
        self.PSF_target_GPU = _f64(PSF_target_GPU)
        self.PSF_object_GPU = _f64(PSF_object_GPU)

        self.sci_is_target = sci_is_target
        self.GKerHW = GKerHW
        self.KerPolyOrder = KerPolyOrder
        self.BGPolyOrder = BGPolyOrder
        self.ConstPhotRatio = ConstPhotRatio
        self.Consider_Matching_Kernel = Consider_Matching_Kernel
        self.CUDA_COMPILER = CUDA_COMPILER
        self.CUDA_DEVICE_4SUBTRACT = CUDA_DEVICE_4SUBTRACT
        self.GAIN = GAIN
        self.RANDOM_SEED = RANDOM_SEED

    def resampling_image_mask_psf(self):
        CR = Cupy_Resampling(RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=1)

        if self.hdr_target["CTYPE1"] == "RA---TAN":
            assert self.hdr_target["CTYPE2"] == "DEC--TAN"
            XX_proj_GPU, YY_proj_GPU = CR.resamp_projection_cd(
                hdr_obj=self.hdr_object, hdr_targ=self.hdr_target, CDKEY="CD")
        if self.hdr_target["CTYPE1"] == "RA---TAN-SIP":
            assert self.hdr_target["CTYPE2"] == "DEC--TAN-SIP"
            XX_proj_GPU, YY_proj_GPU = CR.resamp_projection_sip(
                hdr_obj=self.hdr_object, hdr_targ=self.hdr_target,
                NSAMP=1024, RANDOM_SEED=self.RANDOM_SEED)

        NTX = int(self.hdr_target["NAXIS1"])
        NTY = int(self.hdr_target["NAXIS2"])
        NPIX_INNER = cp.sum(cp.logical_and(
            cp.logical_and(XX_proj_GPU >= 0.5, XX_proj_GPU < NTX+0.5),
            cp.logical_and(YY_proj_GPU >= 0.5, YY_proj_GPU < NTY+0.5)))
        assert NPIX_INNER > 0, "Projection completely outside target image!"

        PixA_Eobj_GPU, EProjDict = CR.frame_extension(
            XX_proj_GPU=XX_proj_GPU, YY_proj_GPU=YY_proj_GPU,
            PixA_obj_GPU=self.PixA_object_GPU, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.)
        self.PixA_resamp_object_GPU = CR.resampling(
            PixA_Eobj_GPU=PixA_Eobj_GPU, EProjDict=EProjDict,
            CUDA_COMPILER=self.CUDA_COMPILER, USE_SHARED_MEMORY=False)

        PixA_EobjVar_GPU, EProjDict_Var = CR.frame_extension(
            XX_proj_GPU=XX_proj_GPU, YY_proj_GPU=YY_proj_GPU,
            PixA_obj_GPU=self.PixA_objectVar_GPU, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.)
        self.PixA_resamp_objectVar_GPU = CR.resampling(
            PixA_Eobj_GPU=PixA_EobjVar_GPU, EProjDict=EProjDict,
            CUDA_COMPILER=self.CUDA_COMPILER, USE_SHARED_MEMORY=False)

        PixA_Eobj_GPU, EProjDict = CR.frame_extension(
            XX_proj_GPU=XX_proj_GPU, YY_proj_GPU=YY_proj_GPU,
            PixA_obj_GPU=self.PixA_object_DMASK_GPU, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.)
        del XX_proj_GPU
        del YY_proj_GPU
        self.PixA_resamp_object_DMASK_GPU = CR.resampling(
            PixA_Eobj_GPU=PixA_Eobj_GPU, EProjDict=EProjDict,
            CUDA_COMPILER=self.CUDA_COMPILER, USE_SHARED_MEMORY=False)
        self.BlankMask_GPU = self.PixA_resamp_object_GPU == 0.

        PATTERN_ROTATE_ANGLE = PatternRotation_Calculator.PRC(
            hdr_obj=self.hdr_object, hdr_targ=self.hdr_target)
        self.PSF_resamp_object_GPU = Cupy_ZoomRotate.CZR(
            PixA_obj_GPU=self.PSF_object_GPU, ZOOM_SCALE_X=1., ZOOM_SCALE_Y=1.,
            OUTSIZE_PARIRY_X='UNCHANGED', OUTSIZE_PARIRY_Y='UNCHANGED',
            PATTERN_ROTATE_ANGLE=PATTERN_ROTATE_ANGLE, RESAMP_METHOD='BILINEAR',
            PAD_FILL_VALUE=0., NAN_FILL_VALUE=0., THREAD_PER_BLOCK=8,
            USE_SHARED_MEMORY=False, VERBOSE_LEVEL=2)

        # Preserve FULL-PRECISION (float64) copies of the PSFs for the
        # decorrelation-kernel calculation.  DeCorrelation_Calculator
        # inverts a near-singular Fourier operator; feeding it f32-rounded
        # PSFs blows the decorrelation kernel up by ~1e16.  These PSF
        # arrays are tiny (image-sized? no -- PSF-sized, ~25x25), so the
        # f64 copy is free.
        self.PSF_target_f64 = self.PSF_target_GPU.astype(cp.float64)
        self.PSF_resamp_object_f64 = self.PSF_resamp_object_GPU.astype(cp.float64)

        # ---- Narrow everything on the SFFT path to float32 now that the
        #      double-typed resampling kernels are done. ----
        def _f32(a):
            return a if a.dtype == _FLT else a.astype(_FLT)
        self.PixA_target_GPU = _f32(self.PixA_target_GPU)
        self.PixA_targetVar_GPU = _f32(self.PixA_targetVar_GPU)
        self.PixA_resamp_object_GPU = _f32(self.PixA_resamp_object_GPU)
        self.PixA_resamp_objectVar_GPU = _f32(self.PixA_resamp_objectVar_GPU)
        self.PixA_target_DMASK_GPU = _f32(self.PixA_target_DMASK_GPU)
        self.PixA_resamp_object_DMASK_GPU = _f32(self.PixA_resamp_object_DMASK_GPU)
        self.PSF_target_GPU = _f32(self.PSF_target_GPU)
        self.PSF_object_GPU = _f32(self.PSF_object_GPU)
        self.PSF_resamp_object_GPU = _f32(self.PSF_resamp_object_GPU)

        # Phase 4: object image no longer needed after resampling.
        self.PixA_object_GPU = None

    def cross_convolution(self):
        self.PixA_Ctarget_GPU = PureCupy_FFTKits.FFT_CONVOLVE(
            PixA_Inp_GPU=self.PixA_target_GPU, KERNEL_GPU=self.PSF_resamp_object_GPU,
            PAD_FILL_VALUE=0., NAN_FILL_VALUE=None, NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True, FFT_BACKEND="Cupy")
        self.PSF_Ctarget_GPU = PureCupy_FFTKits.FFT_CONVOLVE(
            PixA_Inp_GPU=self.PSF_target_GPU, KERNEL_GPU=self.PSF_resamp_object_GPU,
            PAD_FILL_VALUE=0., NAN_FILL_VALUE=None, NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True, FFT_BACKEND="Cupy")
        self.PixA_Cresamp_object_GPU = PureCupy_FFTKits.FFT_CONVOLVE(
            PixA_Inp_GPU=self.PixA_resamp_object_GPU, KERNEL_GPU=self.PSF_target_GPU,
            PAD_FILL_VALUE=0., NAN_FILL_VALUE=None, NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True, FFT_BACKEND="Cupy")

    def sfft_subtraction(self):
        LYMASK_BKG_GPU = cp.logical_or(self.PixA_target_DMASK_GPU == 0,
                                       self.PixA_resamp_object_DMASK_GPU < 0.1)
        NaNmask_Ctarget_GPU = cp.isnan(self.PixA_Ctarget_GPU)
        NaNmask_Cresamp_object_GPU = cp.isnan(self.PixA_Cresamp_object_GPU)
        if NaNmask_Ctarget_GPU.any() or NaNmask_Cresamp_object_GPU.any():
            NaNmask_GPU = cp.logical_or(NaNmask_Ctarget_GPU, NaNmask_Cresamp_object_GPU)
            ZeroMask_GPU = cp.logical_or(NaNmask_GPU, LYMASK_BKG_GPU)
        else:
            ZeroMask_GPU = LYMASK_BKG_GPU
        del LYMASK_BKG_GPU

        PixA_mCtarget_GPU = self.PixA_Ctarget_GPU.copy()
        PixA_mCtarget_GPU[ZeroMask_GPU] = 0.
        PixA_mCresamp_object_GPU = self.PixA_Cresamp_object_GPU.copy()
        PixA_mCresamp_object_GPU[ZeroMask_GPU] = 0.
        del ZeroMask_GPU

        if self.sci_is_target:
            PixA_REF_GPU = self.PixA_Cresamp_object_GPU
            PixA_SCI_GPU = self.PixA_Ctarget_GPU
            PixA_mREF_GPU = PixA_mCresamp_object_GPU
            PixA_mSCI_GPU = PixA_mCtarget_GPU
        else:
            PixA_REF_GPU = self.PixA_Ctarget_GPU
            PixA_SCI_GPU = self.PixA_Cresamp_object_GPU
            PixA_mREF_GPU = PixA_mCtarget_GPU
            PixA_mSCI_GPU = PixA_mCresamp_object_GPU

        self.Solution_GPU, self.PixA_DIFF_GPU = PureCupy_Customized_Packet_F32.PCCP(
            PixA_REF_GPU=PixA_REF_GPU, PixA_SCI_GPU=PixA_SCI_GPU,
            PixA_mREF_GPU=PixA_mREF_GPU, PixA_mSCI_GPU=PixA_mSCI_GPU,
            ForceConv='REF' if self.sci_is_target else 'SCI',
            GKerHW=self.GKerHW, KerPolyOrder=self.KerPolyOrder,
            BGPolyOrder=self.BGPolyOrder, ConstPhotRatio=self.ConstPhotRatio,
            CUDA_DEVICE_4SUBTRACT=self.CUDA_DEVICE_4SUBTRACT)
        self.PixA_DIFF_GPU[self.BlankMask_GPU] = 0.

        # Phase 4: drop masked copies (consumed by the solve).
        del PixA_mCtarget_GPU
        del PixA_mCresamp_object_GPU

    def find_decorrelation(self):
        N0, N1 = self.PixA_DIFF_GPU.shape
        L0, L1 = 2*self.GKerHW + 1, 2*self.GKerHW + 1
        DK = self.KerPolyOrder
        Fpq = int((self.BGPolyOrder+1)*(self.BGPolyOrder+2)/2)
        XY_q = np.array([[N0/2.+0.5, N1/2.+0.5]])

        self.Solution = cp.asnumpy(self.Solution_GPU)
        MATCH_KERNEL_GPU = cp.array(Realize_MatchingKernel(XY_q=XY_q).FromArray(
            Solution=self.Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq
        )[0], dtype=_FLT)
        self.MATCH_KERNEL = cp.asnumpy(MATCH_KERNEL_GPU)

        if self.Consider_Matching_Kernel:
            MK = cp.asnumpy(MATCH_KERNEL_GPU)
        else:
            MK = None
        # DeCorrelation_Calculator runs on the CPU (numpy) -- off-GPU-peak.
        self.FKDECO = DeCorrelation_Calculator(
            NX_IMG=N0, NY_IMG=N1,
            KERNEL_JQueue=[cp.asnumpy(self.PSF_resamp_object_f64)],
            BKGSIG_JQueue=[self.target_skyrms],
            KERNEL_IQueue=[cp.asnumpy(self.PSF_target_f64)],
            BKGSIG_IQueue=[self.object_skyrms],
            MATCH_KERNEL=MK, REAL_OUTPUT=False, REAL_OUTPUT_SIZE=None,
            NORMALIZE_OUTPUT=True, VERBOSE_LEVEL=2)
        # KEEP complex128 here.  The decorrelation / score / variance FFTs
        # operate on FULL-FRAME images that still carry a large DC offset
        # (sky level ~100-200, variance ~200) over N^2 pixels -> the DFT's
        # zero-frequency bin is ~1e8.  In float32/complex64 that bin has
        # only ~7 sig figs, which swamps the small AC structure and
        # destroys the score (decorrelates) and inflates the variance by
        # the squared DC term.  These are only 2-3 transient full-frame
        # arrays AND they run AFTER the core's peak is freed, so float64
        # here costs ~nothing against the committed peak.
        self.FKDECO_GPU = cp.array(self.FKDECO, dtype=cp.complex128)

    def apply_decorrelation(self, img):
        _img = cp.asnumpy(img)
        if _img.shape == self.FKDECO.shape:
            FPixA = np.fft.fft2(_img)
            PixA_decorr = np.fft.ifft2(FPixA * self.FKDECO).real
            decorimg = cp.array(PixA_decorr, dtype=_FLT)
        else:
            NK0, NK1 = _img.shape
            N0, N1 = self.FKDECO.shape
            KERN_CSZ = KERNEL_CSZ(KERNEL=_img, NX_IMG=N0, NY_IMG=N1)
            FKERN_decorr = np.fft.fft2(KERN_CSZ) * self.FKDECO
            PixA_KERN_decorr = KERNEL_CSZ_INV(np.fft.ifft2(FKERN_decorr).real, NX_KERN=NK0, NY_KERN=NK1)
            decorimg = cp.array(PixA_KERN_decorr, dtype=_FLT)
        return decorimg

    def create_score_image(self):
        # full-frame post-processing FFTs run in float64/complex128 (see note
        # in find_decorrelation): cheap, transient, precision-critical.
        NX, NY = self.PixA_target_GPU.shape
        PSF_object_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=self.PSF_object_GPU.astype(cp.float64), NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=self.PSF_target_f64, NX_IMG=NX, NY_IMG=NY)
        FPSF_dDIFF_GPU = cp.fft.fft2(PSF_object_CSZ_GPU) * cp.fft.fft2(PSF_target_CSZ_GPU) * self.FKDECO_GPU
        FPixA_DIFF_GPU = cp.fft.fft2(self.PixA_DIFF_GPU.astype(cp.float64))
        FPixA_dDIFF_GPU = FPixA_DIFF_GPU * self.FKDECO_GPU
        FPixA_SCORE_GPU = FPixA_dDIFF_GPU * cp.conj(FPSF_dDIFF_GPU)
        PixA_SCORE_GPU = cp.fft.ifft2(FPixA_SCORE_GPU).real
        skysig_SCORE = SkyLevel_Estimator.SLE(PixA_obj=cp.asnumpy(PixA_SCORE_GPU))[1]
        PixA_SCORE_GPU /= skysig_SCORE
        return PixA_SCORE_GPU

    def create_variance_image(self):
        assert self.PixA_targetVar_GPU.flags['C_CONTIGUOUS']
        assert self.PixA_resamp_objectVar_GPU.flags['C_CONTIGUOUS']
        NX, NY = self.PixA_target_GPU.shape
        # float64/complex128 (variance images carry a large DC offset).
        PSF_resamp_object_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=self.PSF_resamp_object_f64, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=self.PSF_target_f64, NX_IMG=NX, NY_IMG=NY)
        Var_resamp_f64 = self.PixA_resamp_objectVar_GPU.astype(cp.float64)
        Var_target_f64 = self.PixA_targetVar_GPU.astype(cp.float64)

        # Phase 4: sequential materialize-and-free (caps live complex128
        # full spectra at ~2 instead of ~5).
        tmp = cp.fft.ifft2(cp.fft.fft2(PSF_target_CSZ_GPU) * self.FKDECO_GPU).real
        tmp = tmp ** 2
        Ft = cp.fft.fft2(Var_resamp_f64) * cp.fft.fft2(tmp)
        del tmp
        PixA_dDIFFVar_GPU = cp.fft.ifft2(Ft).real
        del Ft

        tmp = cp.fft.ifft2(cp.fft.fft2(PSF_resamp_object_CSZ_GPU) * self.FKDECO_GPU).real
        tmp = tmp ** 2
        Ft = cp.fft.fft2(Var_target_f64) * cp.fft.fft2(tmp)
        del tmp
        PixA_dDIFFVar_GPU += cp.fft.ifft2(Ft).real
        del Ft
        return PixA_dDIFFVar_GPU

    def cleanup(self):
        pass


# Convenience alias matching the stock public name.
SpaceSFFT_CupyFlow = SpaceSFFT_CupyFlow_LowMem
