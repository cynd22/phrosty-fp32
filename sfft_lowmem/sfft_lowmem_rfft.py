"""
sfft_lowmem_rfft.py  --  PHASE-B entry point (fit 4088 in pure 8 GB VRAM).

Subclasses the phase-A 4088 flow but:
  * routes the subtraction core through the CHUNKED-OMG core
    (GeneralSFFTSubtract_PureCupy_F32_rfft in sfft_lowmem_core_rfft.py),
    which caps the resident PreOMG stack at Fij planes instead of Fij^2
    (the 4.48 GB -> 0.75 GB peak win at N=4088); and
  * runs the float64 post-processing (apply_decorrelation / score /
    variance) with rfft2 / irfft2 half-spectra so each complex128 full
    spectrum (N0 x N1) shrinks to a half plane (N0 x (N1//2+1)).  This is
    a pure win and numerically faithful: the FKDECO / PSF-product
    multipliers are Hermitian (transforms of REAL kernels), so
    irfft2(rfft2(x) * MULT[:, :N1//2+1], s=(N0,N1)) reproduces the stock
    ifft2(fft2(x) * MULT).real to ~1e-13 (verified).

The 2560 / 3072 / 3584 paths keep working through the same code (the
chunked-OMG kernel is bit-exact vs stock; the rfft post-proc is exact).
"""
import numpy as np
import cupy as cp

from sfft_lowmem_4088 import SpaceSFFT_CupyFlow_LowMem_4088
from sfft_lowmem_core_rfft import GeneralSFFTSubtract_PureCupy_F32_rfft
from sfft_lowmem_core import SingleSFFTConfigure_Cupy_F32

from sfft.utils.PureCupyFFTKits import PureCupy_FFTKits
from sfft.utils.DeCorrelationCalculator import KERNEL_CSZ, KERNEL_CSZ_INV
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator

_FLT = cp.float32


class PureCupy_Customized_Packet_F32_rfft:
    @staticmethod
    def PCCP(PixA_REF_GPU, PixA_SCI_GPU, PixA_mREF_GPU, PixA_mSCI_GPU,
             ForceConv, GKerHW, KerPolyOrder=2, BGPolyOrder=2,
             ConstPhotRatio=True, CUDA_DEVICE_4SUBTRACT='0', VERBOSE_LEVEL=2):
        for arr in (PixA_REF_GPU, PixA_SCI_GPU, PixA_mREF_GPU, PixA_mSCI_GPU):
            assert arr.ndim == 2
            assert arr.dtype in (cp.float32, cp.float64)
        assert ForceConv in ['REF', 'SCI']
        # Fixed-size CUDA kernel buffers cap the supported polynomial/kernel sizes:
        #   FIloc[16] -> Fij=(DK+1)(DK+2)/2 <= 16  => KerPolyOrder <= 4
        #   powl/powm[64] -> 2*GKerHW+1 <= 64       => GKerHW <= 31
        assert KerPolyOrder <= 4, "KerPolyOrder > 4 overflows the FIloc[16] kernel buffer"
        assert GKerHW <= 31, "GKerHW > 31 overflows the powl/powm[64] kernel buffers"
        ConvdSide = ForceConv
        KerHW = GKerHW

        device = cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT)); device.use()

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

        Solution_GPU, PixA_DIFF_GPU, _ = GeneralSFFTSubtract_PureCupy_F32_rfft.GSS(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU,
            PixA_mI_GPU=PixA_mI_GPU, PixA_mJ_GPU=PixA_mJ_GPU,
            SFFTConfig=SFFTConfig, ContamMask_I_GPU=None, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if NaNmask_GPU is not None:
            PixA_DIFF_GPU[NaNmask_GPU] = cp.nan
        if ConvdSide == 'SCI':
            PixA_DIFF_GPU *= -1.
        return Solution_GPU, PixA_DIFF_GPU


class SpaceSFFT_CupyFlow_LowMem_rfft(SpaceSFFT_CupyFlow_LowMem_4088):

    # --- route subtraction through the chunked-OMG core ---
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

        self.Solution_GPU, self.PixA_DIFF_GPU = PureCupy_Customized_Packet_F32_rfft.PCCP(
            PixA_REF_GPU=PixA_REF_GPU, PixA_SCI_GPU=PixA_SCI_GPU,
            PixA_mREF_GPU=PixA_mREF_GPU, PixA_mSCI_GPU=PixA_mSCI_GPU,
            ForceConv='REF' if self.sci_is_target else 'SCI',
            GKerHW=self.GKerHW, KerPolyOrder=self.KerPolyOrder,
            BGPolyOrder=self.BGPolyOrder, ConstPhotRatio=self.ConstPhotRatio,
            CUDA_DEVICE_4SUBTRACT=self.CUDA_DEVICE_4SUBTRACT)
        self.PixA_DIFF_GPU[self.BlankMask_GPU] = 0.

        del PixA_mCtarget_GPU
        del PixA_mCresamp_object_GPU

    # --- rfft half-spectrum FKDECO + post-processing ---
    def find_decorrelation(self):
        # Build the full-plane FKDECO exactly as phase A, then cache a
        # half-plane (Hermitian) view for the rfft post-processing.
        super().find_decorrelation()
        N1 = self.FKDECO_GPU.shape[1]
        self._half = N1 // 2 + 1
        # half-plane complex128 multiplier (transform of a REAL kernel ->
        # Hermitian -> first half columns suffice for irfft2).
        self.FKDECO_half_GPU = cp.ascontiguousarray(self.FKDECO_GPU[:, :self._half])

    def apply_decorrelation(self, img):
        # GPU rfft path for full-frame images; CPU KERNEL_CSZ path for the
        # small-kernel case (inherited, off the GPU peak).
        if img.shape == self.FKDECO_GPU.shape:
            x = img.astype(cp.float64)
            N0, N1 = x.shape
            Fx = cp.fft.rfft2(x)
            out = cp.fft.irfft2(Fx * self.FKDECO_half_GPU, s=(N0, N1))
            return out.astype(_FLT)
        # fallback: stock CPU kernel-resize path
        return super().apply_decorrelation(img)

    def create_score_image(self):
        NX, NY = self.PixA_target_GPU.shape
        PSF_object_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(
            KERNEL_GPU=self.PSF_object_GPU.astype(cp.float64), NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(
            KERNEL_GPU=self.PSF_target_f64, NX_IMG=NX, NY_IMG=NY)
        h = self.FKDECO_half_GPU
        # FPSF_dDIFF (half) = rfft2(PSFobj)*rfft2(PSFtgt)*FKDECO   (Hermitian)
        FPSF_dDIFF_half = (cp.fft.rfft2(PSF_object_CSZ_GPU)
                           * cp.fft.rfft2(PSF_target_CSZ_GPU) * h)
        del PSF_object_CSZ_GPU, PSF_target_CSZ_GPU
        # full multiplier on rfft2(DIFF):  FKDECO * conj(FPSF_dDIFF)  (Hermitian)
        MULT_half = h * cp.conj(FPSF_dDIFF_half)
        del FPSF_dDIFF_half
        FPixA_DIFF_half = cp.fft.rfft2(self.PixA_DIFF_GPU.astype(cp.float64))
        PixA_SCORE_GPU = cp.fft.irfft2(FPixA_DIFF_half * MULT_half, s=(NX, NY))
        del FPixA_DIFF_half, MULT_half
        skysig_SCORE = SkyLevel_Estimator.SLE(PixA_obj=cp.asnumpy(PixA_SCORE_GPU))[1]
        PixA_SCORE_GPU /= skysig_SCORE
        return PixA_SCORE_GPU

    def create_variance_image(self):
        assert self.PixA_targetVar_GPU.flags['C_CONTIGUOUS']
        assert self.PixA_resamp_objectVar_GPU.flags['C_CONTIGUOUS']
        NX, NY = self.PixA_target_GPU.shape
        h = self.FKDECO_half_GPU
        PSF_resamp_object_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(
            KERNEL_GPU=self.PSF_resamp_object_f64, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(
            KERNEL_GPU=self.PSF_target_f64, NX_IMG=NX, NY_IMG=NY)
        Var_resamp_f64 = self.PixA_resamp_objectVar_GPU.astype(cp.float64)
        Var_target_f64 = self.PixA_targetVar_GPU.astype(cp.float64)

        # term 1
        tmp = cp.fft.irfft2(cp.fft.rfft2(PSF_target_CSZ_GPU) * h, s=(NX, NY))
        tmp = tmp ** 2
        Ft = cp.fft.rfft2(Var_resamp_f64) * cp.fft.rfft2(tmp)
        del tmp
        PixA_dDIFFVar_GPU = cp.fft.irfft2(Ft, s=(NX, NY))
        del Ft

        # term 2
        tmp = cp.fft.irfft2(cp.fft.rfft2(PSF_resamp_object_CSZ_GPU) * h, s=(NX, NY))
        tmp = tmp ** 2
        Ft = cp.fft.rfft2(Var_target_f64) * cp.fft.rfft2(tmp)
        del tmp
        PixA_dDIFFVar_GPU += cp.fft.irfft2(Ft, s=(NX, NY))
        del Ft
        return PixA_dDIFFVar_GPU


SpaceSFFT_CupyFlow = SpaceSFFT_CupyFlow_LowMem_rfft
