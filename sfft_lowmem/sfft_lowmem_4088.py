"""
sfft_lowmem_4088.py  --  PHASE-A entry point.

Subclasses the working phase-2 flow (SpaceSFFT_CupyFlow_LowMem) but routes
the subtraction core through the plane-by-plane 4088 core
(GeneralSFFTSubtract_PureCupy_F32_4088) which removes the full FOMG complex
stack from the VRAM peak.  Everything else (resampling f64 carve-out,
decorrelation/score/variance f64 post-processing) is inherited unchanged.
"""
import cupy as cp

import sfft_lowmem as _base
from sfft_lowmem import SpaceSFFT_CupyFlow_LowMem
from sfft_lowmem_core_4088 import GeneralSFFTSubtract_PureCupy_F32_4088
from sfft_lowmem_core import SingleSFFTConfigure_Cupy_F32

_FLT = cp.float32


class PureCupy_Customized_Packet_F32_4088:
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

        Solution_GPU, PixA_DIFF_GPU, _ = GeneralSFFTSubtract_PureCupy_F32_4088.GSS(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU,
            PixA_mI_GPU=PixA_mI_GPU, PixA_mJ_GPU=PixA_mJ_GPU,
            SFFTConfig=SFFTConfig, ContamMask_I_GPU=None, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if NaNmask_GPU is not None:
            PixA_DIFF_GPU[NaNmask_GPU] = cp.nan
        if ConvdSide == 'SCI':
            PixA_DIFF_GPU *= -1.
        return Solution_GPU, PixA_DIFF_GPU


class SpaceSFFT_CupyFlow_LowMem_4088(SpaceSFFT_CupyFlow_LowMem):
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

        self.Solution_GPU, self.PixA_DIFF_GPU = PureCupy_Customized_Packet_F32_4088.PCCP(
            PixA_REF_GPU=PixA_REF_GPU, PixA_SCI_GPU=PixA_SCI_GPU,
            PixA_mREF_GPU=PixA_mREF_GPU, PixA_mSCI_GPU=PixA_mSCI_GPU,
            ForceConv='REF' if self.sci_is_target else 'SCI',
            GKerHW=self.GKerHW, KerPolyOrder=self.KerPolyOrder,
            BGPolyOrder=self.BGPolyOrder, ConstPhotRatio=self.ConstPhotRatio,
            CUDA_DEVICE_4SUBTRACT=self.CUDA_DEVICE_4SUBTRACT)
        self.PixA_DIFF_GPU[self.BlankMask_GPU] = 0.

        del PixA_mCtarget_GPU
        del PixA_mCresamp_object_GPU


SpaceSFFT_CupyFlow = SpaceSFFT_CupyFlow_LowMem_4088
