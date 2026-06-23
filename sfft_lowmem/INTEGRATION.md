# Integrating the fork with phrosty

The fork is a drop-in replacement for `sfft.SpaceSFFTCupyFlow.SpaceSFFT_CupyFlow`.
Two small things: put the engine on your path, and route phrosty to it.

## 1. Put the engine on the import path
The six engine files at the top level of this directory are an inheritance chain
(`sfft_lowmem_rfft` → `sfft_lowmem_4088` → `sfft_lowmem` → `sfft_lowmem_core*`), so
they must all sit in one directory on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/phrosty-fp32/sfft_lowmem:$PYTHONPATH
```

## 2. Route phrosty's pipeline to the fork
In `phrosty/pipeline.py`, replace the single import line

```python
from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow
```

with a backend switch (default stays stock float64; `SFFT_BACKEND=rfft` selects the fork):

```python
import os as _os
if _os.environ.get('SFFT_BACKEND', '').lower() == 'rfft':
    from sfft_lowmem_rfft import SpaceSFFT_CupyFlow_LowMem_rfft as SpaceSFFT_CupyFlow
else:
    from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow
```

## 3. (Optional) populate NEA / sky_rms / pix_x / pix_y
Stock phrosty declares these output columns but leaves them NaN. In
`make_phot_info_dict`, right after `results_dict['zpt'] = sci_image.image.zeropoint`:

```python
# pix_x/pix_y are the SN position on the detector (4088^2 SCA), so project the
#   sky coord through the science-image WCS -- NOT `pxcoords`, which is stamp-local
#   (~50,50 on the 100x100 diff cutout) and only valid for phot_at_coords.
det_x, det_y = sci_image.image.get_wcs().world_to_pixel( self.diaobj.ra, self.diaobj.dec )
results_dict['pix_x']   = float( det_x )
results_dict['pix_y']   = float( det_y )
results_dict['sky_rms'] = sci_image.skyrms
results_dict['NEA']     = float( psf_img.data.sum()**2 / np.sum( psf_img.data**2 ) )
```

These changes are already applied in this fork's `phrosty/pipeline.py` (see
`make_phot_info_dict` and `_resolve_sfft_backend`); the snippet above is for porting
them onto a fresh phrosty checkout by hand.

## 4. Run
```bash
SFFT_BACKEND=rfft  python your_phrosty_driver.py   # full-frame 4088² fits in ~6 GB
```
Requires **sky-subtracted input** (the c64 subtract pass is pedestal-sensitive). See
`README.md` for the precision contract and `docs/SFFT_FORK_VALIDATION.md` for the proof.
