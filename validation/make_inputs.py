#!/usr/bin/env python3
"""Regenerate an ``sfft_inputs_<N>.npz`` from a real OpenUniverse2024 (ou2024)
science+template pair, so the validation harness is reproducible WITHOUT shipping
the >100 MB binary inputs.

This mirrors the phrosty input-prep path (DiaObject -> ImageCollection 'ou2024'
-> Pipeline.sky_sub_all_images + get_psfs), then extracts exactly the arrays the
SpaceSFFT flow consumes and saves them under the keys run_backend.py / compare.py
expect.

The data are loaded by snappl as (row, col) = (y, x); the SFFT flow expects
(x, y), so every image array is transposed and made contiguous. Variance is
noise**2. The header is a minimal WCS header with NAXIS/NAXIS1/NAXIS2 inserted.

All inputs/outputs are command-line arguments -- NO hardcoded machine paths.

Output npz keys / shapes / dtypes (for an N x N frame):
  data_sci    (N, N)   float64   sky-subtracted science image (x, y)
  data_templ  (N, N)   float64   sky-subtracted template image (x, y)
  var_sci     (N, N)   float32   science variance (noise**2)
  var_templ   (N, N)   float32   template variance (noise**2)
  dm_sci      (N, N)   uint8     science detection mask
  dm_templ    (N, N)   uint8     template detection mask
  psf_sci     (P, P)   float64   science PSF stamp (x, y)
  psf_templ   (P, P)   float64   template PSF stamp (x, y)
  skyrms_sci  scalar   float64   science sky rms
  skyrms_templ scalar  float64   template sky rms
  hdr_sci     scalar   <U...     science WCS header string (astropy .tostring())
  hdr_templ   scalar   <U...     template WCS header string

NOTE: real validation must be run at >= 2560**2; generate inputs at the scale
you intend to validate at. This requires phrosty + snappl + the ou2024 data
access already configured (same environment that runs the pipeline).
"""
import argparse

import numpy as np

# Upstream sfft still references the removed numpy alias np.in1d.
if not hasattr(np, "in1d"):
    np.in1d = np.isin


def wcs_header_string(img):
    """Minimal WCS header string with NAXIS keys, matching the SFFT input prep."""
    h = img.image.get_wcs().get_astropy_wcs().to_header(relax=True)
    h.insert(0, ("NAXIS", 2))
    h.insert("NAXIS", ("NAXIS1", img.image.data.shape[1]), after=True)
    h.insert("NAXIS1", ("NAXIS2", img.image.data.shape[0]), after=True)
    return h.tostring()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="phrosty/snappl config yaml (runtime arg, never hardcoded)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--oid", type=int, required=True,
                    help="DiaObject id, e.g. 20172782")
    ap.add_argument("--band", default="Y106", help="filter (default Y106)")
    ap.add_argument("--science-csv", required=True,
                    help="phrosty science instances csv")
    ap.add_argument("--template-csv", required=True,
                    help="phrosty template instances csv")
    ap.add_argument("--collection", default="ou2024",
                    help="ImageCollection name (default ou2024)")
    args = ap.parse_args(argv)

    from snappl.config import Config
    Config.get(args.config, setdefault=True)

    from snappl.diaobject import DiaObject
    from snappl.imagecollection import ImageCollection
    from phrosty.pipeline import Pipeline

    obj = DiaObject.find_objects(collection=args.collection, name=args.oid)[0]
    ic = ImageCollection.get_collection(args.collection)

    pip = Pipeline(obj, ic, args.band,
                   science_csv=args.science_csv, template_csv=args.template_csv,
                   oid=args.oid, nprocs=1, nwrite=1, verbose=False,
                   catchfailures=False, dbsave=False)
    pip.sky_sub_all_images()
    pip.get_psfs()

    ti = pip.template_images[0]
    si = pip.science_images[0]

    cont = np.ascontiguousarray
    np.savez(
        args.out,
        data_sci=cont(si.skysub_img.data.T),
        data_templ=cont(ti.skysub_img.data.T),
        var_sci=cont(si.image.noise.T) ** 2,
        var_templ=cont(ti.image.noise.T) ** 2,
        dm_sci=cont(si.detmask_img.data.T),
        dm_templ=cont(ti.detmask_img.data.T),
        psf_sci=cont(si.psf_data.T),
        psf_templ=cont(ti.psf_data.T),
        skyrms_sci=np.float64(si.skyrms),
        skyrms_templ=np.float64(ti.skyrms),
        hdr_sci=wcs_header_string(si),
        hdr_templ=wcs_header_string(ti),
    )
    print(f"wrote {args.out}  sci_shape={si.skysub_img.data.shape}  "
          f"skyrms_sci={float(si.skyrms):.4g}  skyrms_templ={float(ti.skyrms):.4g}")


if __name__ == "__main__":
    main()
