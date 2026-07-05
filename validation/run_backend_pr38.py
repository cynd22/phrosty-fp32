#!/usr/bin/env python3
"""Drive the sfft PR #38 unified flow (SpaceSFFT_Flow) on a validation .npz.

Companion to run_backend.py, for measuring upstream PR #38's Cupy/Numpy
backend parity with the same criteria as the fork's standing rule
(compare.py). Produces ``decorr_pr38_<backend>.npy`` in the output dir.

Usage:
  PYTHONPATH=/path/to/sfft-pr38-checkout $PY run_backend_pr38.py \
      --backend cupy|numpy --npz IN.npz --outdir OUT [--managed] \
      [--kerpolyorder 2] [--cpu-threads 12]

The PR #38 checkout must SHADOW the installed sfft on PYTHONPATH (this
script refuses to run against the installed release, which has no
SpaceSFFTFlow module). Do NOT install the PR branch into the working venv:
it deletes SpaceSFFTCupyFlow and breaks phrosty / the fork / run_backend.py.

Comparison baseline: run_backend.py --backend stock (release 1.6.4 Cupy
flow) on the same .npz, then compare.py against each PR-38 backend.

Notes on the PR #38 API mapping (vs the release flow):
  * class SpaceSFFT_Flow, constructor drops the _GPU suffixes and takes
    numpy arrays + BACKEND_4SUBTRACT switch;
  * our .npz arrays are already transposed to (x, y) -> transpose=False;
  * methods renamed: resample_image_mask_psf / cross_convolve /
    sfft_subtract / find_decorrelation;
  * apply_decorrelation returns a NUMPY array (transposed only if
    transpose=True, so untransposed here -- same orientation as the npz).
"""
import argparse
import os

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=("cupy", "numpy"),
                    help="which PR-38 backend to run")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--managed", action="store_true",
                    help="cupy managed-memory allocator (cupy backend only)")
    ap.add_argument("--kerpolyorder", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=os.cpu_count() or 8,
                    help="NUM_CPU_THREADS_4SUBTRACT for the numpy backend")
    args = ap.parse_args(argv)

    from sfft import SpaceSFFTFlow  # noqa: F401  (fails on release sfft: wrong PYTHONPATH)
    from sfft.SpaceSFFTFlow import SpaceSFFT_Flow

    if args.backend == "cupy" and args.managed:
        import cupy as cp
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    from astropy.io import fits

    z = np.load(args.npz, allow_pickle=True)
    hdr_target = fits.Header.fromstring(str(z["hdr_sci"]))
    hdr_object = fits.Header.fromstring(str(z["hdr_templ"]))
    f64 = lambda k: np.ascontiguousarray(z[k], dtype=np.float64)

    sf = SpaceSFFT_Flow(
        hdr_target=hdr_target, hdr_object=hdr_object,
        target_skyrms=float(z["skyrms_sci"]), object_skyrms=float(z["skyrms_templ"]),
        PixA_target=f64("data_sci"), PixA_object=f64("data_templ"),
        PixA_targetVar=f64("var_sci"), PixA_objectVar=f64("var_templ"),
        PixA_target_DMASK=f64("dm_sci"), PixA_object_DMASK=f64("dm_templ"),
        PSF_target=f64("psf_sci"), PSF_object=f64("psf_templ"),
        transpose=False,                      # npz arrays are already (x, y)
        KerPolyOrder=args.kerpolyorder,
        BACKEND_4SUBTRACT="Cupy" if args.backend == "cupy" else "Numpy",
        NUM_CPU_THREADS_4SUBTRACT=args.cpu_threads,
    )
    sf.resample_image_mask_psf()
    sf.cross_convolve()
    sf.sfft_subtract()
    sf.find_decorrelation()
    dec = sf.apply_decorrelation(sf.PixA_DIFF)   # numpy, untransposed (transpose=False)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f"decorr_pr38_{args.backend}.npy")
    np.save(outpath, np.asarray(dec, dtype=np.float64))
    print(f"pr38-{args.backend}: wrote {outpath}  shape={np.asarray(dec).shape}")


if __name__ == "__main__":
    main()
