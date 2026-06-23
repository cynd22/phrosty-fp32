#!/usr/bin/env python3
"""Run the decorrelated-difference SFFT flow for one backend on a fixed .npz input.

Loads the input arrays, constructs the chosen SpaceSFFT flow, runs
resample -> cross-convolve -> subtract -> find/apply decorrelation, and writes
the decorrelated difference image to ``decorr_<backend>.npy`` in the output dir.

Everything is supplied via command-line arguments -- there are NO hardcoded
machine paths. Run stock and each fork backend with byte-identical input, then
hand the resulting .npy files to compare.py.

Backends
--------
  stock       : upstream  sfft.SpaceSFFTCupyFlow.SpaceSFFT_CupyFlow
  lowmem      : fork  sfft_lowmem.SpaceSFFT_CupyFlow_LowMem
  lowmem4088  : fork  sfft_lowmem_4088.SpaceSFFT_CupyFlow_LowMem_4088
  rfft        : fork  sfft_lowmem_rfft.SpaceSFFT_CupyFlow_LowMem_rfft

The fork backends live in the sibling ``sfft_lowmem/`` directory of this repo;
that directory is added to sys.path automatically (it has no __init__.py and is
imported as top-level modules). You may instead set PYTHONPATH to point there.

REMINDER: real validation must be run at >= 2560**2. 1024**2 is harness-smoke
only and is NOT a valid validation scale.
"""
import argparse
import os
import sys

import numpy as np

# Upstream sfft still references the removed numpy alias np.in1d; restore it.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# Repo-relative location of the fork engine (sibling dir of this file's parent).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FORK_DIR = os.path.join(_REPO_ROOT, "sfft_lowmem")

BACKENDS = ("stock", "lowmem", "lowmem4088", "rfft")


def _import_flow(backend):
    """Return the flow class for the requested backend."""
    if backend == "stock":
        from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow as Flow
        return Flow
    # Fork backends: make the engine modules importable as top-level names.
    if _FORK_DIR not in sys.path:
        sys.path.insert(0, _FORK_DIR)
    if backend == "lowmem":
        from sfft_lowmem import SpaceSFFT_CupyFlow_LowMem as Flow
    elif backend == "lowmem4088":
        from sfft_lowmem_4088 import SpaceSFFT_CupyFlow_LowMem_4088 as Flow
    elif backend == "rfft":
        from sfft_lowmem_rfft import SpaceSFFT_CupyFlow_LowMem_rfft as Flow
    else:
        raise ValueError(f"unknown backend: {backend!r}")
    return Flow


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=BACKENDS,
                    help="which SFFT flow to run")
    ap.add_argument("--npz", required=True,
                    help="input .npz (see make_inputs.py / README for keys)")
    ap.add_argument("--config", required=True,
                    help="phrosty/snappl config yaml path (runtime arg, never hardcoded)")
    ap.add_argument("--outdir", required=True,
                    help="directory to write decorr_<backend>.npy into")
    ap.add_argument("--managed", action="store_true",
                    help="use the cupy managed-memory allocator (lets large frames "
                         "oversubscribe GPU memory)")
    ap.add_argument("--kerpolyorder", type=int, default=2,
                    help="kernel polynomial order (default 2, matches phrosty)")
    args = ap.parse_args(argv)

    import cupy as cp
    if args.managed:
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    # Config must be set up before the flow / snappl machinery is imported.
    from snappl.config import Config
    Config.get(args.config, setdefault=True)

    from astropy.io import fits

    Flow = _import_flow(args.backend)

    z = np.load(args.npz, allow_pickle=True)
    g = lambda k: cp.array(z[k], dtype=cp.float64)
    hdr_target = fits.Header.fromstring(str(z["hdr_sci"]))
    hdr_object = fits.Header.fromstring(str(z["hdr_templ"]))

    sf = Flow(
        hdr_target=hdr_target, hdr_object=hdr_object,
        target_skyrms=float(z["skyrms_sci"]), object_skyrms=float(z["skyrms_templ"]),
        PixA_target_GPU=g("data_sci"), PixA_object_GPU=g("data_templ"),
        PixA_targetVar_GPU=g("var_sci"), PixA_objectVar_GPU=g("var_templ"),
        PixA_target_DMASK_GPU=g("dm_sci"), PixA_object_DMASK_GPU=g("dm_templ"),
        PSF_target_GPU=g("psf_sci"), PSF_object_GPU=g("psf_templ"),
        KerPolyOrder=args.kerpolyorder,
    )
    sf.resampling_image_mask_psf()
    sf.cross_convolution()
    sf.sfft_subtraction()
    sf.find_decorrelation()
    dec = sf.apply_decorrelation(sf.PixA_DIFF_GPU)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f"decorr_{args.backend}.npy")
    np.save(outpath, cp.asnumpy(dec).astype(np.float64))

    pool = cp.get_default_memory_pool()
    print(f"{args.backend}: wrote {outpath}  shape={dec.shape}  "
          f"peak_MB={pool.total_bytes() / 1e6:.0f}")


if __name__ == "__main__":
    main()
