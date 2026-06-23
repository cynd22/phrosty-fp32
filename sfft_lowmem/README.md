# sfft-lowmem-fp32

**A single-precision, low-memory fork of the SFFT GPU image-subtraction engine — runs
full-frame 4088² Roman difference imaging on an 8 GB consumer GPU, numerically faithful to
the float64 reference.**

Roman supernova difference imaging + forced photometry (via
[phrosty](https://github.com/Roman-Supernova-PIT/phrosty)) is effectively pinned to NERSC
A100s: the stock float64 SFFT path needs **~28 GB** for a full 4088² Roman SCA. This fork
reorganizes the SFFT GPU flow to do the same subtraction in **~6 GB** — without cropping,
and without losing the float64-grade precision the pipeline exists to provide — so the work
can run off-cluster on hardware people already own (anything with ≥8 GB NVIDIA VRAM).

The intended use is a **cheap triage tier**: explore candidates at home at full resolution,
confirm the survivors at full precision on the cluster.

## Result, in one line
On the SN PIT's own test supernova (`20172782`), full-frame 4088², recovered fluxes match
**injected OpenUniverse truth within noise** (combined −1.3σ over the rise→peak→fall), using
the published **A25ePSF** (Aldoroty 2025). The engine is **bit-identical to stock float64**
on the difference image; the memory savings come from precision layout + a low-memory
rewrite, not from changing the answer. See `docs/SFFT_FORK_VALIDATION.md` and
`docs/TRUTH_TEST.md` for the numbers.

## How it works (what actually changed)
- **Precision layout, not blanket f32.** The ill-conditioned PSF-matching solve and the
  solve-feeding forward DFTs stay in **complex128** (`SFFT_FFT_PRECISION=c128` default — a
  c64 version of these was the one real bug, caught and fixed); the subtract pass runs c64;
  the linear solve, resampler, decorrelation, score and variance stay f64.
- **Low-memory rewrite of `SpaceSFFTCupyFlow`**: half-spectrum **rfft** with conjugate
  symmetry computed on the fly, a **row-banded** half-Kab `Construct_FDIFF` (NBAND=4 at
  N≥3500), and a **chunked** OMG matrix fill — so the 4088² intermediates never materialize
  at full f64 size.
- Net: full-frame 4088² peaks at **~6 GB** (vs ~28 GB f64), fits a clean 8 GB card.

## Use
Drop-in replacement for `sfft.SpaceSFFTCupyFlow.SpaceSFFT_CupyFlow`. Put this
directory (`sfft_lowmem/`) on `PYTHONPATH`, route phrosty with `SFFT_BACKEND=rfft`,
run. Full steps: **`INTEGRATION.md`**.

## Precision contract & caveats
- **Validated == float64**: decorrelated difference matches stock f64 to <1e-4 of sky;
  forced flux to <3e-4 of a measurement error bar; full adversarial pass on the intermediate
  quantities through the 4088² banded path. Details: **`docs/SFFT_FORK_VALIDATION.md`**.
- **Requires sky-subtracted input** (the c64 subtract pass is pedestal-sensitive).
- **Precision measurement, not detection** — this is the float64-faithful measurement stage;
  detection is a separate package.
- Validate per-process on the *sensitive* quantity (decorrelated difference), never on
  lightcurve flux (blind to background) or fork-vs-fork (circular).
- `KerPolyOrder ≥ 5` overflows a fixed internal array (phrosty uses 2; safe).

## Built on / credit
This fork stands entirely on others' work:
- **SFFT** — Lei Hu (`thomasvrussell`), the image-subtraction algorithm and the
  `SpaceSFFTCupyFlow` this forks. MIT. https://github.com/thomasvrussell/sfft
- **phrosty** & **snappl** — the Roman SN PIT photometry pipeline this plugs into. BSD-3.
  https://github.com/Roman-Supernova-PIT
- **A25ePSF** — Lauren Aldoroty et al. 2025, the empirical Roman PSF used for the truth
  validation. Zenodo [10.5281/zenodo.15609513](https://doi.org/10.5281/zenodo.15609513).
- **OpenUniverse 2024** simulations — the test images and injected truth.

The original contribution here is the **single-precision / low-memory re-engineering** of
the GPU flow (incl. a CUDA-level c128→c64 rework of the SFFT core) and its validation. Parts
of that rework were AI-assisted; the kernels in particular warrant a careful human read
before production use.

**If this is useful to you — especially if it's merged or adapted upstream — a credit or
citation back to this repo (https://github.com/cynd22/phrosty-fp32) is appreciated.**

## License
MIT (see `LICENSE`), retaining SFFT's original MIT copyright. Author: cynd22.
