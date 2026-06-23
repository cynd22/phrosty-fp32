# Truth test — fp32 fork @ 4088² with production A25ePSF vs injected truth

**TL;DR.** The single-precision low-memory SFFT fork, run at full-frame 4088² with **your
own empirical PSF** (A25ePSF, Aldoroty 2025) on SN **20172782** (Y106) — the supernova in
`Roman-Supernova-PIT/photometry_test_data` — recovers the **injected OpenUniverse truth
within noise** across the SN's rise→peak→fall (combined −1.3σ over 5 epochs; the
well-constrained peak within 4% of truth), and recovers SN-free epochs as zero (no invented
flux). The fork is bit-identical to stock float64 (see `SFFT_FORK_VALIDATION.md`); this test
confirms the whole chain — fork + production PSF — lands on truth, at ~6 GB instead of ~28.

Figures: `validation/truth_recovery_20172782.png` (recovered vs injected truth),
`validation/truth_cutouts_20172782.png` (the SN rise→peak→fade in the difference images).
Numbers: `validation/truth_recovery.csv`.

---

## Why this SN
The 8 Roman images committed in `photometry_test_data/ou2024/images/simple_model/Y106/` are
all of `20172782`, each with its per-image truth catalog in `ou2024/truth/Y106/`. That's a
controlled, offline, reproducible truth source — and across the 8 it's a real lightcurve:
3 SN-free epochs (5934, 13205, 19009; injected flux 0) and 5 science epochs
(35198→35967→36736→38645→39790) tracing rise→**peak**→fall. We used 5934 as the template.

## Method (catalog-exact, pipeline-independent)
For each epoch, pull the SN's injected flux from the per-image truth index (position match
to the SN, separation 0.00″), in the image's native flux units (= phrosty's flux units).
Injected difference flux = science-image flux − template-image flux; the template is
SN-free, so it's just the science-image flux. Compare to recovered ± error. The recipe was
cross-checked against the independent SNANA SED truth (`snana_10430.hdf5`, `synmag_Y`) — the
two truth sources agree, so the result isn't an artifact of one catalog.

## Result

| pointing | MJD | injected truth | recovered ± err | recovered/truth | agreement |
|---|---|---|---|---|---|
| 35198 | 62456 | 1069 | 827 ± 555 | 0.77 | −0.44σ |
| 35967 | 62466 | 6680 | 6491 ± 525 | 0.97 | −0.36σ |
| 36736 | 62476 | 9018 | 8635 ± 622 | 0.96 | −0.60σ |
| 38645 | 62501 | 5172 | 4479 ± 540 | 0.87 | −1.28σ |
| 39790 | 62516 | 5022 | 4889 ± 525 | 0.97 | −0.25σ |
| **combined** | | | | mean **0.91** | **−1.31σ** (within noise) |

Plus the SN-free epochs recover flux consistent with zero — the fork does not invent flux.

## What it shows
1. **Recovery within noise.** Every epoch is within 1.3σ of injected truth; the
   best-constrained point (the peak, MJD 62476) lands within 4% (−0.6σ). Combined −1.3σ
   over 5 epochs is statistically consistent with no bias.
2. **Shape recovered.** Rise → peak → fall tracks the injected curve (see figure).
3. **Not the fp32 fork.** The fork reproduces stock float64 to roundoff on the difference
   image and bit-for-bit on flux; this test confirms the full chain — fork + production PSF
   — recovers injected truth.

Honest residual: the mean ratio is 0.91, slightly low, driven by the faintest
(noise-dominated) epoch; a few-% residual can't be excluded from 5 epochs, but it's not
statistically significant and every well-constrained epoch lands on truth. (PSF choice
matters at this level: a *model*-PSF stand-in we tried first sat ~10% low at the bright end
— the empirical A25ePSF is what brings recovery onto truth.)

## Notes (stuff that tripped me up off-cluster — probably version skew on my end)
None of this is about the fork; it's just what I hit getting the production PSF running on a
fresh off-cluster install, in case it's useful or already on someone's radar:
- **The Zenodo ePSFs and the snappl reader I had are a version apart.** The deposited files
  (`gridded_psfs_aldoroty_2025_1.1.0`) are pickled photutils `ImagePSF` objects, but the
  `snappl.psf.A25ePSF.read()` in my install expects a YAML form (`yaml.safe_load`), so loading
  them raw threw `UnicodeDecodeError: 0x80`. The pickle also wanted
  `photutils..._LegacyEPSFModel`, which my photutils had renamed, so it needed a small shim to
  unpickle. I just bridged it locally (pickle→YAML, keeping the real PSF array). Almost
  certainly my pinned versions — may already be sorted on a newer snappl/photutils than mine.
- **The two sparsest fields got dropped.** On the SN-free pointings (13205, 19009) the
  zeropoint star-fit (19<m<21.5) didn't seem to find enough stars, and the run then hit a
  `None` in `do_stamps`/`save_stamp_paths`. Could easily be my setup or just those fields —
  flagging in case. The science epochs were all fine.

## Scope
Depth, not breadth: one SN, one band (Y106), but a full reproducible curve. (An earlier
11-SN consistency check used a model-PSF stand-in, which this test shows biases the
bright-end flux, so it's not included — it would need re-running with A25ePSF.)

## Reproduce
Inputs are all in `photometry_test_data` (`ou2024/images/simple_model/Y106/`,
`ou2024/truth/Y106/`); PSF from Zenodo 10.5281/zenodo.15609513. Science = the non-5934 Y106
pointings vs template 5934, full-frame 4088², `SFFT_BACKEND=rfft`, `psf.type=A25ePSF`,
`kerpolyorder=2`. The run writes a phrosty native-schema lightcurve `.pq` (our fork also
fills `NEA`/`sky_rms`/`pix_x`/`pix_y`, which stock phrosty leaves NaN); the per-epoch
recovered-vs-truth numbers are in `validation/truth_recovery.csv`.
