# SFFT single-precision / low-memory fork — postmortem & validation report

**Date:** 2026-06-21
**Hardware:** RTX 2070 SUPER, 8 GB (~6.5 GB free)
**Reference of truth:** STOCK float64 SFFT (Lei Hu's published `sfft` 1.6.4), never modified.
**Fork files (fixed):** `src/sfft_lowmem_core.py`, `src/sfft_lowmem_core_4088.py`,
`src/sfft_lowmem_core_rfft.py` (plus the thin `src/sfft_lowmem*.py` wrappers).

---

## TL;DR

The fork was MEMORY-correct but NUMERICALLY BROKEN at large image sizes. It passed
at 1024² and silently produced garbage backgrounds at 2560²/4088². The root cause
was running the **solve-feeding forward DFTs in complex64**; their ~1e-7 per-bin
error is amplified by the ill-conditioned PSF-matching solve at large N, poisoning
the linear system and producing structured difference-image noise. The fix
defaults those forward DFTs (and the linear-solve feed) to **complex128**, with a
memory refactor (conj-on-the-fly) to keep 4088² inside 8 GB. After the fix, the
decorrelated difference matches stock float64 to < 1e-4 of sky at 1024² and 2560²
(directly stock-referenced) and is inferred-correct at 4088² (no stock baseline
was feasible on this card).

---

## (a) The bug — size-dependent f32-core error

Proven by an airtight per-process comparison of the **decorrelated difference
image** against STOCK float64 on byte-identical real Roman inputs. Metric region =
full frame minus a central 400 px box (i.e. background/sky); metrics = background
rms and lag-1 spatial autocorrelation (ACF: ~0 = white noise, negative = high-freq
structured artifact). sky_rms ≈ 13.3 in all cases.

### ORIGINAL (broken, complex64) numbers

| size | STOCK f64 (truth) | FORK f32 (broken) | verdict |
|------|-------------------|-------------------|---------|
| 1024² | rms 26.0, ACF +0.06 | rms 26.9, ACF +0.03 | MATCHES (diffs only at bright source cores = benign f32 rounding) |
| 2560² | rms 59, ACF +0.25 | rms 2135, ACF −0.23 | **BROKEN**: ~160× sky-rms discrepancy, structured |
| 4088² | (no stock baseline feasible) | negative ACF, elevated structured noise | **BROKEN** (same signature as 2560²) |

- 1024 = 2¹⁰ **passes**; 2560 = 2⁹·5 and 4088 = 2³·511 **fail** → size-dependent.
- BOTH non-rfft (`lowmem4088`) and rfft forks fail **identically** (agree with each
  other to 2e-4 of sky) → the bug lives in the SHARED f32 CORE (the lowmem
  lineage), NOT the rfft layer.

---

## (b) Why earlier checks missed it

1. **Lightcurve forced-photometry insensitivity.** The earlier "validation" leaned
   on a 1024² light curve that "matched" stock. Forced photometry sits ON the SN
   peak — it is dominated by the bright source core and is essentially insensitive
   to background structure. A broken background can hide entirely under a
   peak-photometry check. **Lightcurve flux is NOT a valid validator for this bug.**

2. **Fork-vs-fork self-validation.** A prior self-reported "4.6e-4" agreement never
   survived an independent stock-vs-fork check at ≥ 2560². Because the two forks
   share the broken core, they agree with *each other* while both diverge from
   stock. **Fork-vs-fork agreement is NOT a valid validator.**

3. **Only tested at 1024².** 1024 = 2¹⁰ happens to be the one size where the
   conditioning stayed benign. Testing a single power-of-two size masked the
   N-dependence.

**Standing rule (below) encodes the fix for all three mistakes.**

---

## (c) Root cause & the fix (file:line, before → after)

**Root cause.** The forward DFTs that feed the PSF-matching linear solve
(`LHMAT`/`RHb`) ran in complex64. A 2560² forward DFT stored at c64 caps accuracy
at ~1e-7 (max ~4e-5 relative). At large N the matching solve is ill-conditioned, so
that ~1e-7 perturbation is amplified into the LHMAT/RHb system and emerges as
structured background noise in the difference. A latent secondary landmine was a
hardcoded `SCALE=1/256²` (correct only for N=256) in an OTF kernel.

### `sfft_lowmem_core_4088.py`
- **line 58** — `SFFT_FFT_PRECISION` default `'c64'` → **`'c128'`**
  (`_PCPX`/`_SCPX` plumbing already present; one-literal flip).

### `sfft_lowmem_core_rfft.py`
- **line 50** — `SFFT_FFT_PRECISION` default `'c64'` → **`'c128'`**.
- **line 127** (`_FDIFF_OTF_SRC`) — `const double SCALE=1.52587890625e-05;`
  (hardcoded 1/256²) → **`const double SCALE=1.0/((double)N0*(double)N1);`**.
  (Kernel is compiled-but-unused; preventive.)
- **Memory refactor (required):** the blanket c128 default doubled the persistent
  forward stacks and OOM'd 4088. Eliminated materialized conjugate stacks
  (`SPixA_CFIij_GPU`, `SPixA_CFTpq_GPU`, `PixA_CFJ_GPU`); `build_pre` gained
  `conjB=True` to conjugate each source plane on the fly; OMG/THE/DEL loops
  conjugate inline; `del SPixA_FIij_GPU` after THE. Saved ~1.4 GB at 4088
  (7542 → 6104 MB peak). Chunking/banding/rfft levers preserved.

### `sfft_lowmem_core.py` (base `lowmem` core — previously had no toggle; line 172 hardwired c64)
- **top** — added `import os`.
- **after line 173** — added
  `_PCPX = np.complex128 if os.environ.get('SFFT_FFT_PRECISION','c128')=='c128' else np.complex64`.
- **forward-DFT block (~line 297)** — added
  `_SCPX = _PCPX if SFFTSolution_GPU is None else _CPX`; routed `PixA_FJ_GPU`,
  `SPixA_FIij_GPU`, `SPixA_FTpq_GPU` and the six Hadamard stacks from `_CPX` to
  `_SCPX`. Subtract-pass twiddle / `Construct_FDIFF` (~lines 449-463) stay `_CPX`
  (proven harmless).
- **Hadamard-product fix** — stock `HadProd_*` CUDA kernels are compiled
  `cuFloatComplex` and emit NaNs when fed c128 (confirmed via lu_factor "array must
  not contain infs or NaNs"). Replaced the six kernel calls with the verified
  elementwise identity `Hp[k] = A[SREF[k,0]] * B[SREF[k,1]]` via cupy
  `_hadprod`/`_hadprod_J` helpers (dtype-correct at c64 AND c128, bit-equivalent to
  the kernel at c64). PSI operand order matched the kernel BODY
  (`FTpq[col0] * conj(FIij)[col1]`), not its argument order.

**Toggle:** `SFFT_FFT_PRECISION` env var. Default is now `c128` (correct). Setting
`c64` reproduces the original break (used below as a controlled A/B).

---

## (d) Corrected full validation table

Decorrelated difference vs STOCK float64, byte-identical real Roman inputs
(`/tmp/sfft_inputs_1024.npz`, `/tmp/sfft_inputs.npz`), per-process. Stock f64 is the
reference (managed allocator at 2560²). sky_rms ≈ 13.3. Fork shown = rfft;
`low4088` and base `lowmem` match it to < 1e-5.

| size | stock bkg-rms / ACF | fork bkg-rms / ACF | pairwise(fork−stock)/sky | max-diff/sky | reference | PASS/FAIL |
|------|---------------------|--------------------|--------------------------|--------------|-----------|-----------|
| 1024² | 22.6 / +0.0839 | 22.6 / +0.0839 | 4.3e-05 | 3.7e-3 | stock f64, real data | **PASS** |
| 2560² | 59.0 / +0.2099 | 59.0 / +0.2099 | 9.4e-05 | 1.0e-2 | stock f64, real data | **PASS** |
| 4088² | 73.4 / +0.1925 | 73.4 / +0.1925 | 6.7e-05 | 1.3e-2 | **stock f64, real data** | **PASS** |

**4088² UPDATE (post-session, measured):** the stock-float64 baseline at 4088² was
produced — stock f64 (27 GB) paged through 64 GB host RAM via CUDA managed memory and
completed. Fork c128 vs stock f64 on byte-identical real 4088² Roman input:
bkg-rms **73.415 == 73.415** (ratio 1.0000), lag-1 ACF **+0.1925 == +0.1925**, pairwise
**6.7e-5 of sky** (max 1.3e-2 at source cores = benign f32 rounding). 4088² is now
**directly stock-referenced PASS**, no longer inferred. (Harness: /tmp/x_extract_4088.py,
/tmp/x_run_4088.py, /tmp/cmp4088.py; inputs /tmp/sfft_inputs_4088.npz.)

All three cores (rfft, low4088, base lowmem) match stock at 1024² and 2560²:
ratio 1.0000, ACF identical to 4 decimals, pairwise < 1e-4 of sky. Max-diff pixels
(≈ 1e-2 of sky) are confined to bright source cores = benign f32 rounding, reported
separately and acceptable.

**Controlled A/B (proves the toggle IS the fix):**
- Real-data RAW SFFT diff at 2560² (isolates the f32 core, no decorrelation):
  c128(fixed) std 21.7 == stock 21.7, pairwise 1.1e-5/sky, ACF +0.6811 == stock.
  c64(broken) std 248.9, pairwise **18.6× sky**.
- 2560² decorrelated, c64 reproduced the original break: ratio 36.2×, pairwise 160× sky.
- 4088² A/B (fork-only, synthetic well-conditioned scene): c64 raw std 4.658 →
  c128 raw std 1.717 (the white ~1.7 level seen across 1024–3584); rfft vs low4088
  agree to 4e-2 at 4088.

**Memory:** rfft 4088² managed peak 6104 MB < 6516 MB free, status ok.

---

## (e) Which sizes are trustworthy — and the standing rule

**Trustworthy (directly stock-referenced on real Roman data):**
- **1024²** — PASS.
- **2560²** — PASS. (rfft / low4088 run under the default allocator; base `lowmem`
  needs `ALLOC=managed` — c128 doubled its monolithic stacks.)

- **4088² (full Roman SCA)** — PASS, **now directly stock-referenced** (measured
  post-session: stock f64 paged through 64 GB RAM via managed memory; fork c128 matches
  it to ratio 1.0000, ACF exact, pairwise 6.7e-5/sky). No longer inferred. (rfft/low4088
  run 4088² under `ALLOC=managed`, ~6.1 GB peak.)

**Could not be exercised this session (environment limitation, not a fork issue):**
- Full end-to-end phrosty pipeline — the environment has **no SExtractor binary**
  (`sex`/`source-extractor` not found); `sky_sub_all_images()` fails before reaching
  SFFT. The SFFT+decorrelation portion (the only part the fork touches) was
  validated via the per-process harness on real post-sky-sub npz inputs.
- Synthetic stock-vs-fork ladder — INCONCLUSIVE and deliberately NOT used: the
  idealized scene makes `find_decorrelation` explode (~1e11–1e13) for stock float64
  itself, so the reference is garbage. Real Roman data is authoritative.

### STANDING RULE (mandatory for any future change to the fork)

> Any change to a fork core (`sfft_lowmem_core*.py`) or its wrappers MUST re-run the
> per-process **stock-vs-fork** check on the **decorrelated difference image** at
> **≥ 2560²** on **byte-identical real Roman input**, and pass: (1) background rms
> within ~5 % of stock, (2) lag-1 ACF matching stock and ~white, (3) background
> pairwise rms(fork−stock) ≲ a few % of sky_rms. Bright-source-core f32 rounding is
> reported separately and acceptable.
>
> NEVER validate with lightcurve / forced-photometry flux. NEVER validate
> fork-vs-fork. NEVER conclude from 1024² alone. Harness:
> `/tmp/x_run*.py` + `/tmp/x_cmp.py`; inputs `/tmp/sfft_inputs*.npz`.

---

## (f) Adversarial caveats (2026-06-21 refutation pass)

A dedicated adversarial pass tried to REFUTE the equivalence claim by probing the
quantities/regions/sizes/conditioning the table above never touched (Solution
coeffs, undecorrelated DIFF, variance, score, edge/blank regions; real-data crops
at N=1153/2050/3070; DC-pedestal and PSF-mismatch conditioning). The core claim
held — Solution / raw DIFF / variance / score / FKDECO all match stock to ≤1e-4 of
scale at every even size tested, the Solution error is unstructured f32 rounding
(not the c64 amplification signature), and the edge/no-overlap/blank regions are
bit-identical. But three caveats were surfaced and must be recorded:

1. **SCORE image is non-reproducible at ODD N — shared STOCK instability, not the
   fork.** At odd N (tested 1153) `SkyLevel_Estimator.SLE` fails ("Outlier rejection
   left too few sky elements") and normalizes the score in OPPOSITE directions for
   stock vs fork (stock std 1.3e8, fork std 2.6, corr −1.0). With SLE bypassed the
   UN-normalized scores match to corr +1.0 / rms 1.6e-5, so the fork's score
   *computation* is faithful; the instability lives in shared stock code and is not
   reproducible for STOCK either at odd N. **The claim's target (4088, EVEN) is
   unaffected — even sizes give score corr +1.0.** Anyone running SFFT at odd image
   dimensions should not trust the SLE-normalized score from either backend.

2. **The c64 subtract pass is harmless ONLY for sky-subtracted input.** Its
   fork-vs-stock error in the undecorrelated DIFF scales ~LINEARLY with any DC sky
   pedestal: 1.1e-5/sky at pedestal 0 (real operating point) → 8.9e-5 at 5000 →
   9.8e-4 at 50000. This is the c64 ~1e-7 relative floor on a pedestal-dominated
   spectrum. The "requires sky-subtracted input" caveat is therefore load-bearing,
   not cosmetic — a non-zero pedestal degrades the difference predictably.

3. **LATENT out-of-scope landmine: banded subtract kernel `FIloc[16]`.** The rfft
   half-Kab accumulation kernel (`sfft_lowmem_core_rfft.py`, `_FDIFF_ACC_SRC`) uses a
   fixed-size `cuFloatComplex FIloc[16]` stack array indexed by Fij. KerPolyOrder=2
   (phrosty default) → Fij=6 (safe). KerPolyOrder≥5 → Fij=21 > 16 → stack overflow /
   silent difference corruption. The claim is safe at KerPolyOrder=2, but any future
   bump to KerPolyOrder≥5 would silently corrupt the difference. FLAGGED for future
   changes.

Note: the banded-Kab subtract (NBAND=4) only triggers at N0≥3500, i.e. 4088 only;
the 2560/3070 adversarial crops ran NBAND=1, so the 4088 stock-vs-fork run remains
the decisive banded-path test (prior session: rawdiff pairwise 6.7e-5/sky).

Artifacts: adversarial harness + findings retained locally; not distributed with the repo.

---

## Harness & artifacts

- Inputs: `/tmp/sfft_inputs.npz` (2560²), `/tmp/sfft_inputs_1024.npz` (1024²),
  produced by `/tmp/x_extract.py` / `/tmp/x_extract_1024.py`.
- Runners: `/tmp/x_run.py`, `/tmp/x_run2.py`, `/tmp/x_run_raw.py`; compare
  `/tmp/x_cmp.py`; synthetic `/tmp/x_synth2.py`.
- Decorr arrays: `/tmp/decorr_{stock,rfft,low4088,lowmem}_{1024,2560}.npy`.
