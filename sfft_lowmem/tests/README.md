# Fork test tiers

Fast, automated tests for the fp32/low-memory engine. They sit UNDER the
standing rule, not instead of it.

| Tier | File | Needs | When to run | What it protects |
|------|------|-------|-------------|------------------|
| 0 | `test_guards.py` | CPU only | every change, CI | the guard rails: `FIloc[16]`/`Fij>16` config guards, fail-loud `SFFT_FFT_PRECISION`, `SFFT_BACKEND` routing (fork stays opt-in; upstream default is stock **float64**) |
| 0 | `test_compare_criteria.py` | CPU only | every change, CI | `validation/compare.py` itself — the judge of the standing rule must fail the failure modes it exists to catch |
| 1 | `test_engine_gpu.py` | CUDA GPU, ~1 min | before/after any engine change, locally | the mechanism claims: chunked-OMG **bit-exactness**, NBAND banding invariance, the rfft Hermitian identity, and small-N stock-vs-fork Solution/raw-DIFF agreement |

Run everything:

```sh
/path/to/phrosty-venv/bin/python -m pytest sfft_lowmem/tests -v
```

CPU-only (CI without GPU): set `SKIP_GPU_TESTS=1` (same convention as
upstream phrosty) or let the conftest skip on missing CUDA device.

## Why these exist (and what they can NOT do)

The c64 bug documented in `docs/SFFT_FORK_VALIDATION.md` passed every
small-size check and only manifested at ≥ 2560² on real data. **No test in
this directory can catch that class of bug.** Tier 0/1 catch *wiring*
regressions — a refactor that breaks the chunk indexing, an env typo that
silently flips precision, a banding change that alters arithmetic — within
seconds instead of during a multi-hour validation session.

Any change to the core numerics (`sfft_lowmem_core*.py`) still requires the
full stock-vs-fork standing-rule run at ≥ 2560² on real Roman input via
`validation/` (see `validation/README.md`).
