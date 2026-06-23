# SFFT fork-core validation harness

This harness validates that changes to the fp32 fork's SFFT engine
(`sfft_lowmem/sfft_lowmem_core*.py` and the flows built on them) do not alter
the **background** of the decorrelated **difference image** relative to the
upstream (stock) SFFT flow, on byte-identical real Roman input.

## STANDING RULE (verbatim)

Any change to `sfft_lowmem_core*.py` must re-run a per-process STOCK-vs-FORK
check on the decorrelated DIFFERENCE IMAGE at **>= 2560²** on **byte-identical
real Roman input**, passing:

1. background rms within ~5% of stock;
2. lag-1 ACF **matching stock** (the fork adds no excess correlation) — absolute
   whiteness is reported for info only; the decorrelated diff need not be white
   (stock f64 isn't either at some scales), so it must not fail a faithful fork;
3. background pairwise rms(fork − stock) ≲ a few % of `sky_rms`.

Bright-source-core f32 rounding is reported **SEPARATELY** and is **acceptable**.

NEVERs:

- **NEVER** validate via lightcurve / forced-photometry flux.
- **NEVER** validate fork-vs-fork (always fork-vs-stock).
- **NEVER** conclude from 1024² alone — 1024² is harness-smoke only.

The bright-source core (central box) is excluded from criteria (1)–(3) and
reported separately, because f32 rounding in bright cores is expected and must
not contaminate the background verdict.

## How to run

Use the same Python environment that runs phrosty. Run stock and each fork
backend on the **same** input `.npz`, then compare. A config yaml is required as
a runtime argument (it is never hardcoded).

```sh
PY=/path/to/phrosty-venv/bin/python3
NPZ=/path/to/sfft_inputs_2560.npz       # >= 2560^2 for real validation
CFG=/path/to/phrosty_local_config.yaml
OUT=/path/to/decorr_outputs

# 1. Stock (upstream) reference
$PY run_backend.py --backend stock      --npz "$NPZ" --config "$CFG" --outdir "$OUT" --managed

# 2. Each fork backend (re-run whichever the change touches)
$PY run_backend.py --backend lowmem     --npz "$NPZ" --config "$CFG" --outdir "$OUT" --managed
$PY run_backend.py --backend lowmem4088 --npz "$NPZ" --config "$CFG" --outdir "$OUT" --managed
$PY run_backend.py --backend rfft       --npz "$NPZ" --config "$CFG" --outdir "$OUT" --managed

# 3. Compare each fork against stock (sky-rms = skyrms_sci stored in the npz)
$PY compare.py --stock "$OUT/decorr_stock.npy" \
               --fork "$OUT/decorr_lowmem.npy" "$OUT/decorr_lowmem4088.npy" "$OUT/decorr_rfft.npy" \
               --sky-rms <skyrms_sci>
```

`--managed` enables the cupy managed-memory allocator so large (>= 2560²) frames
can oversubscribe limited GPU memory; drop it if you have ample VRAM.

`compare.py` prints, per fork, the three criteria with PASS/FAIL and an
`OVERALL` verdict (exit code 0 = PASS, 1 = FAIL), plus the separate bright-core
stats. The `--sky-rms` value is the `skyrms_sci` scalar saved in the input npz
(printed by `make_inputs.py`); read it with
`python -c "import numpy as np; print(np.load('$NPZ')['skyrms_sci'])"`.

## Inputs

The input `.npz` holds the arrays the SFFT flow consumes. For an N×N frame with
a P×P PSF stamp:

| key            | shape  | dtype   | meaning |
| -------------- | ------ | ------- | ------- |
| `data_sci`     | (N, N) | float64 | sky-subtracted science image (x, y) |
| `data_templ`   | (N, N) | float64 | sky-subtracted template image (x, y) |
| `var_sci`      | (N, N) | float32 | science variance (noise²) |
| `var_templ`    | (N, N) | float32 | template variance (noise²) |
| `dm_sci`       | (N, N) | uint8   | science detection mask |
| `dm_templ`     | (N, N) | uint8   | template detection mask |
| `psf_sci`      | (P, P) | float64 | science PSF stamp (x, y) |
| `psf_templ`    | (P, P) | float64 | template PSF stamp (x, y) |
| `skyrms_sci`   | scalar | float64 | science sky rms |
| `skyrms_templ` | scalar | float64 | template sky rms |
| `hdr_sci`      | scalar | <U…     | science WCS header string (astropy `Header.tostring()`) |
| `hdr_templ`    | scalar | <U…     | template WCS header string |

Image arrays are stored transposed to (x, y) and contiguous, matching what the
flow expects. (A 1024² example npz is ~27 MB; a 4088² one is ~415 MB.)

**Inputs are NOT committed** — they exceed 100 MB and are excluded by
`.gitignore` (`*.npz`, `*.npy`). Regenerate them with `make_inputs.py`.

### (Re)generating inputs

`make_inputs.py` rebuilds an npz from a real ou2024 science+template pair via
the phrosty input-prep path (DiaObject → ImageCollection `ou2024` →
`Pipeline.sky_sub_all_images()` + `get_psfs()`), then extracts and transposes
the arrays. It needs phrosty + snappl + ou2024 data access configured (the same
environment that runs the pipeline).

```sh
$PY make_inputs.py \
    --config "$CFG" \
    --out /path/to/sfft_inputs_2560.npz \
    --oid 20172782 --band Y106 \
    --science-csv  /path/to/<oid>_instances_science_2.csv \
    --template-csv /path/to/<oid>_instances_templates_1.csv
```

The frame size N is determined by the OU image footprint for the chosen
object/CSVs. To validate at >= 2560², use an object/region whose science cutout
is at least that size (e.g. the 4088² full-SCA pair).

> **Reminder:** real validation must use input at **>= 2560²**. 1024² is for
> harness-smoke only and is **not** a valid validation scale.
