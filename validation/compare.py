#!/usr/bin/env python3
"""STOCK-vs-FORK comparison of decorrelated difference images, against the
fp32-fork STANDING RULE.

Given the stock decorrelated-diff .npy, one or more fork .npy files, and the
sky rms, this computes -- per fork -- the three rule criteria on the BACKGROUND
of the difference image:

  (1) background rms vs stock (must be within ~5%);
  (2) lag-1 spatial autocorrelation (ACF): the fork must MATCH stock (introduce
      no excess correlation); absolute whiteness is reported for info only -- the
      decorrelated diff is not necessarily white (stock f64 has the same residual
      ACF at some scales), so whiteness must NOT fail a faithful fork -- only
      fork-vs-stock divergence does;
  (3) pairwise rms(fork - stock) over the background, as a % of sky_rms
      (must be <~ a few %).

The bright-source-core statistics are reported SEPARATELY: f32 rounding in
bright cores is expected and acceptable, and must not contaminate the
background verdict. The background is defined as the frame MINUS a central box
(default half-width 200 px) so a bright transient core near frame-centre is
excluded from criteria (1)-(3).

A clear PASS/FAIL is printed per fork against the thresholds. NO hardcoded paths.

REMINDER: a real validation run must use input at >= 2560**2. 1024**2 is
harness-smoke only.
"""
import argparse

import numpy as np

# --- Rule thresholds -------------------------------------------------------
BGRMS_PCT_TOL = 5.0     # (1) |fork bgrms - stock bgrms| / stock  <= 5 %
ACF_WHITE_TOL = 0.05    # (2) |ACF| must be small (~white)
ACF_MATCH_TOL = 0.02    # (2) |ACF_fork - ACF_stock| must be small
PAIR_SKY_PCT_TOL = 3.0  # (3) rms(fork-stock)/sky over bkg  <= ~3 %


def bkg_mask(shape, half):
    """Boolean mask: True over background = frame minus central (2*half)px box."""
    n0, n1 = shape
    c0, c1 = n0 // 2, n1 // 2
    m = np.ones(shape, bool)
    m[c0 - half:c0 + half, c1 - half:c1 + half] = False
    return m


def metrics(a, half):
    """Return (bgrms, lag1_acf, bkg_values) for image ``a``.

    bgrms  : sqrt(mean(bkg**2))
    acf    : lag-1 spatial ACF along columns, computed over the background
             pixels only (mean-subtracted), as a whiteness diagnostic.
    """
    m = bkg_mask(a.shape, half)
    bg = a[m]
    rms = float(np.sqrt(np.mean(bg**2)))

    # lag-1 ACF over background, horizontal neighbours, masked to background.
    x = a - a[m].mean()
    pair_valid = m[:, :-1] & m[:, 1:]
    left = x[:, :-1][pair_valid]
    right = x[:, 1:][pair_valid]
    denom = float(np.mean(x[m] ** 2))
    acf = float(np.mean(left * right) / denom) if denom > 0 else float("nan")
    return rms, acf, bg


def core_stats(a, half):
    """Bright-source-core stats over the central box (reported separately)."""
    n0, n1 = a.shape
    c0, c1 = n0 // 2, n1 // 2
    core = a[c0 - half:c0 + half, c1 - half:c1 + half]
    return dict(rms=float(np.sqrt(np.mean(core**2))),
                absmax=float(np.max(np.abs(core))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", required=True, help="stock decorr .npy")
    ap.add_argument("--fork", required=True, nargs="+",
                    help="one or more fork decorr .npy files")
    ap.add_argument("--sky-rms", required=True, type=float,
                    help="sky rms of the science frame (e.g. skyrms_sci from the npz)")
    ap.add_argument("--core-half", type=int, default=200,
                    help="half-width (px) of the excluded central core box (default 200)")
    args = ap.parse_args(argv)

    S = np.load(args.stock)
    sky = args.sky_rms
    half = args.core_half
    srms, sacf, _ = metrics(S, half)
    score = core_stats(S, half)

    print(f"STOCK  {args.stock}")
    print(f"  shape={S.shape}  bg_rms={srms:.6g}  bg_ACF={sacf:+.4f}  "
          f"sky_rms={sky:.4g}  sqrt2*sky={np.sqrt(2) * sky:.4g}")
    print(f"  [core] rms={score['rms']:.6g}  absmax={score['absmax']:.6g}  (reference)")
    print()

    overall_pass = True
    for f in args.fork:
        F = np.load(f)
        if F.shape != S.shape:
            print(f"FORK   {f}\n  SHAPE MISMATCH {F.shape} vs stock {S.shape} -> FAIL\n")
            overall_pass = False
            continue
        frms, facf, _ = metrics(F, half)
        fcore = core_stats(F, half)

        m = bkg_mask(F.shape, half)
        d = F - S
        pair_bg_rms = float(np.sqrt(np.mean(d[m] ** 2)))
        pair_absmax = float(np.max(np.abs(d)))

        # --- criteria ---
        bgrms_pct = 100.0 * abs(frms - srms) / srms if srms else float("inf")
        c1 = bgrms_pct <= BGRMS_PCT_TOL
        # Criterion (2) pass/fail is MATCH-TO-STOCK: the fork must introduce no
        #   excess spatial correlation relative to the f64 reference. Absolute
        #   whiteness is informational only -- the decorrelated diff is not
        #   necessarily white (it isn't here; stock f64 has the same residual ACF),
        #   so requiring |ACF|~0 would fail a bit-faithful fork. Only fork-vs-stock
        #   DIVERGENCE is a fork defect.
        acf_match = abs(facf - sacf) <= ACF_MATCH_TOL
        c2 = acf_match
        acf_white = abs(facf) <= ACF_WHITE_TOL    # info
        stock_white = abs(sacf) <= ACF_WHITE_TOL  # info
        pair_sky_pct = 100.0 * pair_bg_rms / sky if sky else float("inf")
        c3 = pair_sky_pct <= PAIR_SKY_PCT_TOL
        ok = c1 and c2 and c3
        overall_pass = overall_pass and ok

        print(f"FORK   {f}   ->  {'PASS' if ok else 'FAIL'}")
        print(f"  shape={F.shape}")
        print(f"  (1) bg_rms={frms:.6g}  vs stock {srms:.6g}   "
              f"diff={bgrms_pct:.2f}%  (tol {BGRMS_PCT_TOL}%)   "
              f"[{'ok' if c1 else 'BAD'}]")
        print(f"  (2) bg_ACF={facf:+.4f}  vs stock {sacf:+.4f}   "
              f"match(|Δ|<= {ACF_MATCH_TOL}):[{'ok' if acf_match else 'BAD'}]")
        print(f"      info: ~white(|ACF|<= {ACF_WHITE_TOL})?  fork={'y' if acf_white else 'n'} "
              f"stock={'y' if stock_white else 'n'}"
              f"{'   (decorr not white at this scale -- in stock f64 too, not a fork issue)' if not stock_white else ''}")
        print(f"  (3) rms(fork-stock) over bkg = {pair_bg_rms:.6g}  "
              f"= {pair_sky_pct:.3f}% of sky  (tol {PAIR_SKY_PCT_TOL}%)   "
              f"[{'ok' if c3 else 'BAD'}]")
        print(f"  [core, SEPARATE/acceptable] rms={fcore['rms']:.6g}  "
              f"absmax={fcore['absmax']:.6g}  "
              f"full-frame max|fork-stock|={pair_absmax:.6g}")
        print()

    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
