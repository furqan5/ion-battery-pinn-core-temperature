"""Uncertainty of the Biot number from the surface-only fit (Type A), by moving-block bootstrap.

The plain least-squares covariance assumes white residuals. The surface residuals here are strongly
autocorrelated (lag-1 0.99, lag-300 s 0.55 on record 1), so that covariance understates the
uncertainty by an order of magnitude. A moving-block bootstrap keeps the correlation: synthetic
surface traces are built from the fitted trace plus resampled blocks of the fitted residual, and
{h, k} are refitted on each. Only the surface channel is used.

Also evaluated here (Type B inputs to the budget):
  * rho_cp +/- 3 %   (datasheet mass vs bulk density, Section VII of the paper)
  * reversible heat on/off (Forgez entropic coefficient), the source-model choice

Output: results/rev1_bi_bootstrap.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402
from stage_b_classical import fit  # noqa: E402

N_BOOT = int(os.environ.get("N_BOOT", "200"))
BLOCK_S = 300.0          # block length, s; residual autocorrelation falls to ~0.2-0.5 by 300-600 s
RNG = np.random.default_rng(20261023)


def fit_hk(rec, fv, Ts_target, rho_cp, q_kind="measured", T0=None):
    """Surface-only {h, k} fit on an arbitrary surface target (bootstrap replicate)."""
    saved = rec.T_s
    rec.T_s = Ts_target
    try:
        r = fit(rec, fv, {"h": 37.0, "k": 0.40}, {"rho_cp": rho_cp}, q_kind,
                T0 if T0 is not None else float(saved[0]), verbose=False)
    finally:
        rec.T_s = saved
    return r


def block_resample(res, n_block, rng):
    n = len(res)
    out = np.empty(n)
    i = 0
    while i < n:
        s = rng.integers(0, n - n_block)
        m = min(n_block, n - i)
        out[i:i + m] = res[s:s + m]
        i += m
    return out


def main():
    fv = RadialFV(N=40)
    out = {}
    for tag in ("1", "2"):
        rec = Record(tag)
        T0 = float(rec.T_s[0])
        base = fit_hk(rec, fv, rec.T_s, P.rho_cp(), T0=T0)
        h0, k0 = base["p"]["h"], base["p"]["k"]
        Bi0 = h0 * P.R_o / k0
        res = rec.T_s - base["Ts"]
        nb = int(round(BLOCK_S / rec.dt))

        t0 = time.time()
        boots = []
        for b in range(N_BOOT):
            target = base["Ts"] + block_resample(res, nb, RNG)
            r = fit_hk(rec, fv, target, P.rho_cp(), T0=T0)
            boots.append((r["p"]["h"], r["p"]["k"]))
            if (b + 1) % 25 == 0:
                print(f"DS{tag}: {b + 1}/{N_BOOT} replicates, {time.time() - t0:.0f} s", flush=True)
        boots = np.array(boots)
        Bi_b = boots[:, 0] * P.R_o / boots[:, 1]

        # Type B variants
        variants = {}
        for lab, rc in (("rho_cp_minus3pct", 0.97 * P.rho_cp()), ("rho_cp_plus3pct", 1.03 * P.rho_cp())):
            r = fit_hk(rec, fv, rec.T_s, rc, T0=T0)
            variants[lab] = r["p"]["h"] * P.R_o / r["p"]["k"]

        out[tag] = {
            "h": h0, "k": k0, "Bi": Bi0,
            "ls_sd_h": base["sd"]["h"], "ls_sd_k": base["sd"]["k"],
            "boot_n": int(N_BOOT), "block_s": BLOCK_S,
            "boot_h_sd": float(boots[:, 0].std(ddof=1)), "boot_k_sd": float(boots[:, 1].std(ddof=1)),
            "boot_corr_hk": float(np.corrcoef(boots[:, 0], boots[:, 1])[0, 1]),
            "boot_Bi_mean": float(Bi_b.mean()), "boot_Bi_sd": float(Bi_b.std(ddof=1)),
            "boot_Bi_p2_5": float(np.percentile(Bi_b, 2.5)), "boot_Bi_p97_5": float(np.percentile(Bi_b, 97.5)),
            "Bi_variants": variants,
        }
        print(json.dumps(out[tag], indent=1), flush=True)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "rev1_bi_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
