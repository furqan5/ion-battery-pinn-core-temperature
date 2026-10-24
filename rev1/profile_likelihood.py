"""Profile likelihood of the surface-only fit: practical identifiability of k, h and Bi.

For each record, the surface-only {h, k} fit (measured source I(V-U), rho_cp fixed) is profiled:
  * over k, with h re-optimised at each k;
  * over Bi, with k re-optimised at each Bi (h = Bi k / R).
The profiled objective is the sum of squared surface residuals. Because the residuals are strongly
autocorrelated, the 95 % threshold uses an effective sample size n_eff = n (1 - rho1)/(1 + rho1)
(AR(1) approximation): Delta SSE / SSE_min <= chi2_1(0.95) / n_eff.

The core channel is used only to REPORT the quasi-steady core error along each profile, to show that
the core estimate depends on Bi and not on k and h separately. Nothing is selected on it.

Output: results/rev1_profile_likelihood.json, figures/rev1_profile_likelihood.png
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402


def main():
    fv = RadialFV(N=40)
    out = {}
    for tag in ("1", "2"):
        rec = Record(tag)
        T0 = float(rec.T_s[0])
        q = rec.q_measured() / P.V_b

        def sse(h, k):
            o = fv.solve(rec.t, q, rec.T_inf, T0, k=k, h=h, rho_cp=P.rho_cp())
            return float(np.sum((o["T_surf"] - rec.T_s) ** 2)), o

        def qs_core_rmse(Bi):
            e = rec.T_s + (Bi / 2) * (rec.T_s - rec.T_inf) - rec.T_c
            return float(np.sqrt(np.mean(e ** 2)))

        # k-profile
        ks = np.linspace(0.25, 0.70, 19)
        kprof = []
        for k in ks:
            r = minimize_scalar(lambda h: sse(h, k)[0], bounds=(5.0, 200.0), method="bounded",
                                options={"xatol": 1e-4})
            Bi = r.x * P.R_o / k
            kprof.append((k, r.x, r.fun, Bi, qs_core_rmse(Bi)))
            print(f"DS{tag} k={k:.3f} h*={r.x:.3f} Bi={Bi:.4f} SSE={r.fun:.3f}", flush=True)
        kprof = np.array(kprof)

        # Bi-profile
        bis = np.linspace(0.85, 1.55, 21)
        bprof = []
        for Bi in bis:
            r = minimize_scalar(lambda k: sse(Bi * k / P.R_o, k)[0], bounds=(0.1, 3.0), method="bounded",
                                options={"xatol": 1e-6})
            bprof.append((Bi, r.x, Bi * r.x / P.R_o, r.fun, qs_core_rmse(Bi)))
        bprof = np.array(bprof)

        smin = min(kprof[:, 2].min(), bprof[:, 3].min())
        _, obest = sse(37.0, 0.40)
        i_b = int(np.argmin(bprof[:, 3]))
        res = rec.T_s - sse(bprof[i_b, 2], bprof[i_b, 1])[1]["T_surf"]
        rho1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        n = len(res)
        n_eff = n * (1 - rho1) / (1 + rho1)
        thr = chi2.ppf(0.95, 1) / n_eff

        def interval(x, s):
            ok = (s / smin - 1) <= thr
            return (float(x[ok].min()), float(x[ok].max())) if ok.any() else (np.nan, np.nan)

        kint = interval(kprof[:, 0], kprof[:, 2])
        bint = interval(bprof[:, 0], bprof[:, 3])
        out[tag] = {"n": n, "rho1": rho1, "n_eff": n_eff, "threshold_rel_dSSE": thr,
                    "k_profile": kprof.tolist(), "Bi_profile": bprof.tolist(),
                    "k_95_interval": kint, "Bi_95_interval": bint,
                    "k_range_scanned": [float(ks.min()), float(ks.max())],
                    "Bi_along_k_profile_range": [float(kprof[:, 3].min()), float(kprof[:, 3].max())],
                    "qs_core_rmse_along_k_profile_range_K": [float(kprof[:, 4].min()), float(kprof[:, 4].max())]}
        print(f"DS{tag}: rho1={rho1:.4f} n_eff={n_eff:.1f} thr={thr:.4f}  k 95% {kint}  Bi 95% {bint}  "
              f"Bi along k-profile {kprof[:,3].min():.4f}..{kprof[:,3].max():.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "rev1_profile_likelihood.json"), "w") as f:
        json.dump(out, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=300)
    for tag, col in (("1", "C0"), ("2", "C1")):
        kp = np.array(out[tag]["k_profile"])
        bp = np.array(out[tag]["Bi_profile"])
        s0 = min(kp[:, 2].min(), bp[:, 3].min())
        axs[0].plot(kp[:, 0], kp[:, 2] / s0 - 1, "o-", ms=2, lw=0.8, color=col, label=f"record {tag}")
        axs[1].plot(bp[:, 0], bp[:, 3] / s0 - 1, "o-", ms=2, lw=0.8, color=col, label=f"record {tag}")
        for ax in axs:
            ax.axhline(out[tag]["threshold_rel_dSSE"], color=col, lw=0.5, ls="--")
    axs[0].set_xlabel(r"$k$ (W m$^{-1}$ K$^{-1}$), $h$ re-fitted")
    axs[1].set_xlabel(r"Bi, $k$ re-fitted")
    for ax in axs:
        ax.set_ylabel(r"$\Delta$SSE / SSE$_\min$")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("figures/rev1_profile_likelihood.png")


if __name__ == "__main__":
    main()
