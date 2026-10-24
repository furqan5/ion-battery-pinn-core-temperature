"""Surface-only identification of Bi: why it needs the heat capacity and the heat input to sub-percent.

The surface fit {h, k} with a prescribed source Q = I(V - U) and a prescribed rho_cp matches the measured
surface only if the heat stored in the cell is right. In the quasi-steady regime the profile is
parabolic, so the mean temperature rise is theta_bar = theta_s (1 + Bi/4), and the cell energy balance
    rho_cp V d(theta_bar)/dt = Q - h A theta_s
pins the product rho_cp (1 + Bi/4) relative to Q, while h is pinned by the slow balance. Hence
    dln Bi = (1 + 4/Bi) (dln Q - dln rho_cp),
i.e. surface-only identification of Bi is an indirect calorimetric measurement of the radial excess of
stored heat, and it amplifies relative errors of rho_cp and Q by (1 + 4/Bi).

This script checks that law against refits and reports what the estimator gives with heat capacities
taken from the literature range quoted by Richardson and Howey (Table II of their paper) instead of
their value identified on record 1 with both thermocouples.

Output: results/rev1_surface_route.json, figures/rev1_surface_route.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402
from stage_b_classical import fit  # noqa: E402

FV = RadialFV(N=40)
RHO_DATASHEET = 0.070 / P.V_b       # 70 g core cell weight (A123 datasheet)
SCENARIOS = {
    "R&H identified (record 1, both thermocouples)": (P.rho, 1171.6),
    "R&H Table II literature, upper": (P.rho, 1109.2),
    "R&H Table II literature, lower": (P.rho, 1004.9),
    "datasheet mass, literature upper": (RHO_DATASHEET, 1109.2),
}


def bi_fit(rec, rho_cp, q_scale=1.0):
    q0 = rec.q_measured
    rec.q_measured = lambda: q_scale * q0()
    try:
        r = fit(rec, FV, {"h": 37.0, "k": 0.40}, {"rho_cp": rho_cp}, "measured", float(rec.T_s[0]),
                verbose=False)
    finally:
        rec.q_measured = q0
    return r["p"]["h"], r["p"]["k"], r["p"]["h"] * P.R_o / r["p"]["k"]


def qs_rmse(rec, Bi):
    e = rec.T_s + (Bi / 2) * (rec.T_s - rec.T_inf) - rec.T_c
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    recs = {t: Record(t) for t in ("1", "2")}
    out = {"law": {}, "scenarios": {}}
    for t, rec in recs.items():
        grid = json.load(open(os.path.join("results", f"rev1_bi_vs_rhocp_{t}.json")))
        f, B = np.array(grid["factor"]), np.array(grid["Bi"])
        Bi0 = float(B[np.argmin(np.abs(f - 1))])
        s_rho = float(np.polyfit(np.log(f), np.log(B), 1)[0])
        qs = {}
        for qf in (0.97, 1.03):
            qs[qf] = bi_fit(rec, P.rho_cp(), qf)
        s_q = float(np.log(qs[1.03][2] / qs[0.97][2]) / np.log(1.03 / 0.97))
        s_h = float(np.log(qs[1.03][0] / qs[0.97][0]) / np.log(1.03 / 0.97))
        out["law"][t] = {"Bi0": Bi0, "analytic_1_plus_4_over_Bi": 1 + 4 / Bi0,
                         "dlnBi_dlnrhocp": s_rho, "dlnBi_dlnQ": s_q, "dlnh_dlnQ": s_h,
                         "rho_factor": f.tolist(), "Bi_vs_rho": B.tolist()}
        print(f"record {t}: Bi0 {Bi0:.4f}  1+4/Bi {1+4/Bi0:.3f}  dlnBi/dln(rho_cp) {s_rho:+.3f}  "
              f"dlnBi/dlnQ {s_q:+.3f}  dln h/dlnQ {s_h:+.3f}", flush=True)
    for name, (rho, cp) in SCENARIOS.items():
        row = {"rho": rho, "cp": cp, "rho_cp_rel": rho * cp / P.rho_cp() - 1}
        for t in recs:
            row[f"Bi_from_{t}"] = bi_fit(recs[t], rho * cp)[2]
        row["rmse_rec2_with_Bi1"] = qs_rmse(recs["2"], row["Bi_from_1"])
        row["rmse_rec1_with_Bi2"] = qs_rmse(recs["1"], row["Bi_from_2"])
        out["scenarios"][name] = row
        print(f"{name:48s} rho_cp {100*row['rho_cp_rel']:+5.1f} %  Bi {row['Bi_from_1']:.3f}/{row['Bi_from_2']:.3f}  "
              f"core RMSE rec2 {row['rmse_rec2_with_Bi1']:.3f} K, rec1 {row['rmse_rec1_with_Bi2']:.3f} K", flush=True)
    with open(os.path.join("results", "rev1_surface_route.json"), "w") as fobj:
        json.dump(out, fobj, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    x = np.linspace(-0.07, 0.07, 50)
    for t, col in (("1", "C0"), ("2", "C1")):
        L = out["law"][t]
        ax.plot(np.array(L["rho_factor"]) - 1, np.array(L["Bi_vs_rho"]) / L["Bi0"] - 1, "o", color=col, ms=3,
                label=f"refit, record {t}")
        ax.plot(x, (1 + x) ** (-L["analytic_1_plus_4_over_Bi"]) - 1, "-", color=col, lw=0.8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel(r"relative error of $\rho c_p$ (or $-$ that of $Q$)")
    ax.set_ylabel(r"relative change of identified Bi")
    ax.legend(fontsize=6, frameon=False, title=r"lines: $(1+4/\mathrm{Bi})$ law", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "rev1_surface_route.png"))


if __name__ == "__main__":
    main()
