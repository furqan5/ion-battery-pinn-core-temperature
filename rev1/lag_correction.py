"""First-order lag correction of the quasi-steady relation (Supplement, Section S5).

For small omega*tau_d the transfer function of Eq. (4) gives g = G [theta_s - (tau/16) d(theta_s)/dt], so a
second calibrated coefficient c = -G tau/16 on the smoothed derivative of the surface rise corrects the
leading dynamic error. Both coefficients are fitted on one record and applied to the other. The fitted
tau is an effective lag: on the finite-volume model driven by the same heat input it comes out at 300-730 s
against a true tau_d near 1000 s, because the drive cycle has content where the expansion is no longer
accurate.

Output: results/rev1_lag_correction.json
"""
import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402

WIN = 61   # s, Savitzky-Golay window for the derivative


def feats(Ts, Tinf, Tc):
    th = Ts - Tinf
    return th, savgol_filter(th, WIN, 2, deriv=1, delta=1.0), Tc - Ts


def main():
    recs = {t: Record(t) for t in ("1", "2")}
    out = {"window_s": WIN}
    for c, v in (("1", "2"), ("2", "1")):
        th, dth, g = feats(recs[c].T_s, recs[c].T_inf, recs[c].T_c)
        coef, *_ = np.linalg.lstsq(np.c_[th, dth], g, rcond=None)
        G0 = float(np.sum(g * th) / np.sum(th * th))
        thv, dthv, gv = feats(recs[v].T_s, recs[v].T_inf, recs[v].T_c)
        e0 = G0 * thv - gv
        e1 = coef[0] * thv + coef[1] * dthv - gv
        out[f"cal{c}_val{v}"] = {"G": float(coef[0]), "c": float(coef[1]), "tau_eff_s": float(-16 * coef[1] / coef[0]),
                                 "rmse_static": float(np.sqrt(np.mean(e0 ** 2))), "max_static": float(np.abs(e0).max()),
                                 "rmse_lag": float(np.sqrt(np.mean(e1 ** 2))), "max_lag": float(np.abs(e1).max())}
        print(c, "->", v, {k: round(x, 4) for k, x in out[f"cal{c}_val{v}"].items()})
    fv = RadialFV(N=40)
    model = {}
    for tag, (h, k) in (("2", (37.0607, 0.393747)), ("1", (37.2846, 0.418697))):
        rec = recs[tag]
        o = fv.solve(rec.t, rec.q_measured() / P.V_b, rec.T_inf, float(rec.T_s[0]), k=k, h=h, rho_cp=P.rho_cp())
        th, dth, g = feats(o["T_surf"], rec.T_inf, o["T_core"])
        coef, *_ = np.linalg.lstsq(np.c_[th, dth], g, rcond=None)
        model[tag] = {"tau_d_true_s": float(P.R_o ** 2 * P.rho_cp() / k), "tau_eff_s": float(-16 * coef[1] / coef[0])}
    out["model_check"] = model
    print(model)
    with open(os.path.join("results", "rev1_lag_correction.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
