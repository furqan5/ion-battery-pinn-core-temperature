"""GUM uncertainty budget for the quasi-steady core-temperature estimate (JCGM 100:2008, JCGM 101:2008).

Measurand:  core temperature T_c(t) of the A123 26650 cell, estimated by
                T_c_hat = T_s + (Bi/2) (T_s - T_inf)                      (model equation)
Reference:  core thermocouple reading T_c_ref(t) (held out of every fit).

Three quantities are kept separate, as the review asks:
  (i)   u(T_c_hat)       uncertainty of the estimate (what a BMS user gets from this virtual sensor)
  (ii)  u(T_c_ref)       uncertainty of the reference core thermocouple
  (iii) the estimator ERROR e = T_c_hat - T_c_true, recovered from the observed difference
        d = T_c_hat - T_c_ref by adding back the reference's own error distribution. Its RMSE is
        reported as an interval, not a bare number.

Offsets are expressed relative to the surface thermocouple (reference frame): a calibration offset
common to all three thermocouples cancels in both the estimate's error and in d.

Input quantities (all evaluated in this script or in rev1/bi_bootstrap.py):
  Bi        Type A  moving-block bootstrap of the surface-only fit on the OTHER record (Gaussian)
  Bi        between-record difference: used only as a CONSISTENCY CHECK, not added to the budget.
                    |Bi_1 - Bi_2| is compared with the bootstrap u of the difference; adding it as a
                    further component would count the same fit scatter twice (JCGM 100, 4.2).
  rho_cp    Type B  volumetric heat capacity, relative half-width A_RHO (default 3 %, rectangular).
                    It is not in the model equation, but the surface-only fit converts it into Bi with a
                    relative sensitivity of about -4, so it is propagated through the actual refit:
                    Bi(rho_cp) is refitted on a grid and interpolated in each Monte Carlo draw.
  T_s noise Type A  s.d. of the high-pass residual (Gaussian, per sample)
  T_inf     Type A  s.d. of the high-pass residual (Gaussian, per sample)
  d_si      Type B  surface-ambient sensor offset, bounded by the largest pre-current difference
                    observed in either record (rectangular)
  d_cs      Type B  core-surface sensor offset, bounded the same way (rectangular)
  r_tc      Type B  radial position of the core junction, 0..1.5 mm from the axis (rectangular);
                    reading bias -(T_c - T_s)(r_tc/R)^2 from the parabolic profile
  tau_tc    Type B  thermocouple time constant 0..2 s (rectangular); lag bias -tau dT/dt
Monte Carlo: M = 20000 draws of the per-record constants; per-sample noise drawn independently.

Output: results/rev1_gum_budget.json and a printed GUM table at the instant of peak gradient.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402
from stage_b_classical import fit  # noqa: E402

M = int(os.environ.get("GUM_M", "20000"))
R_TC_MAX = 1.5e-3      # m
TAU_MAX = 2.0          # s
A_RHO = float(os.environ.get("GUM_RHOCP_REL", "0.03"))
RHO_GRID = np.array([0.94, 0.97, 1.0, 1.03, 1.06])
RNG = np.random.default_rng(100_2008)


def hp_sd(x, k=15):
    sm = np.convolve(x, np.ones(k) / k, mode="same")
    return float((x - sm)[k:-k].std())


def rect(a, size):
    return RNG.uniform(-a, a, size)


def bi_vs_rhocp(rec):
    """Surface-only {h, k} refit at each rho_cp factor; returns Bi on RHO_GRID (cached)."""
    path = os.path.join("results", f"rev1_bi_vs_rhocp_{rec.tag}.json")
    if os.path.exists(path):
        d = json.load(open(path))
        if np.allclose(d["factor"], RHO_GRID):
            return np.array(d["Bi"])
    fv = RadialFV(N=40)
    out = []
    for f in RHO_GRID:
        r = fit(rec, fv, {"h": 37.0, "k": 0.40}, {"rho_cp": f * P.rho_cp()}, "measured", float(rec.T_s[0]),
                verbose=False)
        out.append(r["p"]["h"] * P.R_o / r["p"]["k"])
        print(f"  record {rec.tag}: rho_cp x {f:.2f} -> h {r['p']['h']:.3f}, k {r['p']['k']:.4f}, Bi {out[-1]:.4f}",
              flush=True)
    json.dump({"factor": RHO_GRID.tolist(), "Bi": out}, open(path, "w"), indent=1)
    return np.array(out)


def main():
    boot = json.load(open(os.path.join("results", "rev1_bi_bootstrap.json")))
    Bi = {t: boot[t]["Bi"] for t in ("1", "2")}
    u_repro = abs(Bi["1"] - Bi["2"]) / np.sqrt(2.0)

    recs = {t: Record(t) for t in ("1", "2")}
    pre = {}
    for t, r in recs.items():
        m = np.abs(r.I) < 0.5
        m[np.argmax(np.abs(r.I) >= 0.5):] = False          # only before the first current
        pre[t] = {"cs": float(np.mean(r.T_c[m] - r.T_s[m])), "si": float(np.mean(r.T_s[m] - r.T_inf[m]))}
    a_cs = max(abs(pre[t]["cs"]) for t in pre)
    a_si = max(abs(pre[t]["si"]) for t in pre)

    bgrid = {t: bi_vs_rhocp(recs[t]) for t in recs}
    sens = {t: float(np.polyfit(np.log(RHO_GRID), np.log(bgrid[t]), 1)[0]) for t in recs}
    out = {"inputs": {"Bi": Bi, "u_Bi_repro": u_repro, "a_cs": a_cs, "a_si": a_si, "pre_current": pre,
                      "R_tc_max_m": R_TC_MAX, "tau_max_s": TAU_MAX, "M": M, "A_rho_rel": A_RHO,
                      "rho_grid": RHO_GRID.tolist(), "Bi_vs_rho": {t: bgrid[t].tolist() for t in recs},
                      "dlnBi_dlnrhocp": sens}}
    u_diff = float(np.hypot(boot["1"]["boot_Bi_sd"], boot["2"]["boot_Bi_sd"]))
    out["inputs"]["between_record_check"] = {"abs_dBi": abs(Bi["1"] - Bi["2"]), "u_diff_boot": u_diff,
                                             "ratio": abs(Bi["1"] - Bi["2"]) / u_diff}
    print(f"dlnBi/dln(rho_cp): {sens}", flush=True)
    print(f"between-record check: |Bi1-Bi2| = {abs(Bi['1']-Bi['2']):.4f}, bootstrap u of difference {u_diff:.4f} "
          f"(ratio {abs(Bi['1']-Bi['2'])/u_diff:.2f})", flush=True)

    for tag, other in (("2", "1"), ("1", "2")):
        rec = recs[tag]
        b = boot[other]
        Bi0 = b["Bi"]
        u_boot = b["boot_Bi_sd"]
        # Bi as a function of rho_cp for the record Bi is taken from; shift relative to the nominal fit
        coef = np.polyfit(RHO_GRID - 1.0, bgrid[other], 2)
        dBi_rho = lambda x: np.polyval(coef, x) - np.polyval(coef, 0.0)   # noqa: E731
        u_rho_lin = abs(np.polyval(np.polyder(coef), 0.0)) * A_RHO / np.sqrt(3)
        u_s, u_inf, u_c = hp_sd(rec.T_s), hp_sd(rec.T_inf), hp_sd(rec.T_c)

        Ts, Tinf, Tc = rec.T_s, rec.T_inf, rec.T_c
        n = len(Ts)
        dTc = np.gradient(Tc, rec.t)
        dTs = np.gradient(Ts, rec.t)
        That = Ts + (Bi0 / 2) * (Ts - Tinf)
        d_obs = That - Tc

        # ---- Monte Carlo over per-record constants (chunked)
        rmse_true, mean_true, u_hat_t, u_ref_t = [], [], np.zeros(n), np.zeros(n)
        sum_h = np.zeros(n); sum_h2 = np.zeros(n); sum_r = np.zeros(n); sum_r2 = np.zeros(n)
        done = 0
        while done < M:
            m = min(1000, M - done)
            Bi_s = Bi0 + RNG.normal(0, u_boot, (m, 1)) + dBi_rho(rect(A_RHO, (m, 1)))
            d_si = rect(a_si, (m, 1))                         # T_inf reads true + d_si (rel. to surface)
            d_cs = rect(a_cs, (m, 1))                         # T_c_ref reads true + d_cs
            r_tc = RNG.uniform(0, R_TC_MAX, (m, 1))
            tau_c = RNG.uniform(0, TAU_MAX, (m, 1))
            tau_s = RNG.uniform(0, TAU_MAX, (m, 1))
            ns = RNG.normal(0, u_s, (m, n))
            ni = RNG.normal(0, u_inf, (m, n))
            nc = RNG.normal(0, u_c, (m, n))
            # (i) estimate built from perturbed inputs: true inputs unknown, so perturb the readings by
            #     the negatives of their errors (equivalent in distribution)
            Ts_i = Ts - ns + tau_s * dTs                      # remove noise, undo lag
            Tinf_i = Tinf - ni - d_si
            That_s = Ts_i + (Bi_s / 2) * (Ts_i - Tinf_i)
            sum_h += That_s.sum(0); sum_h2 += (That_s ** 2).sum(0)
            # (ii) reference: true core = reading - offset - noise + lag + position under-read
            grad = np.maximum(Tc - Ts, 0.0)
            Tc_true = Tc - d_cs - nc + tau_c * dTc + grad * (r_tc / P.R_o) ** 2
            sum_r += Tc_true.sum(0); sum_r2 += (Tc_true ** 2).sum(0)
            # (iii) error of the (nominal) estimate against the true core
            e_true = That[None, :] - Tc_true
            rmse_true.append(np.sqrt((e_true ** 2).mean(1)))
            mean_true.append(e_true.mean(1))
            done += m
        rmse_true = np.concatenate(rmse_true)
        mean_true = np.concatenate(mean_true)
        u_hat_t = np.sqrt(np.maximum(sum_h2 / M - (sum_h / M) ** 2, 0))
        u_ref_t = np.sqrt(np.maximum(sum_r2 / M - (sum_r / M) ** 2, 0))
        u_d = np.sqrt(u_hat_t ** 2 + u_ref_t ** 2)
        En = d_obs / (2 * u_d)

        # ---- GUM LPU table at the instant of peak measured gradient
        i = int(np.argmax(Tc - Ts))
        rise = Ts[i] - Tinf[i]
        rows = [
            ("Bi, fit (bootstrap)", Bi0, u_boot, "A", "Gaussian", rise / 2),
            (f"Bi via rho_cp +/-{100*A_RHO:.0f} %", 0.0, u_rho_lin, "B", "rectangular", rise / 2),
            ("T_s noise", Ts[i], u_s, "A", "Gaussian", 1 + Bi0 / 2),
            ("T_s lag (tau<=2 s)", 0.0, TAU_MAX * abs(dTs[i]) / np.sqrt(3), "B", "rectangular", 1 + Bi0 / 2),
            ("T_inf noise", Tinf[i], u_inf, "A", "Gaussian", -Bi0 / 2),
            ("T_inf offset vs T_s", 0.0, a_si / np.sqrt(3), "B", "rectangular", -Bi0 / 2),
        ]
        ref_rows = [
            ("T_c,ref noise", Tc[i], u_c, "A", "Gaussian", 1.0),
            ("T_c,ref offset vs T_s", 0.0, a_cs / np.sqrt(3), "B", "rectangular", 1.0),
            ("T_c,ref radial position", 0.0, (Tc[i] - Ts[i]) * (R_TC_MAX / P.R_o) ** 2 / np.sqrt(5), "B",
             "r uniform 0..1.5 mm", 1.0),
            ("T_c,ref lag (tau<=2 s)", 0.0, TAU_MAX * abs(dTc[i]) / np.sqrt(3), "B", "rectangular", 1.0),
        ]
        uc_hat = float(np.sqrt(sum((c * u) ** 2 for _, _, u, _, _, c in rows)))
        uc_ref = float(np.sqrt(sum((c * u) ** 2 for _, _, u, _, _, c in ref_rows)))

        print(f"\n=== Record {tag} (Bi from record {other}) — GUM budget at peak gradient, t = {rec.t[i]:.0f} s ===")
        print(f"  T_s={Ts[i]:.3f} C  T_inf={Tinf[i]:.3f} C  rise={rise:.3f} K  T_c,ref={Tc[i]:.3f} C  "
              f"T_c_hat={That[i]:.3f} C  d={d_obs[i]:+.3f} K")
        print(f"  {'quantity':28s} {'u':>8s} {'type':>8s} {'distribution':>18s} {'c':>8s} {'|c|u (K)':>9s}")
        for name, _, u, typ, dist, c in rows + ref_rows:
            print(f"  {name:28s} {u:8.4f} {typ:>8s} {dist:>18s} {c:8.3f} {abs(c)*u:9.4f}")
        print(f"  u_c(T_c_hat) = {uc_hat:.3f} K   u_c(T_c,ref) = {uc_ref:.3f} K   "
              f"U(d), k=2 = {2*np.hypot(uc_hat, uc_ref):.3f} K   |d| = {abs(d_obs[i]):.3f} K")
        print(f"  Monte Carlo over the record: median u(T_c_hat) {np.median(u_hat_t):.3f} K "
              f"(max {u_hat_t.max():.3f}), median u(T_c,ref) {np.median(u_ref_t):.3f} K, "
              f"|E_n|<=1 for {100*np.mean(np.abs(En)<=1):.1f} % of samples")
        q = np.percentile(rmse_true, [2.5, 50, 97.5])
        print(f"  observed RMSE(d) = {np.sqrt(np.mean(d_obs**2)):.3f} K; estimator RMSE vs TRUE core: "
              f"median {q[1]:.3f} K, 95 % interval [{q[0]:.3f}, {q[2]:.3f}] K")

        out[tag] = {
            "Bi_used": Bi0, "Bi_from": other, "u_Bi_boot": u_boot, "u_Bi_repro": u_repro, "u_Bi_rho_lin": float(u_rho_lin),
            "u_noise_Ts": u_s, "u_noise_Tinf": u_inf, "u_noise_Tc": u_c,
            "t_peak_s": float(rec.t[i]), "rise_peak_K": float(rise), "d_peak_K": float(d_obs[i]),
            "uc_hat_peak_K": uc_hat, "uc_ref_peak_K": uc_ref, "U_d_peak_k2_K": float(2 * np.hypot(uc_hat, uc_ref)),
            "budget_estimate": [dict(zip(("quantity", "value", "u", "type", "distribution", "c"), r)) for r in rows],
            "budget_reference": [dict(zip(("quantity", "value", "u", "type", "distribution", "c"), r)) for r in ref_rows],
            "mc_median_u_hat_K": float(np.median(u_hat_t)), "mc_max_u_hat_K": float(u_hat_t.max()),
            "mc_median_u_ref_K": float(np.median(u_ref_t)),
            "frac_abs_En_le_1": float(np.mean(np.abs(En) <= 1)),
            "rmse_obs_K": float(np.sqrt(np.mean(d_obs ** 2))), "bias_obs_K": float(d_obs.mean()),
            "rmse_true_median_K": float(q[1]), "rmse_true_p2_5_K": float(q[0]), "rmse_true_p97_5_K": float(q[2]),
            "bias_true_p2_5_K": float(np.percentile(mean_true, 2.5)),
            "bias_true_p97_5_K": float(np.percentile(mean_true, 97.5)),
        }
    suffix = "" if abs(A_RHO - 0.03) < 1e-12 else f"_rho{100*A_RHO:g}pct"
    with open(os.path.join("results", f"rev1_gum_budget{suffix}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
