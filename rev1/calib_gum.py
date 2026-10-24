"""Calibration route: the quasi-steady gain G = Bi/2 calibrated on a core-instrumented record, then used
surface-only on another record. GUM budget by Monte Carlo propagation (JCGM 101:2008), with a
one-at-a-time breakdown at the instant of peak gradient (JCGM 100:2008 style table).

Measurement model (deployment record v):
    T_c = T_s + G (T_s - T_inf) + D_qs
    G   = sum(g theta) / sum(theta^2)  on the calibration record c,   g = T_c,ref - T_s,  theta = T_s - T_inf
    D_qs  correction for the quasi-steady approximation; estimate 0, rectangular bound
          eps_qs * G * theta(t), eps_qs predicted from the current alone (rev1/timescale_theory.py)

Input quantities and their treatment
  Type A  G, regression scatter: moving-block bootstrap of the calibration residual (300 s blocks)
  Type A  white noise of T_s, T_inf, T_c,ref: s.d. of the high-pass residual (per sample)
  Type B  T_inf offset relative to T_s, bounded by the largest pre-current difference (rectangular)
  Type B  T_c,ref offset relative to T_s, bounded the same way (rectangular)
  Type B  radial position of the core junction, 0..1.5 mm (uniform): reading low by g (r/R)^2
  Type B  thermocouple time constants 0..2 s (rectangular): lag bias tau dT/dt
  Type B  D_qs as above
The two records come from the same cell, rig and sensors, so each sensor offset is ONE draw shared by
calibration and deployment; the correlation this creates between G and the deployment inputs is
carried by the Monte Carlo and not by hand-entered covariances.

Outputs: u(T_c_hat), u(T_c,ref), u(d) and E_n over the record; the estimator RMSE against the TRUE
core as an interval; the breakdown at peak gradient. results/rev1_calib_gum.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record  # noqa: E402

M = int(os.environ.get("GUM_M", "20000"))
N_BOOT = 4000
BLOCK_S = 300.0
R_TC_MAX = 1.5e-3
TAU_MAX = 2.0
RNG = np.random.default_rng(1002008)


def hp_sd(x, k=15):
    sm = np.convolve(x, np.ones(k) / k, mode="same")
    return float((x - sm)[k:-k].std())


def gain(Ts, Tc, Tinf, axis=-1):
    th = Ts - Tinf
    return np.sum((Tc - Ts) * th, axis=axis) / np.sum(th * th, axis=axis)


def block_boot_gain(rec, G, n_boot, rng):
    th = rec.T_s - rec.T_inf
    e = (rec.T_c - rec.T_s) - G * th
    n, nb = len(e), int(round(BLOCK_S / rec.dt))
    out = np.empty(n_boot)
    for b in range(n_boot):
        eb = np.empty(n)
        i = 0
        while i < n:
            s = rng.integers(0, n - nb)
            m = min(nb, n - i)
            eb[i:i + m] = e[s:s + m]
            i += m
        out[b] = np.sum((G * th + eb) * th) / np.sum(th * th)
    return out


def pre_current_offsets(rec):
    m = np.abs(rec.I) < 0.5
    m[np.argmax(np.abs(rec.I) >= 0.5):] = False
    return float(np.mean(rec.T_c[m] - rec.T_s[m])), float(np.mean(rec.T_s[m] - rec.T_inf[m]))


def simulate(cal, val, G0, Gboot, eps_qs, a_cs, a_si, m, active):
    """One chunk of m Monte Carlo draws. `active` selects which input quantities vary."""
    def on(k):
        return active is None or k in active
    z = lambda shape: np.zeros(shape)  # noqa: E731
    d_cs = RNG.uniform(-a_cs, a_cs, (m, 1)) if on("off_cs") else z((m, 1))
    d_si = RNG.uniform(-a_si, a_si, (m, 1)) if on("off_si") else z((m, 1))
    r_tc = RNG.uniform(0, R_TC_MAX, (m, 1)) if on("pos") else z((m, 1))
    tau_c = RNG.uniform(0, TAU_MAX, (m, 1)) if on("lag") else z((m, 1))
    tau_s = RNG.uniform(0, TAU_MAX, (m, 1)) if on("lag") else z((m, 1))
    dGa = (RNG.choice(Gboot, (m, 1)) - G0) if on("G_A") else z((m, 1))
    uq = RNG.uniform(-1, 1, (m, 1)) if on("qs") else z((m, 1))

    def corrected(rec, noise):
        n = len(rec.t)
        ns = RNG.normal(0, hp_sd(rec.T_s), (m, n)) if noise else z((m, n))
        ni = RNG.normal(0, hp_sd(rec.T_inf), (m, n)) if noise else z((m, n))
        nc = RNG.normal(0, hp_sd(rec.T_c), (m, n)) if noise else z((m, n))
        Ts = rec.T_s - ns + tau_s * np.gradient(rec.T_s, rec.t)
        Tinf = rec.T_inf - ni - d_si
        grad = np.maximum(rec.T_c - rec.T_s, 0.0)
        Tc = rec.T_c - d_cs - nc + tau_c * np.gradient(rec.T_c, rec.t) + grad * (r_tc / P.R_o) ** 2
        return Ts, Tinf, Tc

    Ts_c, Tinf_c, Tc_c = corrected(cal, on("noise"))
    G = gain(Ts_c, Tc_c, Tinf_c)[:, None] + dGa
    Ts_v, Tinf_v, Tc_v = corrected(val, on("noise"))
    th_v = Ts_v - Tinf_v
    Y = Ts_v + G * th_v + uq * eps_qs * G0 * (val.T_s - val.T_inf)[None, :]
    return Y, Tc_v


def main():
    recs = {t: Record(t) for t in ("1", "2")}
    pre = {t: pre_current_offsets(recs[t]) for t in recs}
    a_cs = max(abs(v[0]) for v in pre.values())
    a_si = max(abs(v[1]) for v in pre.values())
    tt = json.load(open(os.path.join("results", "rev1_timescale_theory.json")))["C_drive_cycles"]
    out = {"inputs": {"a_cs": a_cs, "a_si": a_si, "pre_current": pre, "M": M, "block_s": BLOCK_S,
                      "R_tc_max_m": R_TC_MAX, "tau_max_s": TAU_MAX}}
    for c, v in (("1", "2"), ("2", "1")):
        cal, val = recs[c], recs[v]
        G0 = float(gain(cal.T_s, cal.T_c, cal.T_inf))
        Gboot = block_boot_gain(cal, G0, N_BOOT, RNG)
        eps_qs = float(tt[v]["eps_pred_from_current"])
        That = val.T_s + G0 * (val.T_s - val.T_inf)
        d_obs = That - val.T_c
        i = int(np.argmax(val.T_c - val.T_s))

        # full Monte Carlo
        sY = np.zeros(len(val.t)); sY2 = np.zeros_like(sY)
        sR = np.zeros_like(sY); sR2 = np.zeros_like(sY)
        sD = np.zeros_like(sY); sD2 = np.zeros_like(sY)
        rmse_true, Gs = [], []
        done = 0
        while done < M:
            m = min(400, M - done)
            Y, Tref = simulate(cal, val, G0, Gboot, eps_qs, a_cs, a_si, m, None)
            sY += Y.sum(0); sY2 += (Y ** 2).sum(0)
            sR += Tref.sum(0); sR2 += (Tref ** 2).sum(0)
            D = Y - Tref
            sD += D.sum(0); sD2 += (D ** 2).sum(0)
            rmse_true.append(np.sqrt(((That[None, :] - Tref) ** 2).mean(1)))
            done += m
        sd = lambda s, s2: np.sqrt(np.maximum(s2 / M - (s / M) ** 2, 0))  # noqa: E731
        uY, uR, uD = sd(sY, sY2), sd(sR, sR2), sd(sD, sD2)
        En = d_obs / (2 * uD)
        rmse_true = np.concatenate(rmse_true)
        q = np.percentile(rmse_true, [2.5, 50, 97.5])

        # one-at-a-time breakdown at the peak-gradient instant (estimate side and reference side)
        comps = {"G, calibration scatter (bootstrap)": ("G_A", "A"),
                 "sensor white noise (all channels)": ("noise", "A"),
                 "T_inf offset relative to T_s": ("off_si", "B"),
                 "T_c,ref offset relative to T_s": ("off_cs", "B"),
                 "core junction radial position": ("pos", "B"),
                 "thermocouple lag": ("lag", "B"),
                 "quasi-steady approximation D_qs": ("qs", "B")}
        rows = []
        for name, (key, typ) in comps.items():
            Y, Tref = simulate(cal, val, G0, Gboot, eps_qs, a_cs, a_si, 1500, {key})
            rows.append({"quantity": name, "type": typ, "u_hat_K": float(Y[:, i].std()),
                         "u_ref_K": float(Tref[:, i].std()), "u_d_K": float((Y - Tref)[:, i].std())})

        print(f"\n=== calibrate on record {c}, deploy on record {v} ===")
        print(f"  G = {G0:.5f} (Bi = {2*G0:.4f}); bootstrap u_A(G) = {Gboot.std(ddof=1):.5f} "
              f"({100*Gboot.std(ddof=1)/G0:.2f} %); eps_qs = {100*eps_qs:.2f} %")
        print(f"  peak gradient at t = {val.t[i]:.0f} s: theta = {val.T_s[i]-val.T_inf[i]:.3f} K, "
              f"T_c,ref = {val.T_c[i]:.3f} C, T_c_hat = {That[i]:.3f} C, d = {d_obs[i]:+.3f} K")
        print(f"  {'input quantity':38s} {'type':>4s} {'u(T_c_hat)':>11s} {'u(T_c,ref)':>11s} {'u(d)':>8s}")
        for r in rows:
            print(f"  {r['quantity']:38s} {r['type']:>4s} {r['u_hat_K']:11.4f} {r['u_ref_K']:11.4f} {r['u_d_K']:8.4f}")
        print(f"  combined (full MC):                         {uY[i]:11.4f} {uR[i]:11.4f} {uD[i]:8.4f}"
              f"   U(d), k=2 = {2*uD[i]:.3f} K")
        print(f"  over the record: median u(T_c_hat) {np.median(uY):.3f} K (max {uY.max():.3f}); "
              f"|E_n| <= 1 for {100*np.mean(np.abs(En) <= 1):.1f} % of samples")
        print(f"  observed RMSE(d) {np.sqrt(np.mean(d_obs**2)):.3f} K, max |d| {np.abs(d_obs).max():.3f} K; "
              f"RMSE vs TRUE core: median {q[1]:.3f} K, 95 % interval [{q[0]:.3f}, {q[2]:.3f}] K")
        np.savez(os.path.join("results", f"rev1_calib_gum_cal{c}_val{v}.npz"), t=val.t, T_s=val.T_s,
                 T_c=val.T_c, T_inf=val.T_inf, T_hat=That, u_hat=uY, u_ref=uR, u_d=uD, En=En)
        out[f"cal{c}_val{v}"] = {
            "G": G0, "Bi": 2 * G0, "u_A_G": float(Gboot.std(ddof=1)), "eps_qs": eps_qs,
            "t_peak_s": float(val.t[i]), "theta_peak_K": float(val.T_s[i] - val.T_inf[i]),
            "d_peak_K": float(d_obs[i]), "breakdown_peak": rows,
            "u_hat_peak_K": float(uY[i]), "u_ref_peak_K": float(uR[i]), "u_d_peak_K": float(uD[i]),
            "U_d_peak_k2_K": float(2 * uD[i]),
            "u_hat_median_K": float(np.median(uY)), "u_hat_max_K": float(uY.max()),
            "frac_abs_En_le_1": float(np.mean(np.abs(En) <= 1)),
            "rmse_obs_K": float(np.sqrt(np.mean(d_obs ** 2))), "max_abs_d_K": float(np.abs(d_obs).max()),
            "bias_obs_K": float(d_obs.mean()),
            "rmse_true_median_K": float(q[1]), "rmse_true_p2_5_K": float(q[0]), "rmse_true_p97_5_K": float(q[2]),
        }
    with open(os.path.join("results", "rev1_calib_gum.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
