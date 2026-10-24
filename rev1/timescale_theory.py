"""Frequency-domain basis and generality of the quasi-steady (QS) timescale criterion.

Answers three review points:
  * Why does T_core = T_surf + (Bi/2)(T_surf - T_inf) hold at Bi ~ 1.2, far above the lumped-model
    limit Bi < 0.1?  -> It is the steady solution of radial conduction, valid at any Bi; its error
    is set by the forcing frequency relative to 1/tau_d, not by Bi being small.
  * Does the t_f/tau_d criterion transfer to other cells?  -> In dimensionless form the error depends
    only on (omega*tau_d, Bi) [radius, conductivity, heat capacity enter only through tau_d and Bi].
  * How is t_f defined for an arbitrary drive cycle?  -> The QS error of any measured current profile
    follows from its heat-input spectrum and the exact transfer function; an equivalent galvanostatic
    duration t_f,eq is read off the galvanostatic curve at the same error.

Part A  exact frequency response (1-D cylinder, uniform generation, Robin surface, fixed ambient):
        with a = sqrt(j*omega*tau_d), r0 = I0(a)/I1(a), D = a*I1(a) + Bi*I0(a)
          theta_surf / P = a I1 / D ,  theta_core / P = 1 - Bi / D ,  P = q / (rho_cp * j*omega)
        QS error / gradient = |(1 + Bi/2) a I1/D - 1 + Bi/D| / |Bi (I0 - 1)/D|
Part B  galvanostatic discharges simulated with the validated finite-volume model over a grid of Bi
        and t_f/tau_d, in two different geometries (to show the dimensionless collapse).
Part C  the two measured drive cycles: QS error predicted from the current alone (time domain, FV) and
        the equivalent t_f; compared with the error measured against the core thermocouple.

Output: results/rev1_timescale_theory.json, figures/rev1_qs_frequency.png, figures/rev1_qs_collapse.png
"""
import json
import os
import sys

import numpy as np
from scipy.special import iv, ive

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402


# --------------------------------------------------------------------------- Part A
def qs_rel_error_freq(wtau, Bi):
    """|QS error| / |true gradient| for sinusoidal heat input at dimensionless frequency omega*tau_d."""
    a = np.sqrt(1j * np.asarray(wtau, float))
    I1 = iv(1, a)
    r0 = ive(0, a) / ive(1, a)                 # I0/I1, overflow-safe
    den = a + Bi * r0                           # D / I1
    err = (1 + Bi / 2) * a / den - 1 + Bi / (I1 * den)
    grad = Bi * (r0 - 1 / I1) / den
    return np.abs(err) / np.abs(grad)


def crossing(wtau, eps, level):
    i = np.argmax(eps >= level)
    if eps[i] < level:
        return np.nan
    x0, x1, y0, y1 = np.log(wtau[i - 1]), np.log(wtau[i]), np.log(eps[i - 1]), np.log(eps[i])
    return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0)))


# --------------------------------------------------------------------------- Part B
def galvanostatic_error(fv, R, k, rho_cp, h, t_f, q=2.0e4, dt=None):
    """QS relative error over the thermally active window of a constant-q discharge of length t_f."""
    alpha = k / rho_cp
    tau = R ** 2 / alpha
    dt = dt or max(0.5, t_f / 2000.0)
    t = np.arange(0.0, t_f + dt, dt)
    o = fv.solve(t, np.full_like(t, q), np.zeros_like(t), 0.0, k=k, h=h, rho_cp=rho_cp)
    Bi = h * R / k
    grad = o["T_core"] - o["T_surf"]
    rise = o["T_surf"]
    m = rise > 0.5 * rise.max()
    e = (Bi / 2) * rise[m] - grad[m]
    return float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(grad[m] ** 2))), t_f / tau


def main():
    out = {}
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ---- A
    wtau = np.logspace(-3, 3, 1200)
    A = {}
    for Bi in (0.1, 0.5, 1.15, 1.21, 2.0, 5.0):
        e = qs_rel_error_freq(wtau, Bi)
        lowc = float(np.median(e[wtau < 1e-2] / wtau[wtau < 1e-2]))
        A[str(Bi)] = {"low_freq_slope_eps_per_wtau": lowc,
                      "wtau_at_5pct": crossing(wtau, e, 0.05),
                      "wtau_at_10pct": crossing(wtau, e, 0.10),
                      "wtau_at_25pct": crossing(wtau, e, 0.25)}
    out["A_frequency"] = A

    # ---- B: two geometries, grid of Bi and t_f/tau
    geoms = {
        "A123_26650 (R=12.9 mm, k=0.404)": dict(R=P.R_o, k=P.k_t, rho_cp=P.rho_cp()),
        "generic_21700 (R=10.5 mm, k=0.25)": dict(R=0.0105, k=0.25, rho_cp=2.8e6),
    }
    ratios = np.array([0.15, 0.25, 0.4, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0])
    B = {}
    for gname, g in geoms.items():
        fv = RadialFV(N=40, R=g["R"])
        tau = g["R"] ** 2 / (g["k"] / g["rho_cp"])
        B[gname] = {"tau_d_s": tau}
        for Bi in (0.3, 0.6, 1.2, 2.4):
            h = Bi * g["k"] / g["R"]
            rows = []
            for x in ratios:
                eps, xr = galvanostatic_error(fv, g["R"], g["k"], g["rho_cp"], h, x * tau)
                rows.append((xr, eps))
            rows = np.array(rows)
            pf = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 1]), 1)
            x5 = float(np.exp((np.log(0.05) - pf[1]) / pf[0]))
            B[gname][f"Bi={Bi}"] = {"tf_over_tau": rows[:, 0].tolist(), "eps": rows[:, 1].tolist(),
                                    "power_law_coeff_pct": float(100 * np.exp(pf[1])),
                                    "power_law_exp": float(pf[0]), "tf_over_tau_at_5pct": x5}
            print(f"{gname:34s} Bi={Bi:<4} eps = {100*np.exp(pf[1]):.2f} (tf/tau)^{pf[0]:.2f} %, 5 % at tf/tau={x5:.2f}",
                  flush=True)
    out["B_galvanostatic"] = B

    # ---- C: drive cycles
    fv = RadialFV(N=40)
    C = {}
    params = {"2": (37.0607, 0.393747), "1": (37.2846, 0.418697)}   # from the OTHER record's surface fit
    for tag, (h, k) in params.items():
        rec = Record(tag)
        Bi = h * P.R_o / k
        tau = P.R_o ** 2 / (k / P.rho_cp())
        q = rec.q_measured() / P.V_b
        o = fv.solve(rec.t, q, rec.T_inf, float(rec.T_s[0]), k=k, h=h, rho_cp=P.rho_cp())
        rise_m = o["T_surf"] - rec.T_inf
        grad_m = o["T_core"] - o["T_surf"]
        m = rise_m > 0.5 * rise_m.max()
        eps_pred = float(np.sqrt(np.mean(((Bi / 2) * rise_m[m] - grad_m[m]) ** 2)) / np.sqrt(np.mean(grad_m[m] ** 2)))
        # measured against the core thermocouple, same window definition on measured channels
        rise = rec.T_s - rec.T_inf
        grad = rec.T_c - rec.T_s
        mm = rise > 0.5 * rise.max()
        eps_meas = float(np.sqrt(np.mean(((Bi / 2) * rise[mm] - grad[mm]) ** 2)) / np.sqrt(np.mean(grad[mm] ** 2)))
        # equivalent galvanostatic duration at this Bi (A123 geometry, Part B curve)
        gB = B["A123_26650 (R=12.9 mm, k=0.404)"]
        bis = np.array([0.3, 0.6, 1.2, 2.4])
        j = int(np.argmin(np.abs(bis - Bi)))
        row = gB[f"Bi={bis[j]}"]
        c, p = row["power_law_coeff_pct"] / 100, row["power_law_exp"]
        tf_eq = float((eps_pred / c) ** (1 / p))
        # spectral content of the heat input: fraction of (de-meaned) heat-input variance above 1/tau_d
        qd = q - q.mean()
        Qf = np.fft.rfft(qd)
        f = np.fft.rfftfreq(len(qd), d=rec.dt)
        pw = np.abs(Qf) ** 2
        frac_above = float(pw[f > 1 / (2 * np.pi * tau)].sum() / pw.sum())
        C[tag] = {"Bi": Bi, "tau_d_s": tau, "record_s": float(rec.t[-1]),
                  "old_tf_over_tau_record_length": float(rec.t[-1] / tau),
                  "eps_pred_from_current": eps_pred, "eps_measured_vs_core": eps_meas,
                  "tf_eq_over_tau": tf_eq, "heat_var_frac_above_1_over_2pi_tau": frac_above}
        print(f"DS{tag}: Bi={Bi:.3f} tau={tau:.0f}s eps_pred={100*eps_pred:.2f}% eps_meas={100*eps_meas:.2f}% "
              f"tf_eq/tau={tf_eq:.2f} (old record-length def {rec.t[-1]/tau:.2f})", flush=True)
    out["C_drive_cycles"] = C

    with open("results/rev1_timescale_theory.json", "w") as fobj:
        json.dump(out, fobj, indent=1)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    for Bi in (0.1, 0.5, 1.2, 2.0, 5.0):
        ax.loglog(wtau, 100 * qs_rel_error_freq(wtau, Bi), lw=1, label=f"Bi = {Bi}")
    ax.axhline(5, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"$\omega\tau_\mathrm{d}$")
    ax.set_ylabel("QS relative error (%)")
    ax.set_ylim(0.1, 300)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("figures/rev1_qs_frequency.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    for gname, mk in zip(geoms, ("o", "x")):
        for Bi, col in zip((0.3, 0.6, 1.2, 2.4), ("C0", "C1", "C2", "C3")):
            r = B[gname][f"Bi={Bi}"]
            ax.loglog(r["tf_over_tau"], 100 * np.array(r["eps"]), mk, color=col, ms=3,
                      label=f"Bi={Bi}" if mk == "o" else None)
    ax.axhline(5, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"$t_\mathrm{f}/\tau_\mathrm{d}$")
    ax.set_ylabel("QS relative error (%)")
    ax.legend(fontsize=6, frameon=False, title="o 26650, x 21700", title_fontsize=6)
    fig.tight_layout()
    fig.savefig("figures/rev1_qs_collapse.png")


if __name__ == "__main__":
    main()
