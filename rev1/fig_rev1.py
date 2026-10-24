"""Figures for the revised TIM manuscript.

  figures/rev1_validation.png  record 2, gain calibrated on record 1: measured core (reference), estimate with
                               its expanded uncertainty (k = 2), measured surface; lower panel: difference
                               d = T_c_hat - T_c,ref with the expanded uncertainty of d (k = 2)
  figures/rev1_timescale.png   (a) exact frequency response of the quasi-steady error (Bi-independent);
                               (b) galvanostatic error against t_f/tau_d for two geometries and four Bi, the
                               multi-rate records of Table IV (modelled gradient) and the two drive cycles
                               (measured core) placed at their equivalent forcing time
Also prints the measured quasi-steady error of each drive cycle with the cross-calibrated gain.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timescale_theory import qs_rel_error_freq  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "font.family": "serif"})

# multi-rate records (Catenaro and Onori), Table IV of the paper: rate, t_f/tau_d, error %
MULTIRATE = [(1, 3.61, 1.91), (2, 1.79, 3.36), (3, 1.19, 5.70), (5, 0.71, 10.61), (10, 0.35, 25.82),
             (15, 0.23, 42.83), (20, 0.17, 63.01)]


def qs_err_measured(d, G):
    th = d["T_s"] - d["T_inf"]
    g = d["T_c"] - d["T_s"]
    m = th > 0.5 * th.max()
    return float(np.sqrt(np.mean((G * th[m] - g[m]) ** 2)) / np.sqrt(np.mean(g[m] ** 2)))


def main():
    gum = json.load(open("results/rev1_calib_gum.json"))
    tt = json.load(open("results/rev1_timescale_theory.json"))

    # ---------------------------------------------------------------- validation figure
    d = np.load("results/rev1_calib_gum_cal1_val2.npz")
    t = d["t"] / 60.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.1), dpi=300, sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    a1.fill_between(t, d["T_hat"] - 2 * d["u_hat"], d["T_hat"] + 2 * d["u_hat"], color=C1, alpha=0.18, lw=0,
                    label=r"$\hat T_\mathrm{c}\pm U$ ($k=2$)")
    a1.plot(t, d["T_s"], color=INK2, lw=0.8, label=r"$T_\mathrm{s}$ (input)")
    a1.plot(t, d["T_c"], color=INK, lw=1.0, label=r"$T_\mathrm{c,ref}$ (held out)")
    a1.plot(t, d["T_hat"], color=C1, lw=0.8, ls="--", label=r"$\hat T_\mathrm{c}$")
    a1.plot(t, d["T_inf"], color=INK2, lw=0.5, ls=":", label=r"$T_\infty$")
    a1.set_ylabel(r"temperature ($^\circ$C)")
    a1.set_ylim(5.5, 26.5)
    a1.legend(fontsize=6, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    dd = d["T_hat"] - d["T_c"]
    a2.fill_between(t, -2 * d["u_d"], 2 * d["u_d"], color=C2, alpha=0.18, lw=0, label=r"$\pm U(d)$, $k=2$")
    a2.plot(t, dd, color=C2, lw=0.8, label=r"$d=\hat T_\mathrm{c}-T_\mathrm{c,ref}$")
    a2.axhline(0, color=INK2, lw=0.4)
    a2.set_ylabel("d (K)")
    a2.set_xlabel("time (min)")
    a2.legend(fontsize=6, ncol=2, loc="lower right")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=0.3)
    fig.savefig("figures/rev1_validation.png")

    # ---------------------------------------------------------------- timescale figure
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.0, 2.5), dpi=300)
    w = np.logspace(-2, 2, 600)
    e = 100 * qs_rel_error_freq(w, 1.2)
    b1.loglog(w, e, color=C1, lw=1.2, label="exact, any Bi")
    b1.loglog(w, 100 * w / 16, color=INK2, lw=0.6, ls="--", label=r"$\omega\tau_\mathrm{d}/16$")
    b1.axhline(5, color=INK2, lw=0.4, ls=":")
    b1.axvline(0.8, color=INK2, lw=0.4, ls=":")
    b1.text(0.85, 0.12, r"$\omega\tau_\mathrm{d}=0.8$", fontsize=6, color=INK2)
    b1.set_xlabel(r"$\omega\tau_\mathrm{d}$")
    b1.set_ylabel("quasi-steady error (%)")
    b1.set_ylim(0.05, 300)
    b1.legend(fontsize=6, loc="upper left")
    b1.set_title("(a) sinusoidal heat input", fontsize=7, loc="left")

    B = tt["B_galvanostatic"]
    geoms = list(B)
    cols = {"Bi=0.3": C1, "Bi=0.6": C2, "Bi=1.2": C3, "Bi=2.4": C4}
    for gi, (g, mk) in enumerate(zip(geoms, ("o", "^"))):
        for key, col in cols.items():
            r = B[g][key]
            b2.loglog(r["tf_over_tau"], 100 * np.array(r["eps"]), mk, mfc="none" if gi else col, mec=col,
                      ms=3.2, mew=0.6, ls="-" if gi == 0 else "none", lw=0.5, color=col,
                      label=key.replace("Bi=", "Bi = ") if gi == 0 else None)
    mr = np.array(MULTIRATE)
    b2.loglog(mr[:, 1], mr[:, 2], "s", color=INK, ms=3, mfc="white", mew=0.7,
              label="multi-rate records (modelled gradient)")
    G = {"2": gum["cal1_val2"]["G"], "1": gum["cal2_val1"]["G"]}
    for tag, mkr in (("2", "*"), ("1", "P")):
        dd_ = np.load(f"results/rev1_calib_gum_cal{'1' if tag == '2' else '2'}_val{tag}.npz")
        em = qs_err_measured(dd_, G[tag])
        x = tt["C_drive_cycles"][tag]["tf_eq_over_tau"]
        ep = tt["C_drive_cycles"][tag]["eps_pred_from_current"]
        b2.loglog([x], [100 * em], mkr, color=INK, ms=6 if mkr == "*" else 4,
                  label=f"drive cycle {tag}: measured core")
        b2.loglog([x], [100 * ep], mkr, color=INK, mfc="white", mew=0.6, ms=6 if mkr == "*" else 4,
                  label=f"drive cycle {tag}: predicted from $I(t)$")
        print(f"record {tag}: measured QS error with cross-calibrated G = {100*em:.2f} %, "
              f"predicted from current {100*tt['C_drive_cycles'][tag]['eps_pred_from_current']:.2f} %, "
              f"t_f,eq/tau_d = {x:.2f}")
    b2.axhline(5, color=INK2, lw=0.4, ls=":")
    b2.set_xlabel(r"$t_\mathrm{f}/\tau_\mathrm{d}$")
    b2.set_ylabel("quasi-steady error (%)")
    b2.set_title(r"(b) galvanostatic discharge; $\circ$ 26650, $\triangle$ 21700", fontsize=7, loc="left")
    b2.legend(fontsize=5.5, loc="lower left", ncol=1)
    from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter
    b2.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5]))
    b2.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    b2.xaxis.set_minor_formatter(NullFormatter())
    for a in (b1, b2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig("figures/rev1_timescale.png")


if __name__ == "__main__":
    main()
