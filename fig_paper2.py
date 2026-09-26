"""Figures for paper 2: identifiability-gated inverse estimation on real cells.

All inputs come from the Part 3 re-execution in this repository
(verify/Part3_CRLB_Classical_Inverse_executed.ipynb and nasa_final_fits.csv),
not from transcribed values. Reads the CSV with NumPy, so pandas is not needed.
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 660, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "legend.frameon": False})


def load(path):
    """Column name -> NumPy array; True/False columns become booleans."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if set(vals) <= {"True", "False"}:
            cols[k] = np.array([v == "True" for v in vals])
        else:
            cols[k] = np.array([float(v) if v != "" else np.nan for v in vals])
    return cols


def iqr_over_median(a):
    return (np.quantile(a, .75) - np.quantile(a, .25)) / np.median(a)


d = load("nasa_final_fits.csv")
n = len(d["cap"])
print(f"{n} cycles, {int(d['railed'].sum())} railed, "
      f"{int(d['pass_'].sum())} passing gates")

# ---------------------------------------------------------------- Fig 1 ----
# Recovered heat-generation multiplier vs depth of discharge, with the
# cycle-to-cycle band. This is the physical result: a non-monotonic profile
# inferred from one surface channel.
x = np.linspace(0, 1, 101)
curves = np.array([1 + c1 * x + c2 * x ** 2 for c1, c2 in zip(d["c1"], d["c2"])])
med = np.median(curves, axis=0)
p5, p95 = np.percentile(curves, [5, 95], axis=0)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.fill_between(x, p5, p95, color="#1565c0", alpha=0.18, lw=0,
                label="5th–95th percentile, 168 cycles")
ax.plot(x, med, color="#1565c0", lw=1.7, label="median")
i = int(np.argmin(med))
ax.plot(x[i], med[i], "o", color="#c62828", ms=5, zorder=5)
ax.annotate(f"minimum at $x$ = {x[i]:.2f}\n({med[i]:.3f}$\\,R_0$)",
            (x[i], med[i]), textcoords="offset points", xytext=(0, 30),
            fontsize=6.5, color="#c62828", ha="center", va="bottom")
ax.set_xlabel("depth of discharge  $x$")
ax.set_ylabel("heat-generation multiplier  $R(x)/R_0$")
ax.legend(loc="upper left", fontsize=6.5)
fig.tight_layout(); fig.savefig("figures/p2_source_shape.png", bbox_inches="tight")
print(f"  Fig1: minimum at x={x[i]:.3f}, multiplier {med[i]:.4f}, "
      f"end {med[-1]:.3f}x")

# ---------------------------------------------------------------- Fig 2 ----
# Ageing: both the start-of-discharge and end-of-discharge coefficients track
# capacity fade, the latter more strongly.
cap, r0, rend = d["cap"], d["R0_mOhm"], d["R_end_mOhm"]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(cap, r0, "o", ms=3, color="#1565c0", alpha=0.75,
        label=f"$R_0$   $r$ = {np.corrcoef(cap, r0)[0,1]:+.3f}")
ax.plot(cap, rend, "s", ms=3, color="#c62828", alpha=0.75,
        label=f"$R_\\mathrm{{end}}$   $r$ = {np.corrcoef(cap, rend)[0,1]:+.3f}")
for y, c in ((r0, "#1565c0"), (rend, "#c62828")):
    k = np.polyfit(cap, y, 1)
    xs = np.linspace(cap.min(), cap.max(), 10)
    ax.plot(xs, np.polyval(k, xs), "-", color=c, lw=1.0, alpha=0.6)
ax.set_xlabel("discharge capacity  /  A h")
ax.set_ylabel("recovered coefficient  /  m$\\Omega$")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.35), fontsize=6.5,
          frameon=True, framealpha=0.9, edgecolor="none")
fig.tight_layout(); fig.savefig("figures/p2_ageing.png", bbox_inches="tight")
print(f"  Fig2: corr(cap,R0) = {np.corrcoef(cap, r0)[0,1]:.4f}, "
      f"corr(cap,R_end) = {np.corrcoef(cap, rend)[0,1]:.4f}")

# ---------------------------------------------------------------- Fig 3 ----
# Profile likelihood for a shared convection coefficient. Values are the
# executed notebook's printed output (cell 33). Labels sit inside the axes:
# the refinement label is right-aligned against its line, in the empty space
# above the left branch of the curve.
h = np.array([8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 46, 52, 60], float)
tot = np.array([0.5298, 0.4746, 0.4223, 0.3508, 0.2911, 0.2387, 0.2296,
                0.2653, 0.3513, 0.4624, 0.6091, 0.7578, 0.9489])
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(h, tot, "o-", color="#6a1b9a", ms=3.5, lw=1.4)
ax.axvline(24.81, color="#2e7d32", ls="--", lw=1.0)
ax.annotate("parabolic\nrefinement\n$h$ = 24.81", (24.81, 0.80),
            textcoords="offset points", xytext=(-4, 0),
            fontsize=6.5, color="#2e7d32", ha="right", va="center")
ax.axvline(29.57, color="#c62828", ls=":", lw=1.0)
ax.annotate("cooling-branch\nestimate, 29.57\n(truncation-biased)",
            (33.5, 0.80), fontsize=6.5, color="#c62828")
ax.set_xlabel("assumed convection coefficient  $h$  /  W m$^{-2}$K$^{-1}$")
ax.set_ylabel("pooled residual  /  K")
fig.tight_layout(); fig.savefig("figures/p2_profile_likelihood.png",
                                bbox_inches="tight")
print(f"  Fig3: minimum {tot.min():.4f} K at h = {h[int(np.argmin(tot))]:.0f}, "
      f"depth (max-min)/min = {(tot.max()-tot.min())/tot.min():.3f}")

print("\nsummary statistics for the manuscript:")
print(f"  R0     median {np.median(r0):.2f} mOhm  "
      f"IQR/median {iqr_over_median(r0):.3f}")
print(f"  R_end  median {np.median(rend):.2f} mOhm")
print(f"  c1     median {np.median(d['c1']):.3f}   c2 median {np.median(d['c2']):.3f}")
print(f"  RMSE   median {np.median(d['rmse']):.4f} K   rise median {np.median(d['rise']):.2f} K"
      f"  -> {100*np.median(d['rmse'])/np.median(d['rise']):.2f} % relative")
print(f"  usable {int(d['pass_'].sum())}/{n}  railed {int(d['railed'].sum())}")
xm = -d["c1"] / (2 * d["c2"])
print(f"  x_min  median {np.median(xm):.3f}  IQR [{np.quantile(xm, .25):.3f}, {np.quantile(xm, .75):.3f}]")
