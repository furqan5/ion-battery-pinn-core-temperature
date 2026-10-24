"""Second cell type and cell-to-cell transfer: LG M50 21700 cells with embedded core thermocouples.

Data: Gulsoy et al., J. Energy Storage 54 (2022) 105260; Mendeley Data doi 10.17632/bf6fksxpp2.2.
The MATLAB tables are exported to plain .mat files by warwick_export.m (folder np/ under
$BATTERY_PINN_DATA/warwick_lgm50).

Test used: the 1 C constant-current discharge of each reference performance test (RPT), cells VP010,
VP011, VP012, RPTs at 0, 20, ..., 100 cycles (short-term study) and the long-term RPTs if exported.

Data handling, all from the temperature record alone (the estimator never uses the current):
  * temperatures are logged at 10 Hz on their own clock; they are block-averaged to 1 Hz;
  * the discharge is located as the largest heating event: its end is the peak of the mean
    temperature, its start the last point before the peak where the mean is within 0.03 K of its
    pre-event minimum;
  * channel assignment: the core is the channel with the larger rise during the discharge. For VP010
    and VP012 this is the column labelled surface; Gulsoy et al. report the core about 4 K above the
    surface at the end of a 1 C discharge for all three cells, so the labels are taken as swapped there;
  * ambient and offsets: the cell rests 1800 s before the discharge; over the last 300 s of that rest
    the surface reading defines T_inf (so the surface sensor's own offset cancels in theta) and the
    mean core-minus-surface difference is removed from the core reading as an offset.

Outputs: results/rev1_warwick.json, figures/rev1_warwick.png
"""
import glob
import json
import os
import re

import numpy as np
import scipy.io as sio

ROOT = os.path.join(os.environ.get("BATTERY_PINN_DATA", "Data_Sets"), "warwick_lgm50", "np")
CELLS = ("VP010", "VP011", "VP012")
CYCLES_SHORT = (0, 20, 40, 60, 80, 100)
BASE_S = 300
MAX_ONSET_LAG_S = 20      # a larger core-to-surface onset delay means the two channels are offset in time
RNG = np.random.default_rng(2022)


def to_1hz(t, x):
    n = int(np.floor(t[-1]))
    k = np.floor(t).astype(int)
    m = (k >= 0) & (k < n)
    s = np.bincount(k[m], weights=x[m], minlength=n)
    c = np.bincount(k[m], minlength=n)
    with np.errstate(invalid="ignore"):
        return s / c


def load_event(path):
    d = sio.loadmat(path, squeeze_me=True)
    if "Ts" not in d or "Tc" not in d:
        return None
    tt = np.asarray(d["time_temp"], float)
    A = to_1hz(tt, np.asarray(d["Ts"], float))
    B = to_1hz(tt, np.asarray(d["Tc"], float))
    ok = np.isfinite(A) & np.isfinite(B)
    A = np.interp(np.arange(len(A)), np.where(ok)[0], A[ok])
    B = np.interp(np.arange(len(B)), np.where(ok)[0], B[ok])
    m = 0.5 * (A + B)
    p = int(np.argmax(m))
    lo = max(0, p - 6000)
    base_min = m[lo:p].min()
    i = p
    while i > lo and m[i] > base_min + 0.03:
        i -= 1
    s = i
    if s - BASE_S < 0:
        return None
    rA = A[p] - A[s - BASE_S:s].mean()
    rB = B[p] - B[s - BASE_S:s].mean()
    swapped = rA > rB
    Tc, Ts = (A, B) if swapped else (B, A)
    T_inf = float(Ts[s - BASE_S:s].mean())
    off_cs = float((Tc - Ts)[s - BASE_S:s].mean())
    sd_base = float(Ts[s - BASE_S:s].std())

    def onset(x):
        b = x[s - BASE_S:s].mean()
        j = s
        while j < p and x[j] < b + 0.2:
            j += 1
        return j
    lag = onset(Ts) - onset(Tc)          # the core-to-surface delay of the first 0.2 K rise
    w = slice(s, p + 1)
    return {"Ts": Ts[w], "Tc": Tc[w] - off_cs, "T_inf": T_inf, "off_cs": off_cs, "swapped": bool(swapped),
            "dur_s": p - s, "rise_K": float(Ts[p] - T_inf), "core_rise_K": float(Tc[p] - off_cs - T_inf),
            "grad_peak_K": float(Tc[p] - off_cs - Ts[p]), "base_sd_K": sd_base, "onset_lag_s": int(lag)}


def gain(ev):
    th = ev["Ts"] - ev["T_inf"]
    return float(np.sum((ev["Tc"] - ev["Ts"]) * th) / np.sum(th * th))


def score(ev, G):
    e = ev["Ts"] + G * (ev["Ts"] - ev["T_inf"]) - ev["Tc"]
    return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max()), float(e.mean())


def boot_gain(ev, G, nb=2000, block=300):
    th = ev["Ts"] - ev["T_inf"]
    e = (ev["Tc"] - ev["Ts"]) - G * th
    n = len(e)
    out = np.empty(nb)
    for b in range(nb):
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in RNG.integers(0, n - block, n // block + 1)])[:n]
        out[b] = np.sum((G * th + e[idx]) * th) / np.sum(th * th)
    return float(out.std(ddof=1))


def main():
    ev = {}
    for cell in CELLS:
        for k, cyc in enumerate(CYCLES_SHORT, start=1):
            f = os.path.join(ROOT, "rpt_short", f"{cell}_seg{k}.mat")
            if not os.path.exists(f):
                continue
            e = load_event(f)
            if e is None:
                print(f"{cell} RPT {cyc}: a temperature channel is missing, skipped")
                continue
            if abs(e["onset_lag_s"]) > MAX_ONSET_LAG_S:
                print(f"{cell} RPT {cyc}: channels offset in time (onset lag {e['onset_lag_s']} s), excluded")
                continue
            e["G"] = gain(e)
            e["u_A_G"] = boot_gain(e, e["G"])
            ev[(cell, cyc)] = e
            print(f"{cell} RPT {cyc:3d}: swapped {e['swapped']!s:5s} dur {e['dur_s']:5d} s  rise {e['rise_K']:.2f} K  "
                  f"peak grad {e['grad_peak_K']:.2f} K  off_cs {e['off_cs']:+.3f} K  base sd {e['base_sd_K']:.3f}  "
                  f"G {e['G']:.4f} (u_A {e['u_A_G']:.4f})  in-sample RMSE {score(e, e['G'])[0]:.3f} K", flush=True)
    longterm = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "rpt_long", "*_seg*.mat"))):
        cell, k = re.match(r"(VP\d+)_seg(\d+)", os.path.basename(path)).groups()
        if cell not in CELLS:
            continue
        e = load_event(path)
        if e is None:
            continue
        e["G"] = gain(e)
        longterm[f"{cell}_L{int(k)}"] = {"onset_lag_s": e["onset_lag_s"], "G": e["G"],
                                          "in_sample_rmse": score(e, e["G"])[0]}
        print(f"{cell} long-term RPT {k}: onset lag {e['onset_lag_s']} s, G {e['G']:.4f}, "
              f"in-sample {score(e, e['G'])[0]:.3f} K (reported only)", flush=True)

    keys = list(ev)
    out = {"events": {f"{c}_{k}": {a: v for a, v in ev[(c, k)].items() if not isinstance(v, np.ndarray)}
                      for (c, k) in keys}, "long_term_reported_only": longterm}
    ins = np.array([score(ev[k], ev[k]["G"])[0] for k in keys])
    Gs = np.array([ev[k]["G"] for k in keys])
    core_rise = np.array([ev[k]["core_rise_K"] for k in keys])
    surf_rise = np.array([ev[k]["rise_K"] for k in keys])
    cv = lambda x: float(x.std(ddof=1) / x.mean())  # noqa: E731
    out["summary"] = {"n_tests": len(keys), "in_sample_rmse_range": [float(ins.min()), float(ins.max())],
                      "in_sample_rmse_median": float(np.median(ins)),
                      "G_mean": float(Gs.mean()), "G_median": float(np.median(Gs)), "G_sd": float(Gs.std(ddof=1)),
                      "G_cv": cv(Gs), "G_range": [float(Gs.min()), float(Gs.max())],
                      "core_rise_cv": cv(core_rise), "surface_rise_cv": cv(surf_rise),
                      "core_rise_range": [float(core_rise.min()), float(core_rise.max())],
                      "surface_rise_range": [float(surf_rise.min()), float(surf_rise.max())]}
    print(f"\n{len(keys)} tests: in-sample RMSE {ins.min():.3f}..{ins.max():.3f} K (median {np.median(ins):.3f}); "
          f"G {Gs.min():.3f}..{Gs.max():.3f}, median {np.median(Gs):.3f}, CV {100*cv(Gs):.1f} %")
    print(f"core rise {core_rise.min():.2f}..{core_rise.max():.2f} K (CV {100*cv(core_rise):.1f} %), "
          f"surface rise {surf_rise.min():.2f}..{surf_rise.max():.2f} K (CV {100*cv(surf_rise):.1f} %)")

    # pairwise transfer: calibrate on test i, apply to test j
    cats = {"same session, other cell": [], "same cell, other session": [], "other cell and session": []}
    for i in keys:
        for j in keys:
            if i == j:
                continue
            r = score(ev[j], ev[i]["G"])
            if i[1] == j[1]:
                cats["same session, other cell"].append(r)
            elif i[0] == j[0]:
                cats["same cell, other session"].append(r)
            else:
                cats["other cell and session"].append(r)
    out["pairwise"] = {}
    for name, rs in cats.items():
        if not rs:
            continue
        a = np.array(rs)
        out["pairwise"][name] = {"n": len(rs), "rmse_median": float(np.median(a[:, 0])),
                                 "rmse_p90": float(np.percentile(a[:, 0], 90)), "max_err_median": float(np.median(a[:, 1]))}
        print(f"transfer, {name:26s} n={len(rs):3d}: RMSE median {np.median(a[:, 0]):.3f} K, 90th pct "
              f"{np.percentile(a[:, 0], 90):.3f} K; max error median {np.median(a[:, 1]):.3f} K")
    # leave-one-cell-out pooled gain: mean G of the other cells' tests
    loo = []
    for j in keys:
        Gp = float(np.mean([ev[i]["G"] for i in keys if i[0] != j[0]]))
        r = score(ev[j], Gp)
        loo.append(r)
    loo = np.array(loo)
    out["leave_one_cell_out"] = {"rmse_median": float(np.median(loo[:, 0])), "rmse_max": float(loo[:, 0].max()),
                                 "max_err_median": float(np.median(loo[:, 1]))}
    print(f"leave-one-cell-out pooled gain: RMSE median {np.median(loo[:, 0]):.3f} K, worst {loo[:, 0].max():.3f} K")
    eps = []
    for k in keys:
        e = ev[k]
        th = e["Ts"] - e["T_inf"]
        g = e["Tc"] - e["Ts"]
        m = th > 0.5 * th.max()
        e["eps_qs"] = float(np.sqrt(np.mean((e["G"] * th[m] - g[m]) ** 2)) / np.sqrt(np.mean(g[m] ** 2)))
        eps.append(e["eps_qs"])
        out["events"][f"{k[0]}_{k[1]}"]["eps_qs"] = e["eps_qs"]
        print(f"  {k[0]} {str(k[1]):>4s}: G {e['G']:.3f}  core rise {e['core_rise_K']:.2f}  surface rise {e['rise_K']:.2f}  "
              f"in-sample {score(e, e['G'])[0]:.3f} K  QS error {100*e['eps_qs']:.1f} %  onset lag {e['onset_lag_s']} s")
    out["summary"]["eps_qs_range"] = [float(min(eps)), float(max(eps))]
    dev = np.abs(Gs - np.median(Gs))
    out["summary"]["G_robust_sd"] = float(1.4826 * np.median(dev))
    print(f"QS relative error {100*min(eps):.1f}..{100*max(eps):.1f} %; robust sd of G {1.4826*np.median(dev):.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"VP010": "#2a78d6", "VP011": "#eb6834", "VP012": "#1baf7a"}
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "legend.frameon": False, "font.family": "serif"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.9), dpi=300, gridspec_kw={"width_ratios": [1.25, 1]})
    ex = ("VP012", 20) if ("VP012", 20) in ev else keys[0]
    e = ev[ex]
    tm = np.arange(len(e["Ts"])) / 60
    Gp = float(np.mean([ev[i]["G"] for i in keys if i[0] != ex[0]]))
    a1.plot(tm, e["Ts"], color="#52514e", lw=0.8, label=r"$T_\mathrm{s}$")
    a1.plot(tm, e["Tc"], color="k", lw=1.0, label=r"$T_\mathrm{c,ref}$")
    a1.plot(tm, e["Ts"] + e["G"] * (e["Ts"] - e["T_inf"]), "--", color="#2a78d6", lw=0.8, label="own gain")
    a1.plot(tm, e["Ts"] + Gp * (e["Ts"] - e["T_inf"]), ":", color="#eb6834", lw=1.0, label="other cells' gain")
    a1.set_xlabel("time (min)")
    a1.set_ylabel(r"temperature ($^\circ$C)")
    a1.legend(fontsize=5.5, loc="upper left")
    a1.set_title(f"(a) {ex[0]}, 1 C discharge", fontsize=6.5, loc="left")
    for cell, col in C.items():
        xs = [(k[1] if isinstance(k[1], int) else None) for k in keys if k[0] == cell]
        ks = [k for k in keys if k[0] == cell and isinstance(k[1], int)]
        a2.plot([k[1] for k in ks], [ev[k]["G"] for k in ks], "o-", color=col, ms=3, lw=0.6, label=cell)
    a2.axhline(np.median(Gs), color="#52514e", lw=0.4, ls="--")
    a2.set_xlabel("cycle number")
    a2.set_ylabel(r"calibrated gain $\hat G$")
    a2.legend(fontsize=5.5)
    a2.set_title("(b) gain per test", fontsize=6.5, loc="left")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join("figures", "rev1_warwick.png"))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "rev1_warwick.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    np.savez(os.path.join("results", "rev1_warwick_events.npz"),
             **{f"{c}_{k}_{q}": e[q] for (c, k), e in ev.items() for q in ("Ts", "Tc")},
             **{f"{c}_{k}_Tinf": np.array([e["T_inf"]]) for (c, k), e in ev.items()})


if __name__ == "__main__":
    main()
