"""Figures for the supplementary material, redrawn from the saved results (no refitting).

  figures/rev1_supp_profile.png        profile likelihood of the surface-only fit (k and Bi profiles)
  figures/rev1_supp_surface_route.png  identified Bi against relative error of rho_cp, with the (1 + 4/Bi) law
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "legend.frameon": False, "font.family": "serif",
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def main():
    pl = json.load(open(os.path.join("results", "rev1_profile_likelihood.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), dpi=300)
    for tag, col in (("1", C1), ("2", C2)):
        kp, bp = np.array(pl[tag]["k_profile"]), np.array(pl[tag]["Bi_profile"])
        s0 = min(kp[:, 2].min(), bp[:, 3].min())
        axs[0].plot(kp[:, 0], np.maximum(kp[:, 2] / s0 - 1, 1e-6), "o-", ms=2.5, lw=0.8, color=col,
                    label=f"record {tag}")
        axs[1].plot(bp[:, 0], np.maximum(bp[:, 3] / s0 - 1, 1e-6), "o-", ms=2.5, lw=0.8, color=col,
                    label=f"record {tag}")
        for ax in axs:
            ax.axhline(pl[tag]["threshold_rel_dSSE"], color=col, lw=0.5, ls="--")
    axs[0].set_xlabel(r"$k$ (W m$^{-1}$ K$^{-1}$), $h$ refitted")
    axs[1].set_xlabel(r"Bi, $k$ refitted")
    for ax in axs:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3)
        ax.set_ylabel(r"$\Delta$SSE / SSE$_\min$")
        ax.legend(fontsize=6, title="dashed: 95 % threshold", title_fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "rev1_supp_profile.png"))

    sr = json.load(open(os.path.join("results", "rev1_surface_route.json")))
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    x = np.linspace(-0.07, 0.07, 60)
    for tag, col in (("1", C1), ("2", C2)):
        L = sr["law"][tag]
        ax.plot(np.array(L["rho_factor"]) - 1, np.array(L["Bi_vs_rho"]) / L["Bi0"] - 1, "o", color=col, ms=3.5,
                label=f"refit, record {tag}")
        ax.plot(x, (1 + x) ** (-L["analytic_1_plus_4_over_Bi"]) - 1, "-", color=col, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.4)
    ax.axvline(0, color=INK2, lw=0.4)
    ax.set_xlabel(r"relative error of $\rho c_p$ (equivalently, minus that of $Q$)")
    ax.set_ylabel("relative change of identified Bi")
    ax.legend(fontsize=6, title=r"lines: $(1+\Delta)^{-(1+4/\mathrm{Bi})}-1$", title_fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "rev1_supp_surface_route.png"))


if __name__ == "__main__":
    main()
