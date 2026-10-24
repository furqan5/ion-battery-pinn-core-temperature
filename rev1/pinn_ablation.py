"""Inverse-PINN ablation and normalised PDE residual (review points on PINN generality and residual).

Same surface-only inverse problem as stage_e_inverse.py (record 2, h and k from the record-1 surface
fit, core channel read only to score). Varied one at a time against the baseline (tanh, 48 x 4, no
Fourier features, w_data = 200):
  * width 96, depth 6           (capacity)
  * Fourier time features, n_freq = 8
  * SiLU activation
  * w_data = 50 and 1000        (loss weighting)
Selection inside each configuration stays truth-free (lowest PDE + BC residual).

Normalised PDE residual of the split problem, on a fixed collocation set of 20 000 points:
    eta = rms(r_pde) / rms(2 Bi g),
where 2 Bi g is the forcing term of the deviation equation, plus the same residual in W/m^3,
r_pde * k dT_ref / R^2, against the rms volumetric heat input.

Output: results/rev1_pinn_ablation.json, results/rev1_pinn_baseline.npz
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stage_e_inverse as se  # noqa: E402
from part7_lib import P, Record  # noqa: E402
from pinn_split import DeviationNet, LumpedBase, split_pde_residual  # noqa: E402

N_SEEDS = int(os.environ.get("ABL_SEEDS", "3"))
CONFIGS = {
    "baseline (tanh, 48x4, w_data 200)": {},
    "width 96, depth 6": {"width": 96, "depth": 6},
    "Fourier time features, n_freq 8": {"n_freq": 8},
    "SiLU activation": {"act": "silu"},
    "w_data 50": {"w_data": 50.0},
    "w_data 1000": {"w_data": 1000.0},
}

_last = {}


def _net_factory(act):
    def make(width=48, depth=4, n_freq=0):
        net = DeviationNet(width, depth, n_freq)
        if act == "silu":
            for i, m in enumerate(net.net):
                if isinstance(m, nn.Tanh):
                    net.net[i] = nn.SiLU()
        _last["net"] = net
        return net
    return make


class _Base(LumpedBase):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _last["base"] = self


def residual_metrics(rec, r):
    net, base = _last["net"], _last["base"]
    k, h = se.K_FIXED, se.H_FIXED
    Fo = (k / P.rho_cp()) * float(rec.t[-1]) / P.R_o ** 2
    Bi = h * P.R_o / k
    dT_ref = float(rec.T_s.max() - rec.T_inf.mean())
    g = torch.Generator().manual_seed(12345)
    s = torch.rand(20000, 1, generator=g)
    t = torch.rand(20000, 1, generator=g)
    R = torch.tensor(r["R_eff"])
    shp = torch.tensor(r["shape"]) if len(r["shape"]) else None
    rp = split_pde_residual(net, Fo, Bi, base, s, t, R, shp, dT_ref).detach().numpy().ravel()
    forcing = (2.0 * Bi * base.excess(t, R, shp) / dT_ref).detach().numpy().ravel()
    q_rms = float(np.sqrt(np.mean((rec.q_measured() / P.V_b) ** 2)))
    w_rms = float(np.sqrt(np.mean(rp ** 2)) * k * dT_ref / P.R_o ** 2)
    return {"eta_pde": float(np.sqrt(np.mean(rp ** 2)) / np.sqrt(np.mean(forcing ** 2))),
            "pde_residual_rms_W_m3": w_rms, "q_rms_W_m3": q_rms, "residual_over_q": w_rms / q_rms}


def main():
    rec = Record("2")
    se.LumpedBase = _Base
    out = {}
    base_runs = None
    for name, c in CONFIGS.items():
        c = dict(c)
        se.DeviationNet = _net_factory(c.pop("act", "tanh"))
        runs = []
        for sd in range(N_SEEDS):
            t0 = time.time()
            r = se.run_inverse(rec, n_shape=0, seed=sd, **c)
            r.update(residual_metrics(rec, r))
            e = r["Tc"] - rec.T_c
            r["core_rmse"] = float(np.sqrt(np.mean(e ** 2)))
            r["core_max"] = float(np.abs(e).max())
            r["wall_s"] = time.time() - t0
            runs.append(r)
            print(f"{name:36s} seed {sd}: surf {r['surf_rmse']:.4f} K  core {r['core_rmse']:.4f} K  "
                  f"eta {r['eta_pde']:.3e}  sel {r['sel']:.3e}  {r['wall_s']:.0f} s", flush=True)
        b = int(np.argmin([r["sel"] for r in runs]))
        out[name] = {"selected_seed": b,
                     "selected": {k: runs[b][k] for k in ("surf_rmse", "core_rmse", "core_max", "eta_pde",
                                                           "pde_residual_rms_W_m3", "q_rms_W_m3",
                                                           "residual_over_q", "R_eff", "sel")},
                     "core_rmse_all": [r["core_rmse"] for r in runs],
                     "surf_rmse_all": [r["surf_rmse"] for r in runs],
                     "eta_all": [r["eta_pde"] for r in runs]}
        if base_runs is None:
            base_runs = runs
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "rev1_pinn_ablation.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join("results", "rev1_pinn_baseline.npz"),
             Tc=np.array([r["Tc"] for r in base_runs]), Ts=np.array([r["Ts"] for r in base_runs]),
             sel=np.array([r["sel"] for r in base_runs]), t=rec.t, T_c_meas=rec.T_c, T_s_meas=rec.T_s)


if __name__ == "__main__":
    main()
