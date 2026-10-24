"""Sensitivity of the quasi-steady relation to non-uniform heat generation and to axial heat flow.

Part 1 (radial non-uniformity, closed form + check): with a volume-preserving radial weight
        w(r) = 1 + beta (1 - 2 r^2/R^2), steady conduction gives
            T(0) - T(R) = (q R^2 / 4k)(1 + beta/2),  T(R) - T_inf = q R / 2h,
        so the gradient/rise ratio is (Bi/2)(1 + beta/2): the uniform-generation relation is biased by
        the factor (1 + beta/2). Checked against the 1-D finite-volume model.

Part 2 (axial heat flow, tabs, end losses): 2-D axisymmetric (r, z) finite-volume model with
        anisotropic conductivity (k_r radial across layers, k_z axial along the current collectors),
        Robin cooling on the side (h) and on the two end caps (h_end), and an optional extra heat
        fraction f_tab deposited in the top 3 mm next to the positive tab. Driven by the measured
        record-2 heat input I(V - U). The thermocouples sit at mid-height, so the quasi-steady relation
        is scored there: T_c_hat = T_s,mid + (Bi/2)(T_s,mid - T_inf) against T_core,mid.

Output: results/rev1_heatgen_axial.json
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part7_lib import P, Record, RadialFV  # noqa: E402

H, K_R = 37.0607, 0.393747          # record-1 surface-only fit, used to score record 2 (as in the paper)
RHO_CP = P.rho_cp()
LENGTH = 0.065                       # m, A123 ANR26650 (Richardson & Howey, Sec. V)


def part1():
    rec = Record("2")
    fv = RadialFV(N=60)
    q = rec.q_measured() / P.V_b
    rows = []
    for beta in (-0.4, -0.2, 0.0, 0.2, 0.4):
        o = fv.solve(rec.t, q, rec.T_inf, float(rec.T_s[0]), k=K_R, h=H, rho_cp=RHO_CP,
                     q_weight=fv.radial_weight(beta))
        Bi = H * P.R_o / K_R
        rise = o["T_surf"] - rec.T_inf
        grad = o["T_core"] - o["T_surf"]
        m = rise > 0.5 * rise.max()
        ratio = float(np.mean(grad[m] / rise[m]))
        e = (Bi / 2) * rise[m] - grad[m]
        rows.append({"beta": beta, "closed_form_factor": 1 + beta / 2,
                     "fv_ratio_over_Bi2": ratio / (Bi / 2),
                     "qs_rel_err_pct": float(100 * np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(grad[m] ** 2)))})
        print(f"beta={beta:+.1f}: closed-form factor {1+beta/2:.3f}, FV ratio/(Bi/2) {ratio/(Bi/2):.3f}, "
              f"QS error {rows[-1]['qs_rel_err_pct']:.2f} %", flush=True)
    return rows


class Axisym2D:
    """Backward-Euler FV on an (r, z) grid; cell-centred; symmetric at r = 0."""

    def __init__(self, Nr=30, Nz=40, R=P.R_o, L=LENGTH):
        self.Nr, self.Nz, self.R, self.L = Nr, Nz, R, L
        self.dr, self.dz = R / Nr, L / Nz
        self.rc = (np.arange(Nr) + 0.5) * self.dr
        self.zc = (np.arange(Nz) + 0.5) * self.dz

    def idx(self, i, j):
        return j * self.Nr + i

    def build(self, k_r, k_z, h, h_end, rho_cp, dt):
        Nr, Nz, dr, dz = self.Nr, self.Nz, self.dr, self.dz
        n = Nr * Nz
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        bnd = np.zeros(n)       # coefficient multiplying T_inf
        vol = np.zeros(n)
        for j in range(Nz):
            for i in range(Nr):
                p = self.idx(i, j)
                r_in, r_out = i * dr, (i + 1) * dr
                V = np.pi * (r_out ** 2 - r_in ** 2) * dz
                vol[p] = V
                # radial faces
                if i > 0:
                    G = k_r * 2 * np.pi * r_in * dz / dr
                    rows += [p]; cols += [self.idx(i - 1, j)]; vals += [G]; diag[p] -= G
                if i < Nr - 1:
                    G = k_r * 2 * np.pi * r_out * dz / dr
                    rows += [p]; cols += [self.idx(i + 1, j)]; vals += [G]; diag[p] -= G
                else:   # side surface: conduction half-cell in series with convection
                    A = 2 * np.pi * self.R * dz
                    G = 1.0 / (1.0 / (h * A) + (dr / 2) / (k_r * A))
                    diag[p] -= G; bnd[p] += G
                A_z = np.pi * (r_out ** 2 - r_in ** 2)
                for dj in (-1, 1):
                    jj = j + dj
                    if 0 <= jj < Nz:
                        G = k_z * A_z / dz
                        rows += [p]; cols += [self.idx(i, jj)]; vals += [G]; diag[p] -= G
                    else:   # end cap
                        G = 1.0 / (1.0 / (h_end * A_z) + (dz / 2) / (k_z * A_z)) if h_end > 0 else 0.0
                        diag[p] -= G; bnd[p] += G
        K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
        C = sp.diags(rho_cp * vol / dt)
        self.vol = vol
        self.lu = spla.splu((C - K).tocsc())
        self.C, self.bnd = C, bnd

    def run(self, q_t, T_inf_t, T0, dt, w_cell, k_r, h):
        """q_t: W/m^3 cell-average heat per step; w_cell: per-cell weights with volume-mean 1.

        Core = mid-height value at the innermost cell centre; surface = mid-height temperature at r = R,
        from the outer half-cell conduction in series with convection (same convention as RadialFV).
        """
        T = np.full(self.Nr * self.Nz, T0)
        jm = self.Nz // 2
        g = k_r / (self.dr / 2)
        core, surf = [], []
        for q, Ti in zip(q_t, T_inf_t):
            rhs = self.C @ T + q * w_cell * self.vol + self.bnd * Ti
            T = self.lu.solve(rhs)
            core.append(0.5 * (T[self.idx(0, jm)] + T[self.idx(0, jm - 1)]))
            To = 0.5 * (T[self.idx(self.Nr - 1, jm)] + T[self.idx(self.Nr - 1, jm - 1)])
            surf.append((g * To + h * Ti) / (g + h))
        return np.array(core), np.array(surf)


def part2():
    rec = Record("2")
    q = rec.q_measured() / P.V_b
    dt = rec.dt
    Bi = H * P.R_o / K_R
    model = Axisym2D()
    cases = {
        "1-D limit (k_z = k_r, adiabatic ends)": dict(k_z=K_R, h_end=0.0, f_tab=0.0),
        "k_z = 25, ends insulated (h_end = 2)": dict(k_z=25.0, h_end=2.0, f_tab=0.0),
        "k_z = 25, ends insulated, 10 % tab heat": dict(k_z=25.0, h_end=2.0, f_tab=0.10),
        "k_z = 25, ends open (h_end = h)": dict(k_z=25.0, h_end=H, f_tab=0.0),
        "k_z = 25, ends open, 20 % tab heat": dict(k_z=25.0, h_end=H, f_tab=0.20),
        "k_z = 2 (poor axial), ends open, 20 % tab heat": dict(k_z=2.0, h_end=H, f_tab=0.20),
    }
    res = {}
    for name, c in cases.items():
        model.build(K_R, c["k_z"], H, c["h_end"], RHO_CP, dt)
        w = np.ones(model.Nr * model.Nz)
        if c["f_tab"] > 0:
            top = model.zc > (LENGTH - 3e-3)
            mask = np.zeros(model.Nr * model.Nz, bool)
            for j in np.where(top)[0]:
                mask[j * model.Nr:(j + 1) * model.Nr] = True
            vt = model.vol[mask].sum() / model.vol.sum()
            w[mask] = (1 - c["f_tab"]) + c["f_tab"] / vt
            w[~mask] = (1 - c["f_tab"])
            w *= model.vol.sum() / (w * model.vol).sum()
        Tc, Ts = model.run(q, rec.T_inf, float(rec.T_s[0]), dt, w, K_R, H)
        rise = Ts - rec.T_inf
        grad = Tc - Ts
        m = rise > 0.5 * rise.max()
        e = (Bi / 2) * rise[m] - grad[m]
        e_core = (Ts + (Bi / 2) * rise) - Tc
        res[name] = {"ratio_over_Bi2": float(np.mean(grad[m] / rise[m]) / (Bi / 2)),
                     "qs_rel_err_pct": float(100 * np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(grad[m] ** 2))),
                     "qs_core_rmse_K": float(np.sqrt(np.mean(e_core ** 2))),
                     "peak_grad_K": float(grad.max()), "peak_rise_K": float(rise.max())}
        print(f"{name:48s} ratio/(Bi/2) {res[name]['ratio_over_Bi2']:.3f}  QS err {res[name]['qs_rel_err_pct']:.2f} %  "
              f"QS core RMSE vs 2-D core {res[name]['qs_core_rmse_K']:.3f} K  peak grad {grad.max():.2f} K", flush=True)
    return res


def main():
    out = {"part1_radial_nonuniform": part1(), "part2_axial_2d": part2()}
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "rev1_heatgen_axial.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
