#!/usr/bin/env python3
"""
fit_rom.py -- identify a reduced-order electromechanical model (ROM) of the
piezoelectric harvester from the COMSOL exports, per case.

Model (standard SDOF base-excited piezoelectric harvester, e.g. duToit 2005;
Erturk & Inman 2008), in the frequency domain with base acceleration A:

    m x_ddot + c x_dot + k x - theta*v = -m*a(t)
    theta*x_dot + Cp*v_dot + v/R       = 0

giving the voltage frequency response function

    H_v(w;R) = V/A = j*w*theta / [ (wn^2 - w^2 + 2j*zeta*wn*w)(j*w*Cp + 1/R)
                                   + j*w*gamma ]
    with gamma = theta^2/m.

The COMSOL exports give |V| only, at an unknown base-acceleration amplitude,
so theta and A are degenerate: we identify the product as an overall gain and
carry per-block gains for the three excitation levels present in each export.

Identified per case: wn, zeta, Cp, gamma, R_fr (load used during the frequency
sweep), and per-block gains. Everything needed for H_v(f;R) up to one overall
excitation scale, which cancels in every comparative result.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import os
# Finite-element exports (<case>_frequency_response.csv, <case>_load_sweep.csv) are not redistributed.
ROOT = Path(os.environ.get("PEH_FE_DIR", Path(__file__).resolve().parent / "fe_exports"))
OUT = Path(__file__).resolve().parent
CASES = ["human_motion", "traffic", "machinery"]


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def H_v(f, R, wn, zeta, Cp, gamma):
    """Voltage FRF magnitude |V/A| (up to the overall gain)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    num = 1j * w
    den = ((wn**2 - w**2 + 2j * zeta * wn * w) * (1j * w * Cp + 1.0 / R)
           + 1j * w * gamma)
    return np.abs(num / den)


def power(f, R, wn, zeta, Cp, gamma, gain):
    """Harvested power for unit-amplitude base acceleration: |V|^2/(2R)."""
    V = gain * H_v(f, R, wn, zeta, Cp, gamma)
    return V**2 / (2.0 * R)


# ----------------------------------------------------------------------------
# data loading -- split each export into its interleaved parametric blocks
# ----------------------------------------------------------------------------
def _clean(df):
    df = df.dropna(axis=1, how="all").copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def load_blocks(path, keycol):
    """Split a COMSOL export into blocks, detected where the sweep variable resets."""
    d = _clean(pd.read_csv(path))
    kc = [c for c in d.columns if keycol in c.lower()][0]
    vc = [c for c in d.columns if "voltage" in c.lower()][0]
    pc = [c for c in d.columns if "power" in c.lower()][0]
    x = pd.to_numeric(d[kc], errors="coerce").to_numpy()
    v = pd.to_numeric(d[vc], errors="coerce").to_numpy()
    p = pd.to_numeric(d[pc], errors="coerce").to_numpy() * 1e-3  # mW -> W
    ok = np.isfinite(x) & np.isfinite(v)
    x, v, p = x[ok], v[ok], p[ok]
    cuts = np.where(np.diff(x) < 0)[0]
    starts = [0] + list(cuts + 1) + [len(x)]
    return [(x[a:b], v[a:b], p[a:b]) for a, b in zip(starts[:-1], starts[1:]) if b > a]


# ----------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------
def fit_case(case):
    fr_blocks = load_blocks(ROOT / f"{case}_frequency_response.csv", "freq")
    ls_blocks = load_blocks(ROOT / f"{case}_load_sweep.csv", "r_load")
    f_ls = float(_clean(pd.read_csv(ROOT / f"{case}_load_sweep.csv"))
                 .iloc[:, 0].astype(float).iloc[0])

    n_fr, n_ls = len(fr_blocks), len(ls_blocks)

    # crude starting point from the data itself
    f0, v0, _ = fr_blocks[0]
    fn0 = f0[np.argmax(v0)]
    # half-power bandwidth -> zeta
    vmax = v0.max()
    above = f0[v0 >= vmax / np.sqrt(2)]
    zeta0 = (above.max() - above.min()) / (2 * fn0) if len(above) > 2 else 0.02
    zeta0 = float(np.clip(zeta0, 1e-4, 0.2))
    Rls, Vls, _ = ls_blocks[0]
    Ropt0 = Rls[np.argmax(Vls**2 / (2 * Rls))]
    Cp0 = 1.0 / (2 * np.pi * f_ls * Ropt0)

    # parameters: log wn, log zeta, log Cp, log gamma, log R_fr, log gains
    p0 = np.concatenate([
        [np.log(2 * np.pi * fn0), np.log(zeta0), np.log(Cp0), np.log(1e-3),
         np.log(1e4)],
        np.zeros(n_fr), np.zeros(n_ls),
    ])
    # seed gains from peak magnitudes
    for i, (_, v, _) in enumerate(fr_blocks):
        p0[5 + i] = np.log(max(v.max(), 1e-12))
    for i, (R, v, _) in enumerate(ls_blocks):
        p0[5 + n_fr + i] = np.log(max(v.max(), 1e-12))

    def unpack(p):
        wn, zeta, Cp, gamma, Rfr = np.exp(p[:5])
        gfr = np.exp(p[5:5 + n_fr])
        gls = np.exp(p[5 + n_fr:5 + n_fr + n_ls])
        return wn, zeta, Cp, gamma, Rfr, gfr, gls

    def resid(p):
        wn, zeta, Cp, gamma, Rfr, gfr, gls = unpack(p)
        r = []
        for i, (f, v, _) in enumerate(fr_blocks):
            mod = gfr[i] * H_v(f, Rfr, wn, zeta, Cp, gamma)
            r.append(np.log(np.maximum(mod, 1e-30)) - np.log(np.maximum(v, 1e-30)))
        for i, (R, v, _) in enumerate(ls_blocks):
            mod = gls[i] * H_v(np.full_like(R, f_ls), R, wn, zeta, Cp, gamma)
            r.append(np.log(np.maximum(mod, 1e-30)) - np.log(np.maximum(v, 1e-30)))
        return np.concatenate(r)

    sol = least_squares(resid, p0, method="trf", max_nfev=4000, ftol=1e-12,
                        xtol=1e-12, x_scale="jac")
    wn, zeta, Cp, gamma, Rfr, gfr, gls = unpack(sol.x)

    # goodness of fit, in log-magnitude space
    def r2(obs, mod):
        o, m = np.log(np.maximum(obs, 1e-30)), np.log(np.maximum(mod, 1e-30))
        return 1 - np.sum((o - m) ** 2) / np.sum((o - o.mean()) ** 2)

    fr_fit, ls_fit = [], []
    for i, (f, v, _) in enumerate(fr_blocks):
        fr_fit.append(r2(v, gfr[i] * H_v(f, Rfr, wn, zeta, Cp, gamma)))
    for i, (R, v, _) in enumerate(ls_blocks):
        ls_fit.append(r2(v, gls[i] * H_v(np.full_like(R, f_ls), R, wn, zeta, Cp, gamma)))

    kappa2 = gamma / (Cp * wn**2)  # dimensionless electromechanical coupling

    return dict(
        case=case, fn_hz=wn / (2 * np.pi), zeta=zeta, Q=1 / (2 * zeta), Cp_F=Cp,
        gamma=gamma, kappa2=kappa2, R_fr_ohm=Rfr, f_loadsweep_hz=f_ls,
        gains_fr=gfr.tolist(), gains_ls=gls.tolist(),
        excitation_ratios_fr=(gfr / gfr[0]).tolist(),
        excitation_ratios_ls=(gls / gls[0]).tolist(),
        r2_fr=fr_fit, r2_ls=ls_fit,
        R_opt_theory_ohm=1.0 / (2 * np.pi * f_ls * Cp),
        n_points=int(sum(len(b[0]) for b in fr_blocks) + sum(len(b[0]) for b in ls_blocks)),
    )


def main():
    rows = []
    for c in CASES:
        try:
            rows.append(fit_case(c))
        except Exception as e:
            print(f"[FAIL] {c}: {e}")
    print(f"{'case':<14}{'fn (Hz)':>10}{'zeta':>10}{'Q':>8}{'Cp (nF)':>11}"
          f"{'kappa^2':>10}{'R_fr':>11}{'min R2':>9}")
    print("-" * 84)
    for r in rows:
        mn = min(r["r2_fr"] + r["r2_ls"])
        print(f"{r['case']:<14}{r['fn_hz']:>10.3f}{r['zeta']:>10.5f}{r['Q']:>8.1f}"
              f"{r['Cp_F']*1e9:>11.3f}{r['kappa2']:>10.4f}{r['R_fr_ohm']:>11.1f}{mn:>9.4f}")
    print()
    for r in rows:
        print(f"{r['case']}: excitation ratios  FR={np.round(r['excitation_ratios_fr'],4).tolist()}"
              f"  LS={np.round(r['excitation_ratios_ls'],4).tolist()}")
        print(f"{'':<{len(r['case'])+2}}R2 per block  FR={np.round(r['r2_fr'],5).tolist()}"
              f"  LS={np.round(r['r2_ls'],5).tolist()}")
    with open(OUT / "rom_parameters.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {OUT/'rom_parameters.json'}")


if __name__ == "__main__":
    main()
