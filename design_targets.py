#!/usr/bin/env python3
"""
design_targets.py -- what geometry does each source actually require?

Calibrates an Euler-Bernoulli cantilever-with-tip-mass model against the two
designs reported in comsol_cases_summary.csv, then solves for the geometry
needed to place the resonance inside each source's measured energy band.

First-mode natural frequency of a uniform cantilever carrying a tip mass:

    k_eff = 3*E*I / L^3 ,   I = w*t^3/12
    m_eff = 0.2357*rho*w*t*L + m_tip
    f_n   = (1/2pi) * sqrt(k_eff/m_eff)

The bending stiffness E*I of the real (multilayer steel + PZT) beam is not
known independently, so we identify the single lumped constant EI from the
reported COMSOL designs and use it for extrapolation. That is legitimate for
placing the resonance; the electromechanical coupling still has to come from
the re-run.
"""
import numpy as np
from scipy.optimize import brentq

W = 0.010          # beam width, m (stated in the manuscript)
T = 0.0003         # substrate thickness, m (from comsol_cases_summary)
RHO = 7800.0       # steel density, kg/m^3
MTIP = 0.002       # tip mass, kg (from comsol_cases_summary)

# reported COMSOL designs: (length_m, resonance_Hz)
REPORTED = [(0.0398, 77.296), (0.033062, 111.0)]

# target bands: (name, dominant freq, 99%-energy edge, Nyquist of recordings)
TARGETS = [
    ("human_motion", 4.80, 14.7, 51.0),
    ("traffic", 16.92, 40.2, 48.8),
    ("machinery", 21.85, 39.7, 46.3),
]


def f_n(L, EI, m_tip=MTIP, t=T, w=W):
    k = 3.0 * EI / L**3
    m = 0.2357 * RHO * w * t * L + m_tip
    return np.sqrt(k / m) / (2 * np.pi)


def identify_EI():
    """Least-squares EI from the reported (L, f) pairs."""
    def err(EI):
        return sum((f_n(L, EI) - f) ** 2 for L, f in REPORTED)
    lo, hi = 1e-8, 1e2
    grid = np.logspace(np.log10(lo), np.log10(hi), 4000)
    return float(grid[np.argmin([err(e) for e in grid])])


def solve_L(target_f, EI, m_tip=MTIP):
    return brentq(lambda L: f_n(L, EI, m_tip) - target_f, 1e-3, 3.0)


def solve_mtip(target_f, EI, L):
    return brentq(lambda m: f_n(L, EI, m) - target_f, 1e-6, 5.0)


def main():
    EI = identify_EI()
    print(f"Identified lumped bending stiffness EI = {EI:.4e} N*m^2")
    print("Check against the reported COMSOL designs:")
    for L, f in REPORTED:
        print(f"   L = {L*1e3:6.2f} mm -> model {f_n(L, EI):7.2f} Hz   "
              f"(COMSOL {f:7.2f} Hz, error {100*(f_n(L,EI)-f)/f:+.2f} %)")

    print("\n" + "=" * 78)
    print("REQUIRED GEOMETRY TO PLACE THE RESONANCE IN EACH MEASURED BAND")
    print("=" * 78)
    print(f"{'source':<14}{'target f':>10}{'L @ 2g tip':>13}{'L @ 10g':>10}"
          f"{'m_tip @ L=80mm':>17}{'below Nyquist?':>16}")
    print("-" * 78)
    for name, fdom, f99, nyq in TARGETS:
        L2 = solve_L(fdom, EI, 0.002)
        L10 = solve_L(fdom, EI, 0.010)
        try:
            m80 = solve_mtip(fdom, EI, 0.080)
        except ValueError:
            m80 = np.nan
        print(f"{name:<14}{fdom:>9.2f}H{L2*1e3:>12.1f}mm{L10*1e3:>9.1f}mm"
              f"{m80*1e3:>15.1f} g{str(fdom < nyq):>16}")

    print("\nTuning to the upper edge of the energy band instead of the peak")
    print("(a wider, more forgiving target for a real device):")
    print(f"{'source':<14}{'f99':>8}{'L @ 2g tip':>13}{'L @ 10g tip':>14}")
    print("-" * 52)
    for name, fdom, f99, nyq in TARGETS:
        print(f"{name:<14}{f99:>7.1f}H{solve_L(f99, EI, 0.002)*1e3:>12.1f}mm"
              f"{solve_L(f99, EI, 0.010)*1e3:>13.1f}mm")

    print("\nFor reference, the current designs sit at "
          f"{REPORTED[0][1]:.0f} Hz and {REPORTED[1][1]:.0f} Hz.")


if __name__ == "__main__":
    main()
