#!/usr/bin/env python3
"""
feasibility.py -- measurement-driven tuning feasibility analysis.

Two results, both computable from the measured signals alone:

(1) TUNING PENALTY. For a lightly damped SDOF harvester under broadband base
    excitation, the mean harvested power is governed by the excitation PSD
    evaluated AT the resonance:  E[P] ~ S_a(f_n) * (device constants).
    (The bandwidth narrows as 1/zeta while the gain grows as 1/zeta, so the
    two cancel and only the local spectral density survives.)
    The penalty for mistuning is therefore the PSD ratio S_a(f_opt)/S_a(f_n),
    which needs no device parameters at all.

    Because the current designs resonate at or above the recording Nyquist
    frequency, S_a(f_n) is not measurable there; the ratio computed at the
    highest measurable frequency is therefore a LOWER BOUND on the true
    penalty (real spectra keep rolling off above Nyquist).

(2) GEOMETRY MAP. First-mode resonance of a cantilever with tip mass over the
    (length, tip mass) plane, with the three source bands contoured and a
    wearable-scale feasibility box marked.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

import os
# Raw recordings are not redistributed; they come from the open datasets cited in the paper.
# Point PEH_RAW_DIR at a folder holding human_motion/, traffic/ and machinery/ subfolders of CSVs.
UP = Path(os.environ.get("PEH_RAW_DIR", Path(__file__).resolve().parent / "raw"))
OUT = Path(__file__).resolve().parent.parent
SRC = {
    "Human motion": (UP / "human_motion", 50.04, "#1F5C8B"),
    "Traffic": (UP / "traffic", 110.99, "#2E8B57"),
    "Machinery": (UP / "machinery", 89.92, "#B8860B"),
}

# cantilever constants (from comsol_cases_summary.csv / manuscript)
W, T, RHO = 0.010, 0.0003, 7800.0
EI = 1.2134e-02          # identified in design_targets.py, +-5% vs COMSOL

plt.rcParams.update({"savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def f_n(L, m_tip):
    k = 3.0 * EI / L**3
    m = 0.2357 * RHO * W * T * L + m_tip
    return np.sqrt(k / m) / (2 * np.pi)


def mean_psd(folder, grid):
    acc, fss = [], []
    for p in sorted(folder.glob("*.csv")):
        d = pd.read_csv(p)
        t = d["time_s"].to_numpy(float); a = d["accel_mps2"].to_numpy(float)
        fs = 1.0 / np.median(np.diff(t)); fss.append(fs)
        f, P = welch(a, fs=fs, nperseg=int(min(1024, len(a))))
        acc.append(np.interp(grid, f, P, left=np.nan, right=np.nan))
    A = np.array(acc)
    return np.nanmean(A, axis=0), float(np.median(fss))


def main():
    grid = np.logspace(np.log10(0.5), np.log10(60), 700)
    rows = []
    psds = {}
    for name, (folder, fn_cur, col) in SRC.items():
        S, fs = mean_psd(folder, grid)
        psds[name] = (S, fs, fn_cur, col)
        ok = np.isfinite(S) & (S > 0)
        f_ok, S_ok = grid[ok], S[ok]
        f_peak = f_ok[np.argmax(S_ok)]
        S_peak = S_ok.max()
        f_edge = f_ok.max()                       # highest measurable frequency
        S_edge = S_ok[-1]
        # PSD at the current design frequency, if measurable at all
        measurable = fn_cur <= f_ok.max()
        S_at_design = float(np.interp(fn_cur, f_ok, S_ok)) if measurable else np.nan
        penalty = S_peak / S_at_design if measurable else S_peak / S_edge
        rows.append(dict(source=name, f_peak=f_peak, S_peak=S_peak,
                         f_design=fn_cur, nyquist=fs / 2,
                         measurable_at_design=measurable,
                         S_at_design=S_at_design, f_edge=f_edge, S_edge=S_edge,
                         penalty=penalty, bound="exact" if measurable else "lower bound"))

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "tuning_penalty.csv", index=False)

    print("TUNING PENALTY  --  factor by which harvested power is reduced by")
    print("resonating at the as-designed frequency instead of the spectral peak")
    print(f"\n{'source':<14}{'f_peak':>8}{'f_design':>10}{'Nyquist':>9}"
          f"{'penalty':>13}{'type':>14}")
    print("-" * 70)
    for r in rows:
        print(f"{r['source']:<14}{r['f_peak']:>8.2f}{r['f_design']:>10.2f}"
              f"{r['nyquist']:>9.1f}{r['penalty']:>13.3g}x{r['bound']:>13}")
    print("\nFor traffic and machinery the design frequency lies above the")
    print("recording Nyquist limit, so the true penalty is larger than shown.")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.1))

    a = ax[0]
    for name, (S, fs, fn_cur, col) in psds.items():
        ok = np.isfinite(S) & (S > 0)
        a.plot(grid[ok], S[ok], color=col, lw=1.8, label=name)
        pk = grid[ok][np.argmax(S[ok])]
        a.scatter([pk], [S[ok].max()], color=col, s=30, zorder=5)
        a.axvline(fs / 2, color=col, lw=0.8, ls=":", alpha=.7)
    for name, (S, fs, fn_cur, col) in psds.items():
        a.axvline(fn_cur, color=col, lw=1.4, ls="--", alpha=.9)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("Frequency (Hz)"); a.set_ylabel("Mean acceleration PSD (m$^2$s$^{-4}$/Hz)")
    a.set_title("Where the energy is (solid) vs where the harvester listens (dashed)")
    a.grid(alpha=.13, which="both"); a.legend(loc="lower left", fontsize=8)

    b = ax[1]
    L = np.linspace(0.02, 0.30, 400)
    M = np.logspace(np.log10(5e-4), np.log10(0.1), 400)
    LL, MM = np.meshgrid(L, M)
    F = f_n(LL, MM)
    cs = b.contour(LL * 1e3, MM * 1e3, F, levels=[4.8, 16.9, 21.9, 50, 111],
                   colors=["#1F5C8B", "#2E8B57", "#B8860B", "0.45", "0.45"],
                   linewidths=[2, 2, 2, 1, 1])
    b.clabel(cs, fmt=lambda v: f"{v:g} Hz", fontsize=7.5)
    b.add_patch(plt.Rectangle((20, 0.5), 40, 9.5, facecolor="#C2402B", alpha=.13,
                              edgecolor="#C2402B", lw=1.2, ls="--"))
    b.text(24, 6.5, "wearable-scale\nenvelope", fontsize=7.5, color="#C2402B")
    b.set_yscale("log")
    b.set_xlabel("Beam length $L$ (mm)"); b.set_ylabel("Tip mass $m_t$ (g)")
    b.set_title("Achievable resonance over the design plane")
    b.grid(alpha=.13, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "fig_feasibility.png", bbox_inches="tight")
    print(f"\nwrote {OUT/'fig_feasibility.png'}  and  tuning_penalty.csv")


if __name__ == "__main__":
    main()
