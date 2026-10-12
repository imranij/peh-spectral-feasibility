#!/usr/bin/env python3
"""
build_dataset.py -- physics-based dataset construction.

Replaces the Cartesian cross-join of the original pipeline. Each measured
signal is pushed through the reduced-order harvester model, so the label for a
row is determined by THAT signal's own spectrum:

    E[P] = (1/R) * INT |H_v(f; f_n, zeta, R)|^2 S_a(f) df

evaluated per signal, over a grid of harvester tunings f_n and loads R.

Device constants are literature-representative for a PZT bimorph rather than
identified from the COMSOL exports, because the exports' unresolved resonance
makes zeta and C_p unidentifiable (see FINDINGS_pipeline_audit.md, Finding 8).
They are declared here, reported in the manuscript, and carried through a
sensitivity study. Every comparative result -- model ranking, R^2, feature
importance, transfer -- is invariant to the overall device scale.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

import os
# Raw recordings are not redistributed; they come from the open datasets cited in the paper.
# Point PEH_RAW_DIR at a folder holding human_motion/, traffic/ and machinery/ subfolders of CSVs.
UP = Path(os.environ.get("PEH_RAW_DIR", Path(__file__).resolve().parent / "raw"))
OUT = Path(__file__).resolve().parent
SRC = {
    "human_motion": UP / "human_motion",
    "traffic": UP / "traffic",
    "machinery": UP / "machinery",
}
F_DESIGN = {"human_motion": 50.04, "traffic": 110.99, "machinery": 89.92}

# --- declared device constants (literature-representative PZT bimorph) ------
ZETA = 5.0e-3          # Q = 100; sensitivity run at 2.5e-3 and 1.0e-2
KAPPA2 = 0.10          # dimensionless electromechanical coupling
CP = 50e-9             # piezoelectric capacitance, F
M_EFF = 2.5e-3         # effective modal mass, kg

# --- operating grid ---------------------------------------------------------
FN_GRID = np.logspace(np.log10(3.0), np.log10(35.0), 20)    # harvester tunings
BAND_FRAC = 0.8   # only keep tunings the recording can actually resolve
R_RATIO = np.logspace(-2, 2, 9)                              # R / R_opt(f_n)


def signal_psd(path):
    d = pd.read_csv(path)
    t = d["time_s"].to_numpy(float)
    a = d["accel_mps2"].to_numpy(float)
    fs = 1.0 / np.median(np.diff(t))
    f, S = welch(a, fs=fs, nperseg=int(min(1024, len(a))))
    return f, S, fs, a


def H_v2(f, fn, R):
    """|H_v|^2 for the SDOF piezoelectric harvester (unit base acceleration)."""
    w = 2 * np.pi * f
    wn = 2 * np.pi * fn
    k = M_EFF * wn**2
    theta = np.sqrt(KAPPA2 * k * CP)
    c = 2 * ZETA * M_EFF * wn
    num = 1j * w * theta * M_EFF
    den = ((k - w**2 * M_EFF + 1j * w * c) * (1j * w * CP + 1.0 / R)
           + 1j * w * theta**2)
    return np.abs(num / den) ** 2


def features(f, S, fs, a):
    tot = np.trapezoid(S, f)
    Sn = S / tot
    cum = np.cumsum(S) / np.sum(S)
    ent = -np.sum(Sn / Sn.sum() * np.log(Sn / Sn.sum() + 1e-16)) / np.log(len(Sn))
    centroid = float(np.trapezoid(f * S, f) / tot)
    bw = float(np.sqrt(np.trapezoid((f - centroid) ** 2 * S, f) / tot))
    rms = float(np.sqrt(np.mean(a**2)))
    return dict(
        rms=rms,
        dom_freq_hz=float(f[np.argmax(S)]),
        spec_centroid=centroid,
        spec_bandwidth=bw,
        spec_entropy=float(ent),
        f99=float(f[np.searchsorted(cum, 0.99)]),
        intermittency=float(np.mean(np.abs(a) > 2 * np.std(a))),
        crest=float(np.max(np.abs(a)) / (rms + 1e-16)),
        kurtosis=float(kurtosis(a)),
        fs_hz=fs,
    )


def main():
    rows = []
    for case, folder in SRC.items():
        files = sorted(folder.glob("*.csv"))
        print(f"{case}: {len(files)} signals")
        for p in files:
            f, S, fs, a = signal_psd(p)
            feat = features(f, S, fs, a)
            band = f > 0
            fb, Sb = f[band], S[band]
            sig_energy = np.trapezoid(Sb, fb)

            for fn in FN_GRID:
                if fn > BAND_FRAC * (fs / 2.0):
                    continue   # resonance beyond what this recording resolves
                R_opt = 1.0 / (2 * np.pi * fn * CP)
                # spectral overlap: normalised transfer shape vs this signal
                Hn = H_v2(fb, fn, R_opt)
                Hn = Hn / Hn.max()
                gamma = float(np.trapezoid(Hn * Sb, fb) / sig_energy)
                # PSD at the resonance -- the quantity that governs power
                S_at_fn = float(np.interp(fn, fb, Sb, left=0.0, right=0.0))

                for rr in R_RATIO:
                    R = R_opt * rr
                    P = float(np.trapezoid(H_v2(fb, fn, R) * Sb, fb) / R)
                    rows.append(dict(
                        case=case, sample_id=p.stem, **feat,
                        f_n_hz=fn, R_ohm=R, R_over_Ropt=rr,
                        gamma=gamma, S_at_fn=S_at_fn,
                        mismatch=float(feat["dom_freq_hz"] / fn),
                        is_design_tuning=bool(abs(fn - F_DESIGN[case]) < 1e-9),
                        power_W=P,
                    ))

    df = pd.DataFrame(rows)
    df["log10_power"] = np.log10(np.maximum(df["power_W"], 1e-30))
    df.to_csv(OUT / "dataset_rom.csv", index=False)

    print(f"\nrows: {len(df)}   signals: {df.sample_id.nunique()}   "
          f"operating points: {df.groupby(['f_n_hz','R_over_Ropt']).ngroups}")
    print(f"power range: {df.power_W.min():.3e} .. {df.power_W.max():.3e} W")

    print("\n--- THE CHECK THAT FAILED BEFORE ---")
    print("correlation of each vibration feature with log10(power):")
    for c in ["rms", "dom_freq_hz", "spec_entropy", "intermittency",
              "spec_bandwidth", "crest", "gamma"]:
        r = np.corrcoef(df[c], df["log10_power"])[0, 1]
        print(f"   {c:<16} {r:+.4f}")
    print("\n(all four original features were exactly 0.000000 in the old dataset)")
    print(f"\nwrote {OUT/'dataset_rom.csv'}")


if __name__ == "__main__":
    main()
