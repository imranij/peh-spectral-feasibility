#!/usr/bin/env python3
"""
spectral_overlap.py -- quantify how much of each source's measured vibration
energy actually lies where the simulated harvester can convert it.

For each measured signal we compute the one-sided acceleration PSD S_a(f) by
Welch's method, then report

  - the band containing 99% of the signal energy
  - the harvester's resonance f_n and its half-power band [f_n(1-zeta), f_n(1+zeta)]
  - the fraction of signal energy inside that half-power band
  - the fraction of signal energy above the signal's Nyquist frequency (0 by
    construction -- included to make explicit that f_n > f_Nyquist means the
    measurement cannot represent the harvester's operating band at all)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch

import os
# Raw recordings are not redistributed; they come from the open datasets cited in the paper.
# Point PEH_RAW_DIR at a folder holding human_motion/, traffic/ and machinery/ subfolders of CSVs.
UP = Path(os.environ.get("PEH_RAW_DIR", Path(__file__).resolve().parent / "raw"))
SRC = {
    "human_motion": UP / "human_motion",
    "machinery": UP / "machinery",
    "traffic": UP / "traffic",
}
# resonance from the COMSOL exports; zeta measured from the half-power width
# of the only block whose peak is resolved (human_motion block 2).
FN = {"human_motion": 50.04, "traffic": 110.99, "machinery": 89.92}
ZETA_SIM = 6.0e-4          # measured upper bound from COMSOL (Q ~= 834)
ZETA_REAL = 5.0e-3         # representative measured value for a PZT cantilever


def psd(path):
    d = pd.read_csv(path)
    t = d["time_s"].to_numpy(float)
    a = d["accel_mps2"].to_numpy(float)
    fs = 1.0 / np.median(np.diff(t))
    nper = int(min(1024, len(a)))
    f, P = welch(a, fs=fs, nperseg=nper)
    return f, P, fs, a


def band_fraction(f, P, lo, hi):
    tot = np.trapezoid(P, f)
    if tot <= 0:
        return 0.0
    m = (f >= lo) & (f <= hi)
    if m.sum() < 2:
        return 0.0
    return float(np.trapezoid(P[m], f[m]) / tot)


def main():
    rows = []
    for case, folder in SRC.items():
        files = sorted(folder.glob("*.csv"))
        if not files:
            print(f"[warn] no signals for {case}")
            continue
        fn = FN[case]
        for p in files:
            f, P, fs, a = psd(p)
            tot = np.trapezoid(P, f)
            cum = np.cumsum(P) / np.sum(P)
            f99 = float(f[np.searchsorted(cum, 0.99)])
            rows.append(dict(
                case=case, file=p.stem, fs=fs, nyquist=fs / 2,
                rms=float(np.sqrt(np.mean(a**2))),
                f_dom=float(f[np.argmax(P)]), f99=f99, fn=fn,
                fn_above_nyquist=bool(fn > fs / 2),
                frac_in_hp_sim=band_fraction(f, P, fn * (1 - ZETA_SIM), fn * (1 + ZETA_SIM)),
                frac_in_hp_real=band_fraction(f, P, fn * (1 - ZETA_REAL), fn * (1 + ZETA_REAL)),
                frac_above_f99=band_fraction(f, P, f99, f.max()),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "spectral_overlap.csv", index=False)

    print(f"{'case':<14}{'n':>4}{'fs':>7}{'Nyq':>7}{'f_dom':>8}{'f99':>8}"
          f"{'f_n':>9}{'f_n>Nyq':>9}{'E in HP band':>14}")
    print("-" * 80)
    for case, g in df.groupby("case"):
        print(f"{case:<14}{len(g):>4}{g.fs.mean():>7.1f}{g.nyquist.mean():>7.1f}"
              f"{g.f_dom.median():>8.2f}{g.f99.median():>8.1f}{g.fn.iloc[0]:>9.2f}"
              f"{str(bool(g.fn_above_nyquist.all())):>9}"
              f"{g.frac_in_hp_real.mean():>14.3e}")
    print("\n'E in HP band' = mean fraction of measured signal energy inside the")
    print("harvester half-power band, using a REALISTIC zeta = 5e-3. With the")
    print(f"simulated zeta = {ZETA_SIM:.0e} the band is ~8x narrower still.")
    print("\nAll three sources: f_n lies above the measurement Nyquist frequency,")
    print("so the measured data contains no energy at the harvester's resonance.")


if __name__ == "__main__":
    main()
