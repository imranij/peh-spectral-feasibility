# peh-spectral-feasibility

Code and processed data accompanying the manuscript
*"Spectral Feasibility and Performance Prediction of Piezoelectric Energy
Harvesters under Measured IoT Vibrations"* (under review).

## Contents

| File | Purpose |
|---|---|
| `fit_rom.py` | identifies the reduced-order model; writes `rom_parameters.json` |
| `build_dataset.py` | generates the labelled dataset `dataset_rom.csv` |
| `spectral_overlap.py`, `feasibility.py`, `design_targets.py` | spectral and tuning analyses |
| `benchmark.py` | model benchmark; writes `benchmark_*.csv`, `feature_importance.csv`, `transfer.csv` |
| `calibration.py` | calibration analysis; writes `calibration*.csv` |
| `*.csv`, `*.json` | processed data and results reported in the manuscript |

Raw vibration recordings and finite-element exports are not redistributed.
The recordings are available from the open datasets cited in the manuscript.

## Reproducing the reported results

```
pip install numpy pandas scipy scikit-learn
python benchmark.py
python calibration.py
```

Both scripts read `dataset_rom.csv` from this folder.

## Licence

Copyright © 2026 the authors. All rights reserved. The material is provided
solely so that the results in the manuscript can be checked and reproduced.
No licence to reuse, modify or redistribute is granted.
