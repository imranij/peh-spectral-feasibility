#!/usr/bin/env python3
"""
calibration.py -- few-shot calibration of the transfer failure.

Question: when a model trained on two vibration sources meets a third it has
never seen, how many records of the new source must be added to training
before prediction on that source becomes useful?

Protocol, per held-out source S and per seed:
  1. split S's records by record: 25 % are the test set, 75 % the calibration
     pool (GroupShuffleSplit, so no record straddles the two)
  2. for k in K: draw k records from the pool; train the same gradient-boosted
     model as the main benchmark on [all rows of the other two sources] +
     [all rows of the k drawn records]; score R^2 on S's test rows
  3. two references at every k:
       target-only : train on the k records alone (no prior from other sources)
       in-domain   : train on the whole 75 % pool (the ceiling)
Everything is grouped by record, as in the paper. 10 seeds; mean +- SD.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
FEATS = ["rms", "dom_freq_hz", "spec_centroid", "spec_bandwidth", "spec_entropy",
         "f99", "intermittency", "crest", "kurtosis",
         "f_n_hz", "R_over_Ropt", "gamma", "mismatch"]
TARGET = "log10_power"
K = [0, 1, 2, 3, 5, 8, 10, 15, 20]
SEEDS = list(range(10))

def gbt():
    return HistGradientBoostingRegressor(max_iter=400, random_state=0)

df = pd.read_csv(HERE / "dataset_rom.csv")
X, y, g, c = df[FEATS].values, df[TARGET].values, df["sample_id"].values, df["case"].values
rows = []
for held in sorted(df.case.unique()):
    tgt = np.where(c == held)[0]; oth = np.where(c != held)[0]
    for s in SEEDS:
        pool_i, test_i = next(GroupShuffleSplit(1, test_size=.25, random_state=s)
                              .split(X[tgt], y[tgt], g[tgt]))
        pool, test = tgt[pool_i], tgt[test_i]
        pool_recs = np.array(sorted(set(g[pool])))
        rng = np.random.default_rng(100 + s)
        order = rng.permutation(pool_recs)          # one nested draw per seed
        # ceiling: whole pool, target only
        r2_ceiling = r2_score(y[test], gbt().fit(X[pool], y[pool]).predict(X[test]))
        for k in K:
            recs = order[:k]
            add = pool[np.isin(g[pool], recs)]
            tr = np.concatenate([oth, add])
            r2_prior = r2_score(y[test], gbt().fit(X[tr], y[tr]).predict(X[test]))
            r2_only = (r2_score(y[test], gbt().fit(X[add], y[add]).predict(X[test]))
                       if k >= 2 else np.nan)   # one record cannot be fit sensibly
            rows.append(dict(held_out=held, seed=s, k=k, r2_two_sources_plus_k=r2_prior,
                             r2_target_only_k=r2_only, r2_in_domain_ceiling=r2_ceiling,
                             n_pool_records=len(pool_recs)))
        print(f"{held:<13} seed {s} done")
out = pd.DataFrame(rows)
out.to_csv(HERE / "calibration_per_seed.csv", index=False)
agg = (out.groupby(["held_out", "k"])
          .agg(r2_mean=("r2_two_sources_plus_k", "mean"), r2_sd=("r2_two_sources_plus_k", "std"),
               only_mean=("r2_target_only_k", "mean"), only_sd=("r2_target_only_k", "std"),
               ceiling=("r2_in_domain_ceiling", "mean"))
          .reset_index())
agg.to_csv(HERE / "calibration.csv", index=False)
pd.set_option("display.width", 160)
print(agg.round(3).to_string(index=False))
