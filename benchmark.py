#!/usr/bin/env python3
"""
benchmark.py -- the rebuilt ML benchmark on physics-based labels.

Addresses, in one pass, the reviewer requests that were previously
uncomputable:

  * grouped splits by signal, so no signal appears in both train and test
  * a ZERO-PARAMETER analytical baseline that the learned models must beat
  * permutation feature importance with a physical reading
  * conformal prediction intervals with empirical coverage
  * leave-one-source-out transfer
  * repeated seeds with a paired significance test
  * error reported in log AND linear space

The analytical baseline is the narrowband random-vibration result: near
resonance the transfer function is sharply peaked, so

    E[P] ~ S_a(f_n) * C(f_n, R),    C(f_n,R) = INT |H_v(f;f_n,R)|^2 df / R

where C depends only on declared device constants. It has no fitted
parameters at all -- it is physics, evaluated per row.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_approximation import Nystroem

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset_rom.csv"
SEEDS = list(range(20))

FEATS = ["rms", "dom_freq_hz", "spec_centroid", "spec_bandwidth", "spec_entropy",
         "f99", "intermittency", "crest", "kurtosis",
         "f_n_hz", "R_over_Ropt", "gamma", "mismatch"]
TARGET = "log10_power"


def models():
    return {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "poly_ridge_d3": make_pipeline(StandardScaler(),
                                       PolynomialFeatures(3, include_bias=False),
                                       Ridge(alpha=10.0)),
        # RBF-kernel SVR via Nystroem approximation: exact SVR is O(n^2) and
        # does not scale to this dataset. 500 components, documented as such.
        "svr_rbf_nystroem": make_pipeline(
            StandardScaler(), Nystroem(gamma=0.1, n_components=500, random_state=0),
            LinearSVR(C=10.0, epsilon=0.05, max_iter=5000, dual="auto")),
        "random_forest": RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                               random_state=0),
        "gbrt": HistGradientBoostingRegressor(max_iter=400, random_state=0),
        "mlp": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(128, 64),
                                          max_iter=400, early_stopping=True,
                                          random_state=0)),
    }


def analytic_baseline(df):
    """Zero-parameter physics predictor: log10( S_a(f_n) * C(f_n,R) )."""
    from build_dataset import H_v2, CP
    pred = np.empty(len(df))
    grid = np.logspace(np.log10(0.2), np.log10(300), 3000)
    cache = {}
    for i, (fn, rr) in enumerate(zip(df.f_n_hz.values, df.R_over_Ropt.values)):
        key = (fn, rr)
        if key not in cache:
            R = rr / (2 * np.pi * fn * CP)
            cache[key] = np.trapezoid(H_v2(grid, fn, R), grid) / R
        pred[i] = cache[key]
    return np.log10(np.maximum(pred * df.S_at_fn.values, 1e-30))


def main():
    df = pd.read_csv(DATA)
    X, y, g = df[FEATS].values, df[TARGET].values, df["sample_id"].values
    print(f"rows {len(df)}   signals {df.sample_id.nunique()}   "
          f"sources {df.case.nunique()}")

    # ---------------- analytical baseline -----------------------------------
    ab = analytic_baseline(df)
    ok = np.isfinite(ab)
    print("\n=== ZERO-PARAMETER ANALYTICAL BASELINE ===")
    print(f"  R2   = {r2_score(y[ok], ab[ok]): .4f}")
    print(f"  RMSE = {mean_squared_error(y[ok], ab[ok])**.5: .4f}  (log10 W)")

    # ---------------- repeated grouped splits -------------------------------
    print(f"\n=== MODELS, {len(SEEDS)} grouped splits (by signal) ===")
    per_seed = {k: [] for k in models()}
    for s in SEEDS:
        tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=s)
                      .split(X, y, g))
        for name, m in models().items():
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            per_seed[name].append(dict(
                seed=s, r2=r2_score(y[te], p),
                rmse=mean_squared_error(y[te], p) ** .5,
                mae=mean_absolute_error(y[te], p),
                # linear-space relative error (median |P_hat/P - 1|)
                relerr=float(np.median(np.abs(10 ** (p - y[te]) - 1))),
            ))
    agg = []
    for name, rs in per_seed.items():
        d = pd.DataFrame(rs)
        agg.append(dict(model=name, r2=d.r2.mean(), r2_sd=d.r2.std(),
                        rmse=d.rmse.mean(), rmse_sd=d.rmse.std(),
                        mae=d.mae.mean(), median_rel_err=d.relerr.mean()))
    agg = pd.DataFrame(agg).sort_values("rmse")
    print(f"{'model':<15}{'R2':>9}{'+-':>7}{'RMSE':>9}{'+-':>7}"
          f"{'MAE':>8}{'med.rel.err':>13}")
    print("-" * 70)
    for _, r in agg.iterrows():
        print(f"{r.model:<15}{r.r2:>9.4f}{r.r2_sd:>7.4f}{r.rmse:>9.4f}"
              f"{r.rmse_sd:>7.4f}{r.mae:>8.4f}{r.median_rel_err:>12.1%}")
    agg.to_csv(HERE / "benchmark_models.csv", index=False)
    pd.concat([pd.DataFrame(v).assign(model=k) for k, v in per_seed.items()]) \
        .to_csv(HERE / "benchmark_per_seed.csv", index=False)

    # paired test, top two
    top2 = agg.model.iloc[:2].tolist()
    a = [r["rmse"] for r in per_seed[top2[0]]]
    b = [r["rmse"] for r in per_seed[top2[1]]]
    st, pv = wilcoxon(a, b)
    print(f"\npaired Wilcoxon, {top2[0]} vs {top2[1]}: p = {pv:.4g}"
          f"  -> {'significant' if pv < .05 else 'NOT significant'}")

    # ---------------- permutation importance --------------------------------
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=0)
                  .split(X, y, g))
    best = models()[agg.model.iloc[0]].fit(X[tr], y[tr])
    pi = permutation_importance(best, X[te], y[te], n_repeats=10,
                                random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATS, "importance": pi.importances_mean,
                        "sd": pi.importances_std}).sort_values(
        "importance", ascending=False)
    print(f"\n=== PERMUTATION IMPORTANCE ({agg.model.iloc[0]}) ===")
    for _, r in imp.iterrows():
        print(f"   {r.feature:<16}{r.importance:>9.4f} +- {r.sd:.4f}")
    imp.to_csv(HERE / "feature_importance.csv", index=False)

    # ---------------- conformal intervals -----------------------------------
    tr2, cal = next(GroupShuffleSplit(1, test_size=.3, random_state=1)
                    .split(X[tr], y[tr], g[tr]))
    m = models()[agg.model.iloc[0]].fit(X[tr][tr2], y[tr][tr2])
    resid = np.abs(y[tr][cal] - m.predict(X[tr][cal]))
    q = np.quantile(resid, 0.90)
    pred = m.predict(X[te])
    cov = float(np.mean(np.abs(y[te] - pred) <= q))
    print(f"\n=== CONFORMAL PREDICTION (target 90%) ===")
    print(f"   half-width {q:.3f} log10 W   empirical coverage {cov:.1%}")

    # ---------------- leave-one-source-out transfer -------------------------
    print("\n=== LEAVE-ONE-SOURCE-OUT TRANSFER ===")
    rows = []
    for held in df.case.unique():
        m_tr = df.case.values != held
        mdl = models()[agg.model.iloc[0]].fit(X[m_tr], y[m_tr])
        p = mdl.predict(X[~m_tr])
        r2o = r2_score(y[~m_tr], p)
        # in-domain reference for the same held-out source
        idx = np.where(~m_tr)[0]
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=.25, random_state=0)
                          .split(X[idx], y[idx], g[idx]))
        mdl2 = models()[agg.model.iloc[0]].fit(X[idx][tr_i], y[idx][tr_i])
        r2i = r2_score(y[idx][te_i], mdl2.predict(X[idx][te_i]))
        rows.append(dict(held_out=held, r2_transfer=r2o, r2_in_domain=r2i,
                         drop=r2i - r2o))
        print(f"   hold out {held:<14} transfer R2={r2o:>7.4f}   "
              f"in-domain R2={r2i:>7.4f}   drop={r2i-r2o:>7.4f}")
    pd.DataFrame(rows).to_csv(HERE / "transfer.csv", index=False)

    json.dump(dict(n_rows=len(df), n_signals=int(df.sample_id.nunique()),
                   seeds=len(SEEDS), features=FEATS,
                   analytic_r2=float(r2_score(y[ok], ab[ok])),
                   best_model=agg.model.iloc[0]),
              open(HERE / "benchmark_manifest.json", "w"), indent=2)
    print("\nwrote benchmark_models.csv, benchmark_per_seed.csv, "
          "feature_importance.csv, transfer.csv")


if __name__ == "__main__":
    main()
