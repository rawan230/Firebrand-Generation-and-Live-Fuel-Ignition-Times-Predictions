"""
Physics-structured model for firebrand generation time (FGT) of wooden dowels
in a heated crossflow.

Scaling law (multiplicative, one factor per physical driver):

    FGT = A_s * (d / 2 mm)^n * f_M^[moist] * f_H^[low heat] * f_W^[low wind]

    A_s : generation time of a dry 2 mm dowel of species s at high heat and wind (s)
    n   : diameter exponent. Convective heating of a cylinder gives n = 1.5
          (Nu ~ Re^0.5), diffusion-limited burn-through gives n = 2.
    f_M, f_H, f_W : multipliers for 15 % moisture, low heat and low wind.

The law is linear in log space, so it is fitted by ordinary least squares, which
also gives confidence intervals. Heat and wind are only recorded as low/high in
FGT_Edited.csv, so they enter as two-level factors, not as temperatures or speeds.

Each condition is stored as three rows (mean, mean + spread, mean - spread); the
script collapses them to one row per condition (63 conditions).
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from igt_graybox import make_gp, metrics

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fgt_physics_outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42
SPECIES = ["Douglas-fir", "Juniper", "Ponderosa", "White oak"]


def load_conditions():
    raw = pd.read_csv(os.path.join(HERE, "FGT_Edited.csv"))
    raw.columns = ["d", "species", "mc", "heat", "wind", "mean_time", "t"]
    keys = ["d", "species", "mc", "heat", "wind"]
    g = raw.groupby(keys, sort=False)["t"]
    assert (g.size() == 3).all(), "expected three rows per condition"
    df = g.mean().reset_index()
    df["sd"] = ((g.max() - g.min()) / 2).values
    df["logd"] = np.log(df["d"] / 2.0)
    df["moist"] = (df["mc"] == "15%").astype(float)
    df["lowheat"] = (df["heat"] == "low").astype(float)
    df["lowwind"] = (df["wind"] == "low").astype(float)
    return df


# ----------------------------------------------------------------------------
# Physics scaling law (log-linear)
# ----------------------------------------------------------------------------
class ScalingLaw:
    """mode = 'agnostic' : one A and one n for all species (usable for unseen species)
       mode = 'shared_n' : A per species, one n
       mode = 'species_n': A and n per species"""

    def __init__(self, mode):
        self.mode = mode
        self.species_aware = mode != "agnostic"

    def design(self, df):
        cols, names = [], []
        if self.mode == "agnostic":
            cols.append(np.ones(len(df))); names.append("ln A")
        else:
            for s in SPECIES:
                cols.append((df["species"] == s).values.astype(float)); names.append(f"ln A {s}")
        if self.mode == "species_n":
            for s in SPECIES:
                cols.append(((df["species"] == s).values * df["logd"].values)); names.append(f"n {s}")
        else:
            cols.append(df["logd"].values); names.append("n")
        for c, nm in [("moist", "ln f_M"), ("lowheat", "ln f_H"), ("lowwind", "ln f_W")]:
            cols.append(df[c].values); names.append(nm)
        return np.stack(cols, 1), names

    def fit(self, df):
        X, self.names_ = self.design(df)
        y = np.log(df["t"].values)
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ self.beta_
        dof = max(len(y) - np.linalg.matrix_rank(X), 1)
        self.sigma_ = np.sqrt(res @ res / dof)
        self.cov_ = self.sigma_ ** 2 * np.linalg.pinv(X.T @ X)
        self.dof_ = dof
        return self

    def predict(self, df):
        return np.exp(self.design(df)[0] @ self.beta_)

    def table(self):
        se = np.sqrt(np.diag(self.cov_))
        tq = stats.t.ppf(0.975, self.dof_)
        rows = []
        for nm, b, s in zip(self.names_, self.beta_, se):
            lo, hi = b - tq * s, b + tq * s
            if nm.startswith("ln "):
                rows.append({"parameter": nm[3:], "estimate": np.exp(b),
                             "95% CI low": np.exp(lo), "95% CI high": np.exp(hi)})
            else:
                rows.append({"parameter": nm, "estimate": b, "95% CI low": lo, "95% CI high": hi})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Data-driven models
# ----------------------------------------------------------------------------
def features(df, species_aware):
    X = df[["d", "moist", "lowheat", "lowwind"]].values.astype(float)
    if species_aware:
        X = np.hstack([X, np.stack([(df["species"] == s).values.astype(float) for s in SPECIES], 1)])
    return X


class BlackBox:
    def __init__(self, kind, species_aware):
        self.kind, self.species_aware = kind, species_aware

    def fit(self, df):
        X = features(df, self.species_aware)
        self.sc_ = StandardScaler().fit(X)
        if self.kind == "RF":
            self.m_ = RandomForestRegressor(n_estimators=300, random_state=SEED)
        elif self.kind == "XGB":
            self.m_ = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=SEED, verbosity=0)
        else:
            self.m_ = make_gp(X.shape[1])
        self.m_.fit(self.sc_.transform(X), np.log(df["t"].values))
        return self

    def predict(self, df):
        return np.exp(self.m_.predict(self.sc_.transform(features(df, self.species_aware))))


class GrayBox:
    """Scaling law + Gaussian-process residual in log space."""

    def __init__(self, species_aware):
        self.species_aware = species_aware
        self.phys = ScalingLaw("shared_n" if species_aware else "agnostic")

    def fit(self, df):
        self.phys.fit(df)
        r = np.log(df["t"].values) - np.log(self.phys.predict(df))
        X = features(df, self.species_aware)
        self.sc_ = StandardScaler().fit(X)
        self.gp_ = make_gp(X.shape[1]).fit(self.sc_.transform(X), r)
        return self

    def predict(self, df):
        X = self.sc_.transform(features(df, self.species_aware))
        return self.phys.predict(df) * np.exp(self.gp_.predict(X))


def model_zoo(species_aware, allow_species_n=True):
    if not species_aware:
        return {"Scaling law (species-agnostic, 5 par)": lambda: ScalingLaw("agnostic"),
                "Gray-box (law + GP residual)": lambda: GrayBox(False),
                "Black-box GP": lambda: BlackBox("GP", False),
                "Black-box RF": lambda: BlackBox("RF", False),
                "Black-box XGB": lambda: BlackBox("XGB", False)}
    zoo = {"Scaling law (shared n, 8 par)": lambda: ScalingLaw("shared_n")}
    if allow_species_n:
        zoo["Scaling law (species n, 11 par)"] = lambda: ScalingLaw("species_n")
    zoo.update({"Gray-box (law + GP residual)": lambda: GrayBox(True),
                "Black-box GP": lambda: BlackBox("GP", True),
                "Black-box RF": lambda: BlackBox("RF", True),
                "Black-box XGB": lambda: BlackBox("XGB", True)})
    return zoo


# ----------------------------------------------------------------------------
# Validation protocols
# ----------------------------------------------------------------------------
def splits(df):
    idx = np.arange(len(df))
    out = {}
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    out["A. Interpolation (5-fold x10, by condition)"] = (list(rkf.split(idx)), True, True)
    out["B. Unseen species (leave-one-species-out)"] = (
        [(idx[df.species != s], idx[df.species == s]) for s in SPECIES], False, True)
    # A species tested only as 2 mm dowels: predict its 6 mm behaviour
    out["C. Species seen only at 2 mm (its 6 mm held out)"] = (
        [(idx[~((df.species == s) & (df.d == 6))], idx[(df.species == s) & (df.d == 6)])
         for s in SPECIES], True, False)
    corner = (df.heat == "high") & (df.wind == "high")
    out["D. Unseen heat-wind combination (high heat + high wind held out)"] = (
        [(idx[~corner], idx[corner])], True, True)
    return out


def run_validation(df):
    rows = []
    for proto, (folds, species_aware, allow_sn) in splits(df).items():
        for name, make in model_zoo(species_aware, allow_sn).items():
            ys, ps = [], []
            for tr, te in folds:
                m = make().fit(df.iloc[tr])
                ys.append(df.t.values[te]); ps.append(m.predict(df.iloc[te]))
            y, p = np.concatenate(ys), np.concatenate(ps)
            rows.append({"protocol": proto, "model": name, "n_test": len(y), **metrics(y, p)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
COL = {"Douglas-fir": "#b5482a", "Juniper": "#2a6fb5", "Ponderosa": "#2f8f4e", "White oak": "#7a4fa3"}


def figures(df, law, law_sn, res):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    tab = law_sn.table().set_index("parameter")
    rng = np.random.default_rng(0)
    for s in SPECIES:
        d = df[df.species == s]
        ax.scatter(d.d * (1 + 0.04 * rng.standard_normal(len(d))), d.t, s=28, color=COL[s], alpha=0.7)
        gm = [np.exp(np.log(d[d.d == k].t).mean()) for k in (2, 6)]
        ax.plot([2, 6], gm, color=COL[s], lw=2, label=f"{s}: n = {tab.loc['n ' + s, 'estimate']:.2f}")
    dd = np.array([2, 6.0])
    for n, ls in [(1.5, ":"), (2.0, "--")]:
        ax.plot(dd, 7 * (dd / 2) ** n, color="k", lw=1, ls=ls, label=f"slope n = {n}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.minorticks_off(); ax.set_xticks([2, 6]); ax.set_xticklabels(["2", "6"])
    ax.set_yticks([5, 10, 20, 40, 60]); ax.set_yticklabels(["5", "10", "20", "40", "60"])
    ax.set_xlabel("Dowel diameter (mm)"); ax.set_ylabel("Firebrand generation time (s)")
    ax.set_title("(a) Diameter scaling by species"); ax.legend(fontsize=8)

    ax = axes[1]
    p = law_sn.predict(df)
    for s in SPECIES:
        m = df.species == s
        ax.errorbar(df.t[m], p[m], xerr=df.sd[m], fmt="o", ms=5, color=COL[s], alpha=0.8,
                    elinewidth=0.6, label=s)
    lim = [3, 70]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.minorticks_off()
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([5, 10, 20, 40, 60]); axis.set_ticklabels(["5", "10", "20", "40", "60"])
    ax.set_xlabel("Observed FGT (s), bar = experimental spread"); ax.set_ylabel("Scaling-law prediction (s)")
    ax.set_title("(b) Scaling law, species-specific exponent"); ax.legend(fontsize=8)

    ax = axes[2]
    t = law_sn.table().set_index("parameter").loc[["f_M", "f_H", "f_W"]]
    lab = ["15 % moisture\n(vs dry)", "Low heat\n(vs high)", "Low wind\n(vs high)"]
    ax.errorbar(range(3), t["estimate"], yerr=[t["estimate"] - t["95% CI low"], t["95% CI high"] - t["estimate"]],
                fmt="o", color="#1b7f5c", capsize=5, ms=8)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Multiplier on FGT (95 % CI)"); ax.set_title("(c) Effect of moisture, heat and wind")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fgt_scaling_law.png"), dpi=200); plt.close(fig)

    protos = list(res.protocol.unique())
    fig, axes = plt.subplots(1, len(protos), figsize=(4.6 * len(protos), 4.4))
    for ax, proto in zip(axes, protos):
        d = res[res.protocol == proto]
        col = ["#1b7f5c" if "Gray" in m else "#c9792b" if "Scaling" in m else "#7a8794" for m in d.model]
        ax.barh(d.model, d["RMSE [s]"], color=col); ax.invert_yaxis(); ax.set_xlabel("RMSE (s)")
        ax.set_title(proto.replace(" (", "\n("), fontsize=9); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fgt_validation_rmse.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20,
                  "display.float_format", "{:.3f}".format)
    df = load_conditions()
    print(f"{len(df)} conditions, {df.species.nunique()} species; "
          f"mean experimental spread = {df.sd.mean():.2f} s "
          f"({100 * (df.sd / df.t).mean():.0f} % of the mean)\n")

    laws = {}
    for mode in ("agnostic", "shared_n", "species_n"):
        L = laws[mode] = ScalingLaw(mode).fit(df)
        p = L.predict(df)
        m = metrics(df.t.values, p)
        print(f"[{mode}] in-sample RMSE = {m['RMSE [s]']:.2f} s, R2 = {m['R2']:.3f}, "
              f"residual scatter = {100 * L.sigma_:.0f} % (log space)")
        print(L.table().to_string(index=False), "\n")
        L.table().to_csv(os.path.join(OUT, f"scaling_law_parameters_{mode}.csv"), index=False)

    # Is a species-specific exponent justified? (F-test, nested log-linear models)
    a, b = laws["shared_n"], laws["species_n"]
    rss = lambda L: L.sigma_ ** 2 * L.dof_
    F = ((rss(a) - rss(b)) / (a.dof_ - b.dof_)) / (rss(b) / b.dof_)
    print(f"F-test, species-specific exponent vs shared exponent: F = {F:.1f}, "
          f"p = {stats.f.sf(F, a.dof_ - b.dof_, b.dof_):.2g}\n")

    res = run_validation(df)
    res.to_csv(os.path.join(OUT, "fgt_validation_results.csv"), index=False)
    for proto in res.protocol.unique():
        print(f"=== {proto} ===")
        print(res[res.protocol == proto].drop(columns="protocol").to_string(index=False), "\n")
    figures(df, laws["shared_n"], laws["species_n"], res)
