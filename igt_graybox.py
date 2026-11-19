"""
Physics-encoded gray-box model for ignition time (IGT) of live shrub leaves.

Physics core (thermally thin fuel energy balance):
    t_ig = A * (1 + B * M) / (q - q_cr)
        M    = LFMC / 100                      (kg water per kg dry fuel)
        q    = incident heat flux              (kW/m2)
        q_cr = critical heat flux for ignition (kW/m2)
        A    = rho*delta*c_dry*(T_ig - T0)/absorptivity   (kJ/m2)
        B    = extra energy per unit moisture / dry sensible energy (-)

Gray-box: log(t_obs) = log(t_phys) + GP residual.

Data handling: every condition in IGT_Edited.csv is stored as three rows
(mean, mean + spread, mean - spread). The script collapses them to one row per
condition (56 conditions) so that no condition appears in both train and test.
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "igt_graybox_outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_conditions():
    raw = pd.read_csv(os.path.join(HERE, "IGT_Edited.csv"))
    raw.columns = ["species", "lfmc", "thick", "q", "t"]
    keys = ["species", "lfmc", "thick", "q"]
    g = raw.groupby(keys, sort=False)["t"]
    df = g.mean().reset_index()
    df["sd"] = ((g.max() - g.min()) / 2).values
    df["M"] = df["lfmc"] / 100.0
    assert (g.size() == 3).all(), "expected three rows per condition"
    return df


# ----------------------------------------------------------------------------
# Physics models
# ----------------------------------------------------------------------------
class Physics:
    """t = A * (1 + B*M) / (q - q_cr), fitted in log space.

    mode = 'global'  : A, B, q_cr shared by all species
    mode = 'thick'   : A = a * leaf thickness; a, B, q_cr shared
    mode = 'species' : A per species; B, q_cr shared
    mode = 'full'    : A, B, q_cr per species
    """

    def __init__(self, mode):
        self.mode = mode
        self.species_aware = mode in ("species", "full")

    def _unpack(self, p, df):
        sp = df["species"].map(self.idx_).values if self.species_aware else None
        if self.mode == "global":
            return p[0], p[1], p[2]
        if self.mode == "thick":
            return p[0] * df["thick"].values, p[1], p[2]
        n = len(self.idx_)
        if self.mode == "species":
            return p[:n][sp], p[n], p[n + 1]
        return p[:n][sp], p[n:2 * n][sp], p[2 * n:3 * n][sp]

    def _t(self, p, df):
        A, B, qcr = self._unpack(p, df)
        return A * (1 + B * df["M"].values) / (df["q"].values - qcr)

    def fit(self, df):
        self.idx_ = {s: i for i, s in enumerate(sorted(df["species"].unique()))}
        n = len(self.idx_)
        qmax = df["q"].min() - 0.5          # q_cr must stay below every tested flux
        if self.mode in ("global", "thick"):
            a0 = 30.0 if self.mode == "global" else 80.0
            x0, lo, hi = [a0, 2.0, 5.0], [1e-3, 0, 0], [1e4, 50, qmax]
        elif self.mode == "species":
            x0 = [30.0] * n + [2.0, 5.0]
            lo = [1e-3] * n + [0, 0]
            hi = [1e4] * n + [50, qmax]
        else:
            x0 = [30.0] * n + [2.0] * n + [5.0] * n
            lo = [1e-3] * n + [0] * n + [0] * n
            hi = [1e4] * n + [50] * n + [qmax] * n
        y = np.log(df["t"].values)
        res = least_squares(lambda p: np.log(self._t(p, df)) - y, x0, bounds=(lo, hi))
        self.p_ = res.x
        return self

    def predict(self, df):
        return self._t(self.p_, df)

    def params(self):
        if self.mode in ("global", "thick"):
            name = "A [kJ/m2]" if self.mode == "global" else "a [kJ/m2 per mm]"
            return pd.DataFrame([{"species": "all", name: self.p_[0],
                                  "B [-]": self.p_[1], "q_cr [kW/m2]": self.p_[2]}])
        n = len(self.idx_)
        rows = []
        for s, i in self.idx_.items():
            if self.mode == "species":
                rows.append({"species": s, "A [kJ/m2]": self.p_[i],
                             "B [-]": self.p_[n], "q_cr [kW/m2]": self.p_[n + 1]})
            else:
                rows.append({"species": s, "A [kJ/m2]": self.p_[i],
                             "B [-]": self.p_[n + i], "q_cr [kW/m2]": self.p_[2 * n + i]})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Feature matrices for the data-driven parts
# ----------------------------------------------------------------------------
ALL_SPECIES = ["Manzanita", "Oregon grape", "Salal"]


def features(df, species_aware):
    X = df[["M", "q"]].values.astype(float)
    if species_aware:
        onehot = np.stack([(df["species"] == s).values.astype(float) for s in ALL_SPECIES], 1)
        return np.hstack([X, onehot])
    return np.hstack([X, df[["thick"]].values.astype(float)])


def make_gp(n_feat):
    kernel = (ConstantKernel(1.0, (1e-3, 1e2)) * RBF(np.ones(n_feat), (1e-1, 1e2))
              + WhiteKernel(0.01, (1e-4, 1.0)))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=SEED)


class BlackBox:
    """Data-driven baseline predicting log(t) from raw features."""

    def __init__(self, kind, species_aware):
        self.kind, self.species_aware = kind, species_aware

    def fit(self, df):
        X = features(df, self.species_aware)
        self.sc_ = StandardScaler().fit(X)
        if self.kind == "RF":
            self.m_ = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=SEED)
        elif self.kind == "XGB":
            self.m_ = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=SEED, verbosity=0)
        else:
            self.m_ = make_gp(X.shape[1])
        self.m_.fit(self.sc_.transform(X), np.log(df["t"].values))
        return self

    def predict(self, df, return_std=False):
        X = self.sc_.transform(features(df, self.species_aware))
        if return_std:
            mu, sd = self.m_.predict(X, return_std=True)
            return np.exp(mu), sd
        return np.exp(self.m_.predict(X))


class GrayBox:
    """Physics core + Gaussian-process residual in log space."""

    def __init__(self, mode):
        self.phys = Physics(mode)
        self.species_aware = self.phys.species_aware

    def fit(self, df):
        self.phys.fit(df)
        r = np.log(df["t"].values) - np.log(self.phys.predict(df))
        X = features(df, self.species_aware)
        self.sc_ = StandardScaler().fit(X)
        self.gp_ = make_gp(X.shape[1]).fit(self.sc_.transform(X), r)
        return self

    def predict(self, df, return_std=False):
        X = self.sc_.transform(features(df, self.species_aware))
        mu, sd = self.gp_.predict(X, return_std=True)
        t = self.phys.predict(df) * np.exp(mu)
        return (t, sd) if return_std else t


def model_zoo(species_aware):
    if species_aware:
        return {
            "Physics (global, 3 par)": lambda: Physics("global"),
            "Physics (per-species A, 5 par)": lambda: Physics("species"),
            "Physics (per-species all, 9 par)": lambda: Physics("full"),
            "Gray-box (physics + GP residual)": lambda: GrayBox("species"),
            "Black-box GP": lambda: BlackBox("GP", True),
            "Black-box RF": lambda: BlackBox("RF", True),
            "Black-box XGB": lambda: BlackBox("XGB", True),
        }
    return {
        "Physics (global, 3 par)": lambda: Physics("global"),
        "Physics (thickness-scaled, 3 par)": lambda: Physics("thick"),
        "Gray-box (physics + GP residual)": lambda: GrayBox("thick"),
        "Black-box GP": lambda: BlackBox("GP", False),
        "Black-box RF": lambda: BlackBox("RF", False),
        "Black-box XGB": lambda: BlackBox("XGB", False),
    }


# ----------------------------------------------------------------------------
# Validation protocols
# ----------------------------------------------------------------------------
def splits(df):
    idx = np.arange(len(df))
    out = {}
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    out["A. Interpolation (5-fold x10, by condition)"] = (list(rkf.split(idx)), True)
    out["B. Unseen species (leave-one-species-out)"] = (
        [(idx[df.species != s], idx[df.species == s]) for s in ALL_SPECIES], False)
    out["C. Unseen high heat flux (train 11-16, test 21)"] = (
        [(idx[df.q < 21], idx[df.q == 21])], True)
    out["D. Unseen low heat flux (train 13-21, test 11)"] = (
        [(idx[df.q > 11], idx[df.q == 11])], True)
    top = df.groupby("species")["lfmc"].transform("max")
    out["E. Unseen high moisture (top LFMC per species held out)"] = (
        [(idx[df.lfmc != top], idx[df.lfmc == top])], True)
    return out


def metrics(y, p):
    return {"RMSE [s]": np.sqrt(mean_squared_error(y, p)),
            "MAE [s]": mean_absolute_error(y, p),
            "MAPE [%]": 100 * np.mean(np.abs(p - y) / y),
            "R2": r2_score(y, p)}


def run_validation(df):
    rows, cover = [], []
    for proto, (folds, species_aware) in splits(df).items():
        for name, make in model_zoo(species_aware).items():
            ys, ps, zs = [], [], []
            for tr, te in folds:
                m = make().fit(df.iloc[tr])
                if isinstance(m, (GrayBox,)) or (isinstance(m, BlackBox) and m.kind == "GP"):
                    p, s = m.predict(df.iloc[te], return_std=True)
                    zs.append((np.log(df.t.values[te]) - np.log(p)) / s)
                else:
                    p = m.predict(df.iloc[te])
                ys.append(df.t.values[te]); ps.append(p)
            y, p = np.concatenate(ys), np.concatenate(ps)
            rows.append({"protocol": proto, "model": name, "n_test": len(y), **metrics(y, p)})
            if zs:
                z = np.concatenate(zs)
                cover.append({"protocol": proto, "model": name,
                              "90% interval coverage [%]": 100 * np.mean(np.abs(z) < 1.645)})
    return pd.DataFrame(rows), pd.DataFrame(cover)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figures(df):
    phys = Physics("full").fit(df)
    gray = GrayBox("species").fit(df)
    colors = {"Manzanita": "#b5482a", "Oregon grape": "#2a6fb5", "Salal": "#2f8f4e"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 1/t vs q: thin-fuel theory predicts straight lines crossing zero at q_cr
    ax = axes[0]
    qq = np.linspace(0, 22, 100)
    par = phys.params().set_index("species")
    for s in ALL_SPECIES:
        d = df[df.species == s]
        sc = ax.scatter(d.q, 1 / d.t, c=d.lfmc, cmap="viridis", s=45,
                        edgecolor=colors[s], linewidth=1.5, vmin=10, vmax=130)
        for M in (0.2, 1.0):
            A, B, qc = par.loc[s, ["A [kJ/m2]", "B [-]", "q_cr [kW/m2]"]]
            ax.plot(qq, np.clip(qq - qc, 0, None) / (A * (1 + B * M)), color=colors[s],
                    lw=1.2, ls="-" if M == 0.2 else "--",
                    label=f"{s}, LFMC {int(M*100)}%")
    fig.colorbar(sc, ax=ax, label="LFMC (%)")
    ax.set_xlabel("Heat flux q (kW/m$^2$)"); ax.set_ylabel("1 / ignition time (1/s)")
    ax.set_title("(a) Thin-fuel physics: 1/t linear in q"); ax.legend(fontsize=7)
    ax.set_xlim(0, 22); ax.set_ylim(0, None)

    # (b) parity of physics core and gray-box (in-sample fit)
    ax = axes[1]
    for s in ALL_SPECIES:
        d = df[df.species == s]
        ax.errorbar(d.t, phys.predict(d), xerr=d.sd, fmt="o", ms=5, color=colors[s],
                    alpha=0.8, label=s, elinewidth=0.6)
    lim = [0, df.t.max() * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Observed ignition time (s), bar = experimental spread")
    ax.set_ylabel("Physics-core prediction (s)")
    ax.set_title(f"(b) Physics core, 9 parameters (R$^2$ = {r2_score(df.t, phys.predict(df)):.3f})")
    ax.legend(fontsize=8)

    # (c) fitted critical heat flux and effective energy per species
    ax = axes[2]
    x = np.arange(len(ALL_SPECIES))
    ax.bar(x - 0.2, [par.loc[s, "q_cr [kW/m2]"] for s in ALL_SPECIES], 0.4,
           color=[colors[s] for s in ALL_SPECIES])
    ax.set_ylabel("Critical heat flux q$_{cr}$ (kW/m$^2$)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [par.loc[s, "A [kJ/m2]"] for s in ALL_SPECIES], 0.4,
            color=[colors[s] for s in ALL_SPECIES], alpha=0.45, hatch="//")
    ax2.set_ylabel("Dry-fuel ignition energy A (kJ/m$^2$), hatched")
    ax.set_xticks(x); ax.set_xticklabels(ALL_SPECIES)
    ax.set_title("(c) Identified physical parameters")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_physics_core.png"), dpi=200)
    plt.close(fig)
    return phys, gray


def figure_validation(res):
    protos = list(res.protocol.unique())
    fig, axes = plt.subplots(1, len(protos), figsize=(4.2 * len(protos), 4.6), sharey=False)
    for ax, proto in zip(axes, protos):
        d = res[res.protocol == proto]
        col = ["#1b7f5c" if "Gray" in m else "#c9792b" if "Physics" in m else "#7a8794"
               for m in d.model]
        ax.barh(d.model, d["RMSE [s]"], color=col)
        ax.invert_yaxis(); ax.set_xlabel("RMSE (s)")
        ax.set_title(proto.replace(" (", "\n("), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_validation_rmse.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20,
                  "display.float_format", "{:.3f}".format)
    df = load_conditions()
    print(f"{len(df)} conditions, {df.species.nunique()} species\n")

    phys, gray = figures(df)
    print("Identified physical parameters (all data):")
    for mode in ("global", "species", "full"):
        P = Physics(mode).fit(df)
        tab = P.params()
        print(f"\n[{mode}]  in-sample R2 = {r2_score(df.t, P.predict(df)):.3f}, "
              f"RMSE = {np.sqrt(mean_squared_error(df.t, P.predict(df))):.3f} s")
        print(tab.to_string(index=False))
        tab.to_csv(os.path.join(OUT, f"physics_parameters_{mode}.csv"), index=False)

    res, cov = run_validation(df)
    res.to_csv(os.path.join(OUT, "validation_results.csv"), index=False)
    cov.to_csv(os.path.join(OUT, "interval_coverage.csv"), index=False)
    figure_validation(res)
    for proto in res.protocol.unique():
        print(f"\n=== {proto} ===")
        print(res[res.protocol == proto].drop(columns="protocol").to_string(index=False))
    print("\n=== 90% prediction-interval coverage ===")
    print(cov.to_string(index=False))
    print(f"\nExperimental spread (mean SD across conditions): {df.sd.mean():.3f} s")
