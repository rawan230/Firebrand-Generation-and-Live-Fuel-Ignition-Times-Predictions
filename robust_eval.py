"""
Robust evaluation for the FGT / IGT study (adds to igt_graybox.py and fgt_physics.py):

  1. stronger, fairly tuned baselines (log-linear regression with the same number of
     parameters, quadratic ridge, GP with linear mean, nested-CV-tuned RF / XGBoost /
     monotone XGBoost / MLP) next to the untuned ones used before;
  2. bootstrap confidence intervals (resampling held-out CONDITIONS) for RMSE and R2,
     and paired bootstrap intervals for the RMSE difference to the physics model;
  3. a quantitative leakage experiment: random row split (as in earlier ML studies)
     versus a split that keeps the three rows of a condition together.

Run with the Anaconda interpreter:  python robust_eval.py [igt|fgt|leak|all]
Outputs go to robust_outputs/.
"""
import os
import sys
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

import igt_graybox as IG
import fgt_physics as FG

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "robust_outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42
NJOBS = 20
NBOOT = 5000


# ----------------------------------------------------------------------------
# Dataset specifications
# ----------------------------------------------------------------------------
class IGTSpec:
    name = "IGT"
    species = IG.ALL_SPECIES
    ref = "Heat balance (simple, 5)"

    def conditions(self):
        return IG.load_conditions()

    def raw_rows(self):
        raw = pd.read_csv(os.path.join(HERE, "IGT_Edited.csv"))
        raw.columns = ["species", "lfmc", "thick", "q", "t"]
        raw["M"] = raw["lfmc"] / 100.0
        raw["cond"] = raw.groupby(["species", "lfmc", "thick", "q"], sort=False).ngroup()
        return raw

    def raw_features(self, df, sa):                  # for trees (scale-free)
        return IG.features(df, sa)

    def smooth_features(self, df, sa):               # log-scaled inputs for smooth models
        cols = [np.log(df["q"].values), df["M"].values]
        if sa:
            cols += [(df["species"] == s).values.astype(float) for s in self.species]
        else:
            cols += [np.log(df["thick"].values)]
        return np.stack(cols, 1)

    def monotone(self, sa):                          # raw feature order: M, q, (species | thick)
        return (1, -1) + (0,) * (len(self.species) if sa else 1)

    def physics_zoo(self, sa):
        if sa:
            return {"Heat balance (simple, 5)": lambda: IG.Physics("species"),
                    "Heat balance (exact, 5)": lambda: ExactPhysics("species")}
        return {"Heat balance (simple, 3)": lambda: IG.Physics("thick"),
                "Heat balance (exact, 3)": lambda: ExactPhysics("thick"),
                "Heat balance (simple, A=a*thick^k, 4)": lambda: ThickPow(False)}

    def protocols(self, df):
        idx = np.arange(len(df))
        out = {}
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
        out["A. Normal (5-fold x10)"] = (list(rkf.split(idx)), True)
        out["B. New species"] = ([(idx[df.species != s], idx[df.species == s])
                                  for s in self.species], False)
        out["C. New high heat flux"] = ([(idx[df.q < 21], idx[df.q == 21])], True)
        out["D. New low heat flux"] = ([(idx[df.q > 11], idx[df.q == 11])], True)
        top = df.groupby("species")["lfmc"].transform("max")
        out["E. New high moisture"] = ([(idx[df.lfmc != top], idx[df.lfmc == top])], True)
        return out


class FGTSpec:
    name = "FGT"
    species = FG.SPECIES
    ref = "Law, species n (11)"

    def conditions(self):
        return FG.load_conditions()

    def raw_rows(self):
        raw = pd.read_csv(os.path.join(HERE, "FGT_Edited.csv"))
        raw.columns = ["d", "species", "mc", "heat", "wind", "mean_time", "t"]
        raw["logd"] = np.log(raw["d"] / 2.0)
        raw["moist"] = (raw["mc"] == "15%").astype(float)
        raw["lowheat"] = (raw["heat"] == "low").astype(float)
        raw["lowwind"] = (raw["wind"] == "low").astype(float)
        raw["cond"] = raw.groupby(["d", "species", "mc", "heat", "wind"], sort=False).ngroup()
        return raw

    def raw_features(self, df, sa):
        return FG.features(df, sa)

    def smooth_features(self, df, sa):
        cols = [df["logd"].values, df["moist"].values, df["lowheat"].values, df["lowwind"].values]
        if sa:
            cols += [(df["species"] == s).values.astype(float) for s in self.species]
        return np.stack(cols, 1)

    def monotone(self, sa):                          # d, moist, lowheat, lowwind, (species)
        return (1, 1, 1, 1) + ((0,) * len(self.species) if sa else ())

    def physics_zoo(self, sa, allow_sn=True):
        if not sa:
            return {"Law, one n (5)": lambda: FG.ScalingLaw("agnostic")}
        z = {"Law, shared n (8)": lambda: FG.ScalingLaw("shared_n")}
        if allow_sn:
            z["Law, species n (11)"] = lambda: FG.ScalingLaw("species_n")
        return z

    def protocols(self, df):
        idx = np.arange(len(df))
        out = {}
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
        out["A. Normal (5-fold x10)"] = (list(rkf.split(idx)), True, True)
        out["B. New species"] = ([(idx[df.species != s], idx[df.species == s])
                                  for s in self.species], False, True)
        corner = (df.heat == "high") & (df.wind == "high")
        out["C. New temperature-wind pair"] = ([(idx[~corner], idx[corner])], True, True)
        out["D. Species seen at 2 mm only"] = (
            [(idx[~((df.species == s) & (df.d == 6))], idx[(df.species == s) & (df.d == 6)])
             for s in self.species], True, False)
        return {k: (v[0], v[1], v[2]) for k, v in out.items()}


class ThickPow:
    """Heat balance with A = a * thickness^k (k = 1 is the thin-body energy argument).
    Used only when the species is unseen (species enters through its thickness)."""

    def __init__(self, exact=False):
        self.exact = exact

    def _t(self, p, df):
        a, k, B, qcr = p
        q = df["q"].values
        E = a * df["thick"].values ** k * (1 + B * df["M"].values)
        if self.exact:
            return -(E / qcr) * np.log(1 - qcr / q)
        return E / (q - qcr)

    def fit(self, df):
        qmax = df["q"].min() - 0.5
        y = np.log(df["t"].values)
        best = None
        for k0 in (0.5, 1.0, 2.0):
            r = least_squares(lambda p: np.log(self._t(p, df)) - y, [80.0, k0, 2.0, 5.0],
                              bounds=([1e-3, 0, 0, 0], [1e4, 5, 50, qmax]))
            if best is None or r.cost < best.cost:
                best = r
        self.p_ = best.x
        return self

    def predict(self, df):
        return self._t(self.p_, df)


class ExactPhysics(IG.Physics):
    """Exact solution of the lumped heat balance, t = -(E/qcr) ln(1 - qcr/q)."""

    def _t(self, p, df):
        A, B, qcr = self._unpack(p, df)
        q = df["q"].values
        return -(A * (1 + B * df["M"].values) / qcr) * np.log(1 - qcr / q)


# ----------------------------------------------------------------------------
# Baseline models (all predict log time, return seconds)
# ----------------------------------------------------------------------------
class LogLinear:
    """ln t = beta^T [1, smooth features], ordinary least squares."""

    def __init__(self, spec, sa):
        self.spec, self.sa = spec, sa

    def _X(self, df):
        F = self.spec.smooth_features(df, self.sa)
        return np.hstack([np.ones((len(df), 1)), F])

    def fit(self, df):
        self.b_, *_ = np.linalg.lstsq(self._X(df), np.log(df["t"].values), rcond=None)
        return self

    def predict(self, df):
        return np.exp(self._X(df) @ self.b_)


class PowerLaw(LogLinear):
    """ln t = b0 + b1 ln q + b2 ln(1+M) + species term (IGT only): moisture enters
    as a power of (1+M) instead of exponentially."""

    def _X(self, df):
        cols = [np.ones(len(df)), np.log(df["q"].values), np.log(1 + df["M"].values)]
        if self.sa:
            cols += [(df["species"] == s).values.astype(float) for s in self.spec.species]
        else:
            cols += [np.log(df["thick"].values)]
        return np.stack(cols, 1)


class GPLinearMean:
    """Log-linear regression (linear mean) + Gaussian process on its residuals."""

    def __init__(self, spec, sa):
        self.spec, self.sa = spec, sa
        self.lin = LogLinear(spec, sa)

    def fit(self, df):
        self.lin.fit(df)
        r = np.log(df["t"].values) - np.log(self.lin.predict(df))
        F = self.spec.smooth_features(df, self.sa)
        self.sc_ = StandardScaler().fit(F)
        self.gp_ = IG.make_gp(F.shape[1]).fit(self.sc_.transform(F), r)
        return self

    def predict(self, df):
        F = self.sc_.transform(self.spec.smooth_features(df, self.sa))
        return self.lin.predict(df) * np.exp(self.gp_.predict(F))


class QuadRidge:
    """Degree-2 polynomial ridge regression on the log-scaled inputs."""

    def __init__(self, spec, sa):
        self.spec, self.sa = spec, sa

    def fit(self, df):
        F = self.spec.smooth_features(df, self.sa)
        self.m_ = make_pipeline(StandardScaler(), PolynomialFeatures(2),
                                RidgeCV(alphas=np.logspace(-4, 2, 13)))
        self.m_.fit(F, np.log(df["t"].values))
        return self

    def predict(self, df):
        return np.exp(self.m_.predict(self.spec.smooth_features(df, self.sa)))


class Tuned:
    """RF / XGBoost / monotone XGBoost / MLP with hyper-parameters chosen by an inner
    3-fold cross-validation on the training conditions only (nested tuning)."""

    def __init__(self, kind, spec, sa):
        self.kind, self.spec, self.sa = kind, spec, sa

    def _estimator(self):
        k = self.kind
        if k == "RF":
            return (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
                    {"max_depth": [2, 4, None], "min_samples_leaf": [1, 2, 4],
                     "max_features": [0.6, 1.0]})
        if k in ("XGB", "XGBmono"):
            base = dict(random_state=SEED, verbosity=0, n_jobs=1, subsample=0.8,
                        tree_method="hist")
            if k == "XGBmono":
                base["monotone_constraints"] = self.spec.monotone(self.sa)
            return (XGBRegressor(**base),
                    {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
                     "n_estimators": [100, 300], "reg_lambda": [1.0, 5.0]})
        return (MLPRegressor(solver="lbfgs", max_iter=3000, random_state=SEED),
                {"hidden_layer_sizes": [(8,), (16,), (32, 32)],
                 "alpha": [1e-3, 1e-2, 1e-1, 1.0], "activation": ["tanh", "relu"]})

    def _X(self, df):
        if self.kind == "MLP":
            return self.sc_.transform(self.spec.smooth_features(df, self.sa))
        return self.spec.raw_features(df, self.sa)

    def fit(self, df):
        y = np.log(df["t"].values)
        if self.kind == "MLP":
            self.sc_ = StandardScaler().fit(self.spec.smooth_features(df, self.sa))
            self.ym_, self.ys_ = y.mean(), y.std()
            y = (y - self.ym_) / self.ys_
        est, grid = self._estimator()
        gs = GridSearchCV(est, grid, cv=KFold(3, shuffle=True, random_state=0),
                          scoring="neg_mean_squared_error", refit=True, n_jobs=1)
        gs.fit(self._X(df), y)
        self.m_, self.best_ = gs.best_estimator_, gs.best_params_
        return self

    def predict(self, df):
        p = self.m_.predict(self._X(df))
        if self.kind == "MLP":
            p = p * self.ys_ + self.ym_
        return np.exp(p)


class Untuned:
    """The default GP / RF / XGBoost of the earlier analysis (fixed settings)."""

    def __init__(self, kind, spec, sa):
        self.kind, self.spec, self.sa = kind, spec, sa

    def fit(self, df):
        X = self.spec.raw_features(df, self.sa)
        self.sc_ = StandardScaler().fit(X)
        if self.kind == "RF":
            m = RandomForestRegressor(n_estimators=300, random_state=SEED)
        elif self.kind == "XGB":
            m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, random_state=SEED, verbosity=0, n_jobs=1)
        else:
            m = IG.make_gp(X.shape[1])
        self.m_ = m.fit(self.sc_.transform(X), np.log(df["t"].values))
        return self

    def predict(self, df):
        X = self.sc_.transform(self.spec.raw_features(df, self.sa))
        return np.exp(self.m_.predict(X))


def baseline_zoo(spec, sa):
    npar = np.linalg.matrix_rank(LogLinear(spec, sa)._X(spec.conditions()))
    extra = {}
    if spec.name == "IGT":
        extra[f"Power law in q, 1+M ({npar})"] = lambda: PowerLaw(spec, sa)
    return {
        f"Log-linear ({npar})": lambda: LogLinear(spec, sa),
        **extra,
        "Quadratic ridge": lambda: QuadRidge(spec, sa),
        "GP, linear mean": lambda: GPLinearMean(spec, sa),
        "GP (default)": lambda: Untuned("GP", spec, sa),
        "RF (default)": lambda: Untuned("RF", spec, sa),
        "XGBoost (default)": lambda: Untuned("XGB", spec, sa),
        "RF (tuned)": lambda: Tuned("RF", spec, sa),
        "XGBoost (tuned)": lambda: Tuned("XGB", spec, sa),
        "XGBoost (tuned, monotone)": lambda: Tuned("XGBmono", spec, sa),
        "MLP (tuned)": lambda: Tuned("MLP", spec, sa),
    }


def get_spec(name):
    return IGTSpec() if name == "IGT" else FGTSpec()


def zoo_for(spec, sa, allow_sn=True):
    z = dict(spec.physics_zoo(sa, allow_sn) if spec.name == "FGT" else spec.physics_zoo(sa))
    z.update(baseline_zoo(spec, sa))
    return z


# ----------------------------------------------------------------------------
# Parallel evaluation of one (protocol, model, fold)
# ----------------------------------------------------------------------------
def _job(specname, proto, sa, allow_sn, model, fold_id, tr, te):
    spec = get_spec(specname)
    df = spec.conditions()
    m = zoo_for(spec, sa, allow_sn)[model]().fit(df.iloc[tr])
    return proto, model, fold_id, te, m.predict(df.iloc[te])


def evaluate(specname):
    spec = get_spec(specname)
    df = spec.conditions()
    y = df["t"].values
    jobs, meta = [], {}
    for proto, spl in spec.protocols(df).items():
        folds, sa = spl[0], spl[1]
        allow_sn = spl[2] if len(spl) > 2 else True
        meta[proto] = (sa, allow_sn)
        for model in zoo_for(spec, sa, allow_sn):
            for k, (tr, te) in enumerate(folds):
                jobs.append((specname, proto, sa, allow_sn, model, k, tr, te))
    print(f"[{specname}] {len(jobs)} fits on {NJOBS} workers", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=NJOBS, verbose=0)(delayed(_job)(*j) for j in jobs)
    print(f"[{specname}] done in {time.time() - t0:.0f} s", flush=True)
    # collect: per (protocol, model) the squared error per test condition (mean over repeats)
    acc = {}
    for proto, model, k, te, p in res:
        d = acc.setdefault((proto, model), {})
        for i, pi in zip(te, p):
            d.setdefault(int(i), []).append(float(pi))
    store = {}
    for (proto, model), d in acc.items():
        ids = np.array(sorted(d))
        se = np.array([np.mean((np.array(d[i]) - y[i]) ** 2) for i in ids])
        store[(proto, model)] = (ids, se)
    with open(os.path.join(OUT, f"{specname}_se.pkl"), "wb") as f:
        pickle.dump({"store": store, "y": y, "meta": meta}, f)
    return store, y, meta


def summarize(specname, store, y, meta, rng=np.random.default_rng(0)):
    spec = get_spec(specname)
    rows = []
    protos = list(dict.fromkeys(p for p, _ in store))
    for proto in protos:
        models = [m for p, m in store if p == proto]
        ids = store[(proto, models[0])][0]
        n = len(ids)
        yy = y[ids]
        sst = ((yy - yy.mean()) ** 2)
        boot = rng.integers(0, n, size=(NBOOT, n))
        ref_name = spec.ref if (proto, spec.ref) in store else models[0]
        se_ref = store[(proto, ref_name)][1]
        rmse_ref_b = np.sqrt(se_ref[boot].mean(1))
        physics = set(spec.physics_zoo(meta[proto][0]) if specname == "IGT"
                      else spec.physics_zoo(meta[proto][0], meta[proto][1]))
        for m in models:
            se = store[(proto, m)][1]
            rmse = np.sqrt(se.mean())
            r2 = 1 - se.sum() / sst.sum()
            rb = np.sqrt(se[boot].mean(1))
            # the denominator of R2 uses the bootstrap sample's own mean
            ybm = yy[boot].mean(1, keepdims=True)
            r2b = 1 - se[boot].sum(1) / ((yy[boot] - ybm) ** 2).sum(1)
            d = rb - rmse_ref_b
            rows.append({
                "dataset": specname, "protocol": proto, "model": m, "is_physics": m in physics,
                "n_test": n, "RMSE": rmse, "RMSE_lo": np.percentile(rb, 2.5),
                "RMSE_hi": np.percentile(rb, 97.5), "R2": r2,
                "R2_lo": np.percentile(r2b, 2.5), "R2_hi": np.percentile(r2b, 97.5),
                "dRMSE_vs_ref": rmse - np.sqrt(se_ref.mean()),
                "dRMSE_lo": np.percentile(d, 2.5), "dRMSE_hi": np.percentile(d, 97.5),
                "P_ref_better": float(np.mean(d > 0)) if m != ref_name else np.nan,
                "ref": ref_name})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Leakage experiment (raw rows, random-row split vs condition-grouped split)
# ----------------------------------------------------------------------------
def _leak_job(specname, scheme, rep, model, tr, te):
    spec = get_spec(specname)
    raw = spec.raw_rows()
    z = {"Physics": lambda: (IG.Physics("species") if specname == "IGT"
                             else FG.ScalingLaw("species_n")),
         "Log-linear": lambda: LogLinear(spec, True),
         "GP (default)": lambda: Untuned("GP", spec, True),
         "RF (default)": lambda: Untuned("RF", spec, True),
         "XGBoost (default)": lambda: Untuned("XGB", spec, True)}
    m = z[model]().fit(raw.iloc[tr])
    return scheme, rep, model, te, m.predict(raw.iloc[te])


def leakage(specname, nrep=10):
    spec = get_spec(specname)
    raw = spec.raw_rows()
    n = len(raw)
    conds = raw["cond"].values
    ucond = np.unique(conds)
    jobs = []
    for rep in range(nrep):
        rs = np.random.default_rng(1000 + rep)
        perm = rs.permutation(n)
        row_folds = np.array_split(perm, 5)
        cperm = rs.permutation(ucond)
        cfold = {c: i % 5 for i, c in enumerate(cperm)}
        grp = np.array([cfold[c] for c in conds])
        for scheme, folds in [("Random rows (leaky)", row_folds),
                              ("By condition", [np.where(grp == k)[0] for k in range(5)])]:
            for te in folds:
                tr = np.setdiff1d(np.arange(n), te)
                for model in ["Physics", "Log-linear", "GP (default)", "RF (default)",
                              "XGBoost (default)"]:
                    jobs.append((specname, scheme, rep, model, tr, te))
    print(f"[leak {specname}] {len(jobs)} fits", flush=True)
    res = Parallel(n_jobs=NJOBS)(delayed(_leak_job)(*j) for j in jobs)
    y = raw["t"].values
    rows = []
    for scheme in ["Random rows (leaky)", "By condition"]:
        for model in ["Physics", "Log-linear", "GP (default)", "RF (default)", "XGBoost (default)"]:
            per = []
            for rep in range(nrep):
                idx = np.concatenate([r[3] for r in res if r[0] == scheme and r[1] == rep and r[2] == model])
                pr = np.concatenate([r[4] for r in res if r[0] == scheme and r[1] == rep and r[2] == model])
                yy = y[idx]
                per.append([np.sqrt(np.mean((pr - yy) ** 2)), 100 * np.mean(np.abs(pr - yy) / yy),
                            1 - np.sum((pr - yy) ** 2) / np.sum((yy - yy.mean()) ** 2)])
            per = np.array(per)
            rows.append({"dataset": specname, "scheme": scheme, "model": model,
                         "RMSE": per[:, 0].mean(), "RMSE_sd": per[:, 0].std(),
                         "MAPE_pct": per[:, 1].mean(), "MAPE_sd": per[:, 1].std(),
                         "R2": per[:, 2].mean(), "R2_sd": per[:, 2].std()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500,
                  "display.float_format", "{:.3f}".format)
    outs = []
    for name in ("IGT", "FGT"):
        if what in (name.lower(), "all"):
            store, y, meta = evaluate(name)
            tab = summarize(name, store, y, meta)
            tab.to_csv(os.path.join(OUT, f"{name}_robust_results.csv"), index=False)
            print(tab.drop(columns=["dataset", "ref", "is_physics"]).to_string(index=False))
    if what in ("leak", "all"):
        L = pd.concat([leakage("IGT"), leakage("FGT")])
        L.to_csv(os.path.join(OUT, "leakage_results.csv"), index=False)
        print(L.to_string(index=False))
