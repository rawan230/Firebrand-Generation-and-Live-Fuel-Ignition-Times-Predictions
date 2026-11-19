"""
Inverse physics-informed neural network (PINN) for ignition time of live leaves.

Governing equation (lumped, thermally thin leaf, linearised heat losses):

    E(M, s) * d(theta)/dt = q - q_cr * theta,      theta(0) = 0,   ignition at theta = 1

    theta = (T - T0) / (T_ig - T0)        dimensionless leaf temperature
    E     = A_s * (1 + B * M)             energy to bring the moist leaf to ignition (kJ/m2)
    q_cr  = critical heat flux            (kW/m2): losses equal the supply at T_ig

The network learns the temperature history theta(t; q, M, species). Only ignition
times are measured, so the data loss is theta(t_ig_observed) = 1, and the ODE
residual is enforced at random collocation points over the whole (t, q, M) domain.
A_s, B and q_cr are trainable, so the PINN solves an inverse problem.

The same ODE has the closed form  t_ig = -(E / q_cr) * ln(1 - q_cr / q), which is
fitted directly by least squares and used to verify what the PINN identifies.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

from igt_graybox import ALL_SPECIES, OUT, Physics, load_conditions, metrics, splits

torch.set_num_threads(4)
T_SCALE = 30.0     # s, time normalisation
T_MAX = 1.5        # collocation and root search up to T_SCALE * T_MAX seconds
Q_SCALE = 21.0
QCR_MAX = 10.5     # q_cr must stay below the lowest tested flux (11 kW/m2)


# ----------------------------------------------------------------------------
# Closed-form solution of the same ODE, fitted by least squares (reference)
# ----------------------------------------------------------------------------
class ClosedFormODE:
    def __init__(self, species_aware):
        self.species_aware = species_aware

    def _t(self, p, df):
        if self.species_aware:
            A = p[:3][df["species"].map({s: i for i, s in enumerate(ALL_SPECIES)}).values]
        else:
            A = p[0] * df["thick"].values
        B, qcr = p[-2], p[-1]
        return -(A * (1 + B * df["M"].values) / qcr) * np.log(1 - qcr / df["q"].values)

    def fit(self, df):
        nA = 3 if self.species_aware else 1
        x0 = [20.0 if self.species_aware else 60.0] * nA + [1.0, 2.0]
        lo, hi = [1e-3] * nA + [0, 1e-3], [1e4] * nA + [50, QCR_MAX]
        y = np.log(df["t"].values)
        self.p_ = least_squares(lambda p: np.log(self._t(p, df)) - y, x0, bounds=(lo, hi)).x
        return self

    def predict(self, df):
        return self._t(self.p_, df)


# ----------------------------------------------------------------------------
# PINN
# ----------------------------------------------------------------------------
class IgnitionPINN(nn.Module):
    def __init__(self, species_aware, width=32, depth=4):
        super().__init__()
        self.species_aware = species_aware
        n_in = 3 + (3 if species_aware else 1)       # t, q, M + species one-hot or thickness
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))
        # physical parameters, deliberately initialised away from the fitted values
        nA = 3 if species_aware else 1
        self.logA = nn.Parameter(torch.full((nA,), np.log(20.0 if species_aware else 60.0)))
        self.rawB = nn.Parameter(torch.tensor(0.5413))      # softplus -> 1.0
        self.rawQ = nn.Parameter(torch.tensor(-1.4469))     # sigmoid * 10.5 -> 2.0

    @property
    def B(self):
        return nn.functional.softplus(self.rawB)

    @property
    def qcr(self):
        return QCR_MAX * torch.sigmoid(self.rawQ)

    def energy(self, M, z):
        """z = species one-hot (species-aware) or leaf thickness in mm (species-agnostic)."""
        A = (z * torch.exp(self.logA)).sum(1, keepdim=True)
        return A * (1 + self.B * M)

    def theta(self, tt, q, M, z):
        zin = z if self.species_aware else (z - 0.37) / 0.03
        x = torch.cat([tt, q / Q_SCALE, M, zin], 1)
        return tt * nn.functional.softplus(self.net(x))      # hard-enforces theta(0) = 0

    def residual(self, tt, q, M, z):
        tt = tt.requires_grad_(True)
        th = self.theta(tt, q, M, z)
        dth = torch.autograd.grad(th.sum(), tt, create_graph=True)[0]
        # ODE divided by q so that every term is O(1)
        return self.energy(M, z) / (T_SCALE * q) * dth - (1 - self.qcr * th / q)


def tensors(df, species_aware):
    f = lambda a: torch.tensor(np.asarray(a, dtype=np.float32)).reshape(len(df), -1)
    if species_aware:
        z = f(np.stack([(df["species"] == s).values for s in ALL_SPECIES], 1))
    else:
        z = f(df["thick"].values)
    return f(df["t"].values / T_SCALE), f(df["q"].values), f(df["M"].values), z


def collocation(n, df, species_aware, gen):
    tt = torch.rand(n, 1, generator=gen) * T_MAX
    q = 9.0 + 14.0 * torch.rand(n, 1, generator=gen)          # 9-23 kW/m2
    M = 0.05 + 1.40 * torch.rand(n, 1, generator=gen)         # LFMC 5-145 %
    if species_aware:
        present = [i for i, s in enumerate(ALL_SPECIES) if s in set(df["species"])]
        pick = torch.tensor(present)[torch.randint(len(present), (n,), generator=gen)]
        z = nn.functional.one_hot(pick, 3).float()
    else:
        z = 0.33 + 0.09 * torch.rand(n, 1, generator=gen)     # thickness 0.33-0.42 mm
    return tt, q, M, z


def train_pinn(df, species_aware, steps=6000, seed=0, w_phys=10.0, log=None,
               lbfgs_iters=2000, w_lbfgs=1000.0):
    """Stage 1: Adam with resampled collocation points (finds the basin).
    Stage 2: L-BFGS on a fixed collocation set with a heavy physics weight, which
    drives the ODE residual to ~1e-6 so the network cannot trade physics for fit."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = IgnitionPINN(species_aware)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    tt, q, M, z = tensors(df, species_aware)
    for step in range(steps):
        opt.zero_grad()
        loss_d = ((model.theta(tt, q, M, z) - 1.0) ** 2).mean()
        loss_p = (model.residual(*collocation(1024, df, species_aware, gen)) ** 2).mean()
        (loss_d + w_phys * loss_p).backward()
        opt.step(); sched.step()
        if log is not None and step % 100 == 0:
            log.append({"step": step, "data": loss_d.item(), "physics": loss_p.item(),
                        "B": model.B.item(), "q_cr": model.qcr.item(),
                        **{f"A{i}": a for i, a in enumerate(torch.exp(model.logA).tolist())}})
    if lbfgs_iters:
        col = collocation(4096, df, species_aware, gen)
        opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters,
                                 history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-9, tolerance_change=1e-12)
        last = {}

        def closure():
            opt2.zero_grad()
            ld = ((model.theta(tt, q, M, z) - 1.0) ** 2).mean()
            lp = (model.residual(col[0].clone(), *col[1:]) ** 2).mean()
            (ld + w_lbfgs * lp).backward()
            last.update(data=ld.item(), physics=lp.item())
            return ld + w_lbfgs * lp

        opt2.step(closure)
        if log is not None:
            log.append({"step": steps + lbfgs_iters, **last, "B": model.B.item(),
                        "q_cr": model.qcr.item(),
                        **{f"A{i}": a for i, a in enumerate(torch.exp(model.logA).tolist())}})
    return model


@torch.no_grad()
def predict_tig(model, df):
    """Ignition time = first time the network temperature reaches theta = 1 (bisection)."""
    _, q, M, z = tensors(df, model.species_aware)
    lo, hi = torch.zeros(len(df), 1), torch.full((len(df), 1), T_MAX)
    for _ in range(40):
        mid = (lo + hi) / 2
        above = model.theta(mid, q, M, z) >= 1.0
        hi = torch.where(above, mid, hi)
        lo = torch.where(above, lo, mid)
    return (hi * T_SCALE).numpy().ravel()


def pinn_params(model):
    A = torch.exp(model.logA).tolist()
    names = ALL_SPECIES if model.species_aware else ["a per mm thickness"]
    return {**{f"A {n}": a for n, a in zip(names, A)},
            "B": model.B.item(), "q_cr": model.qcr.item()}


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20,
                  "display.float_format", "{:.3f}".format)
    quick = "--quick" in sys.argv
    steps = 1500 if quick else 6000      # Adam steps; L-BFGS stage always runs
    df = load_conditions()

    # ---- 1. Verification on all data: does the PINN recover the closed-form fit? ----
    t0 = time.time()
    log = []
    pinn = train_pinn(df, True, steps=steps, log=log)
    ref = ClosedFormODE(True).fit(df)
    p_pinn = pinn_params(pinn)
    p_ref = dict(zip([f"A {s}" for s in ALL_SPECIES] + ["B", "q_cr"], ref.p_))
    ver = pd.DataFrame({"PINN (inverse)": p_pinn, "Closed-form least squares": p_ref})
    ver["difference [%]"] = 100 * (ver.iloc[:, 0] - ver.iloc[:, 1]) / ver.iloc[:, 1]
    print(f"Verification on all 56 conditions ({time.time()-t0:.0f} s training)")
    print(ver.to_string())
    ver.to_csv(os.path.join(OUT, "pinn_verification_parameters.csv"))
    tp = predict_tig(pinn, df)
    print(f"\nIn-sample: PINN R2 = {r2_score(df.t, tp):.3f}, "
          f"closed-form ODE R2 = {r2_score(df.t, ref.predict(df)):.3f}, "
          f"final data loss = {log[-1]['data']:.2e}, physics loss = {log[-1]['physics']:.2e}")
    pd.DataFrame(log).to_csv(os.path.join(OUT, "pinn_training_log.csv"), index=False)
    own = ClosedFormODE(True); own.p_ = np.array(list(p_pinn.values()))
    print(f"Network ignition time vs analytic solution with the PINN's own parameters: "
          f"max difference = {np.abs(tp - own.predict(df)).max():.3f} s")

    # How well do ignition times alone pin down the critical heat flux?
    ylog, prof = np.log(df.t.values), []
    for qc in np.arange(1.0, 10.51, 0.5):
        r = least_squares(lambda p: np.log(ref._t(np.r_[p, qc], df)) - ylog, [20, 20, 20, 1],
                          bounds=([1e-3] * 3 + [0], [1e4] * 3 + [50]))
        prof.append({"q_cr fixed [kW/m2]": qc, "rms log error": np.sqrt(np.mean(r.fun ** 2)),
                     **{f"A {s}": a for s, a in zip(ALL_SPECIES, r.x[:3])}, "B": r.x[3]})
    prof = pd.DataFrame(prof)
    prof.to_csv(os.path.join(OUT, "qcr_profile.csv"), index=False)
    print("\nProfile of the critical heat flux (closed-form ODE, other parameters refitted):")
    print(prof.iloc[::2].to_string(index=False))

    # ---- 2. Validation on the same protocols as the gray-box study ----
    rows = []
    for proto, (folds, species_aware) in splits(df).items():
        if proto.startswith("A."):
            folds = folds[:5]                       # one 5-fold pass (each fold retrains a PINN)
        zoo = {
            "PINN (inverse, ODE)": lambda tr: train_pinn(tr, species_aware, steps=steps),
            "Closed-form ODE (log law)": lambda tr: ClosedFormODE(species_aware).fit(tr),
            "Algebraic physics (previous)": lambda tr: Physics("species" if species_aware else "thick").fit(tr),
        }
        for name, make in zoo.items():
            ys, ps = [], []
            for tr, te in folds:
                m = make(df.iloc[tr])
                p = predict_tig(m, df.iloc[te]) if isinstance(m, IgnitionPINN) else m.predict(df.iloc[te])
                ys.append(df.t.values[te]); ps.append(p)
            y, p = np.concatenate(ys), np.concatenate(ps)
            rows.append({"protocol": proto, "model": name, "n_test": len(y), **metrics(y, p)})
        print(f"\n=== {proto} ===")
        print(pd.DataFrame(rows)[lambda d: d.protocol == proto].drop(columns="protocol").to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "pinn_validation_results.csv"), index=False)

    # ---- 3. Figure ----
    L = pd.DataFrame(log)
    colors = {"Manzanita": "#b5482a", "Oregon grape": "#2a6fb5", "Salal": "#2f8f4e"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.semilogy(L.step, L.data, label="data loss"); ax.semilogy(L.step, L.physics, label="ODE residual loss")
    ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.set_title("(a) PINN training"); ax.legend()
    ax = axes[1]
    for i, s in enumerate(ALL_SPECIES):
        ax.plot(L.step, L[f"A{i}"], color=colors[s], label=f"A {s}")
        ax.axhline(ref.p_[i], color=colors[s], ls=":", lw=1)
    ax.plot(L.step, L.q_cr, "k", label="q_cr"); ax.axhline(ref.p_[-1], color="k", ls=":", lw=1)
    ax.plot(L.step, L.B, "grey", label="B"); ax.axhline(ref.p_[-2], color="grey", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("parameter value")
    ax.set_title("(b) Parameter discovery (dotted = closed-form fit)"); ax.legend(fontsize=7)
    ax = axes[2]
    with torch.no_grad():
        ts = torch.linspace(0, 1, 200).reshape(-1, 1)
        for s, M, qv, ls in [("Manzanita", 1.04, 11, "-"), ("Manzanita", 1.04, 21, "--"),
                             ("Oregon grape", 0.15, 11, "-"), ("Oregon grape", 0.15, 21, "--")]:
            z = torch.tensor([[float(s == k) for k in ALL_SPECIES]]).repeat(200, 1)
            th = pinn.theta(ts, torch.full_like(ts, qv), torch.full_like(ts, M), z)
            ax.plot(ts * T_SCALE, th, color=colors[s], ls=ls, label=f"{s}, LFMC {int(M*100)}%, q={qv}")
            obs = df[(df.species == s) & (df.q == qv) & (np.isclose(df.M, M))].t.values
            ax.plot(obs, [1.0], "o", color=colors[s], ms=7, mec="k")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_ylim(0, 2); ax.set_xlabel("time (s)"); ax.set_ylabel(r"$\theta$ = (T - T$_0$)/(T$_{ig}$ - T$_0$)")
    ax.set_title("(c) Inferred temperature histories (dots = measured)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pinn.png"), dpi=200)
    print(f"\nTotal run time {time.time()-t0:.0f} s")
