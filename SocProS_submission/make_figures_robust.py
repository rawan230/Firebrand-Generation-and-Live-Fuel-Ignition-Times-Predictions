"""Forest plots of paired RMSE differences (baseline minus physics) with 95 % bootstrap CIs.
Reads ../robust_outputs/{IGT,FGT}_robust_results.csv written by robust_eval.py."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "robust_outputs")
plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
                     "font.family": "serif"})
W = 12.2 / 2.54
GOOD, BAD, NEU = "#1b7f5c", "#b5482a", "#7a8794"


def forest(csv, protos, titles, models, labels, fname, height, xlabels=None):
    d = pd.read_csv(os.path.join(R, csv))
    fig, axes = plt.subplots(1, len(protos), figsize=(W, height), sharey=True)
    for ax, p, t in zip(axes, protos, titles):
        sub = d[d.protocol == p].set_index("model")
        for i, m in enumerate(models):
            r = sub.loc[m]
            lo, hi, v = r.dRMSE_lo, r.dRMSE_hi, r.dRMSE_vs_ref
            c = GOOD if lo > 0 else BAD if hi < 0 else NEU
            ax.plot([lo, hi], [i, i], color=c, lw=1.2, solid_capstyle="butt")
            ax.plot(v, i, "o", color=c, ms=2.8)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(t)
        ax.set_ylim(len(models) - 0.4, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(labels)
    fig.supxlabel("RMSE of baseline minus RMSE of the physics model (s)", fontsize=6.5, y=0.0)
    fig.tight_layout(pad=0.35, w_pad=0.4)
    fig.savefig(os.path.join(HERE, fname)); plt.close(fig)


igt_models = ["Log-linear (5)", "Power law in q, 1+M (5)", "Quadratic ridge", "GP, linear mean",
              "GP (default)", "RF (tuned)", "XGBoost (tuned)", "XGBoost (tuned, monotone)",
              "MLP (tuned)"]
igt_labels = ["Log-linear", "Power law", "Quadratic ridge", "GP, linear mean", "GP (default)",
              "RF (tuned)", "XGBoost (tuned)", "XGBoost, monotone", "MLP (tuned)"]
forest("IGT_robust_results.csv",
       ["A. Normal (5-fold x10)", "C. New high heat flux", "D. New low heat flux",
        "E. New high moisture"],
       ["(a) Normal", "(b) High flux", "(c) Low flux", "(d) High moisture"],
       igt_models, igt_labels, "fig3_forest_igt.pdf", 2.25)

fgt_models = ["Log-linear (8)", "Quadratic ridge", "GP, linear mean", "GP (default)",
              "RF (tuned)", "XGBoost (tuned)", "XGBoost (tuned, monotone)", "MLP (tuned)"]
fgt_labels = ["Log-linear / shared $n$", "Quadratic ridge", "GP, linear mean", "GP (default)",
              "RF (tuned)", "XGBoost (tuned)", "XGBoost, monotone", "MLP (tuned)"]
forest("FGT_robust_results.csv",
       ["A. Normal (5-fold x10)", "C. New temperature-wind pair", "D. Species seen at 2 mm only"],
       ["(a) Normal", "(b) Temp.-wind pair", "(c) Seen at 2 mm"],
       fgt_models, fgt_labels, "fig4_forest_fgt.pdf", 2.05)
print("forest figures written")
