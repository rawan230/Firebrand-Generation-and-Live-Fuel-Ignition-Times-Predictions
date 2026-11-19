"""Compact figures sized for the Springer proceedings text width (12.2 cm)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import igt_graybox as ig
import fgt_physics as fg

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                     "legend.fontsize": 5.8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "lines.linewidth": 1.0, "axes.linewidth": 0.6, "font.family": "serif"})
W = 12.2 / 2.54

# ---------------- Fig. 1: ignition time ----------------
df = ig.load_conditions()
phys = ig.Physics("species").fit(df)
par = phys.params().set_index("species")
col = {"Manzanita": "#b5482a", "Oregon grape": "#2a6fb5", "Salal": "#2f8f4e"}
mk = {"Manzanita": "o", "Oregon grape": "s", "Salal": "^"}
fig, ax = plt.subplots(1, 2, figsize=(W, 1.7))
qq = np.linspace(0, 22, 100)
for s in ig.ALL_SPECIES:
    d = df[df.species == s]
    sc = ax[0].scatter(d.q, 1 / d.t, c=d.lfmc, cmap="viridis", s=13, marker=mk[s],
                       edgecolor=col[s], linewidth=0.7, vmin=10, vmax=130, zorder=3)
    A, B, qc = par.loc[s, ["A [kJ/m2]", "B [-]", "q_cr [kW/m2]"]]
    for M, ls in ((0.2, "-"), (1.0, "--")):
        ax[0].plot(qq, np.clip(qq - qc, 0, None) / (A * (1 + B * M)), color=col[s], ls=ls)
cb = fig.colorbar(sc, ax=ax[0], pad=0.02); cb.set_label("Moisture content (%)", fontsize=6)
cb.ax.tick_params(labelsize=6)
ax[0].set_xlim(0, 22); ax[0].set_ylim(0, 0.5)
ax[0].set_xlabel("Heat flux $q$ (kW/m$^2$)"); ax[0].set_ylabel("$1/t_{ig}$ (1/s)")
from matplotlib.lines import Line2D
ax[0].set_title("(a)")
ax[0].legend(handles=[Line2D([], [], color="k", ls="-", label="model, MC 20%"),
                      Line2D([], [], color="k", ls="--", label="model, MC 100%")],
             loc="upper left", frameon=False, handlelength=1.8)
p = phys.predict(df)
for s in ig.ALL_SPECIES:
    m = (df.species == s).values
    ax[1].errorbar(df.t[m], p[m], xerr=df.sd[m], fmt=mk[s], ms=3, color=col[s], alpha=0.85,
                   elinewidth=0.4, label=s)
ax[1].plot([0, 30], [0, 30], "k--", lw=0.6)
ax[1].set_xlim(0, 30); ax[1].set_ylim(0, 30)
ax[1].set_xlabel("Observed $t_{ig}$ (s)"); ax[1].set_ylabel("Predicted $t_{ig}$ (s)")
ax[1].set_title("(b)"); ax[1].legend(loc="upper left", frameon=False)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig1_ignition.pdf")); plt.close(fig)

# ---------------- Fig. 2: firebrand generation time ----------------
df = fg.load_conditions()
law = fg.ScalingLaw("species_n").fit(df)
tab = law.table().set_index("parameter")
colf = {"Douglas-fir": "#b5482a", "Juniper": "#2a6fb5", "Ponderosa": "#2f8f4e", "White oak": "#7a4fa3"}
mkf = {"Douglas-fir": "o", "Juniper": "s", "Ponderosa": "^", "White oak": "D"}
fig, ax = plt.subplots(1, 2, figsize=(W, 1.7), gridspec_kw={"width_ratios": [1.25, 1]})
rng = np.random.default_rng(0)
for s in fg.SPECIES:
    d = df[df.species == s]
    ax[0].scatter(d.d * (1 + 0.035 * rng.standard_normal(len(d))), d.t, s=9, marker=mkf[s],
                  color=colf[s], alpha=0.75, linewidth=0)
    gm = [np.exp(np.log(d[d.d == k].t).mean()) for k in (2, 6)]
    ax[0].plot([2, 6], gm, color=colf[s], label=f"{s}, $n$ = {tab.loc['n ' + s, 'estimate']:.2f}")
dd = np.array([2, 6.0])
for n, ls in ((1.5, ":"), (2.0, "--")):
    ax[0].plot(dd, 7 * (dd / 2) ** n, color="k", lw=0.6, ls=ls, label=f"slope {n}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].minorticks_off()
ax[0].set_xticks([2, 6]); ax[0].set_xticklabels(["2", "6"])
ax[0].set_yticks([5, 10, 20, 40]); ax[0].set_yticklabels(["5", "10", "20", "40"])
ax[0].set_xlim(1.75, 6.9); ax[0].set_ylim(1.7, 75)
ax[0].set_xlabel("Dowel diameter (mm)"); ax[0].set_ylabel("Generation time (s)")
ax[0].set_title("(a)"); ax[0].legend(loc="lower right", frameon=False, handlelength=1.4, labelspacing=0.15, borderaxespad=0.2)
t = tab.loc[["f_M", "f_H", "f_W"]]
ax[1].errorbar(range(3), t["estimate"],
               yerr=[t["estimate"] - t["95% CI low"], t["95% CI high"] - t["estimate"]],
               fmt="o", color="#1b7f5c", capsize=3, ms=4, elinewidth=0.8)
ax[1].axhline(1, color="k", lw=0.5, ls=":")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["15%\nmoisture", "Low\ntemperature", "Low\nwind speed"])
ax[1].set_xlim(-0.5, 2.5); ax[1].set_ylabel("Multiplier on FGT")
ax[1].set_title("(b)")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig2_firebrand.pdf")); plt.close(fig)
print("figures written")
