"""Write LaTeX table bodies from robust_outputs/*.csv (no numbers are typed by hand)."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "robust_outputs")


def num(x, nd=2):
    s = f"{abs(x):.{nd}f}"
    return ("$-$" + s) if x < 0 and round(abs(x), nd) != 0 else s


def cell(r, best):
    mark = ""
    if r.dRMSE_lo > 0:
        mark = "$^{\\dagger}$"          # reference significantly better
    elif r.dRMSE_hi < 0:
        mark = "$^{\\ddagger}$"         # this model significantly better
    rm = num(r.RMSE)
    if abs(r.RMSE - best) < 1e-9:
        rm = f"\\textbf{{{rm}}}"
    return f"{rm} ({num(r.R2)}){mark}"


def body(csv, protos, rows):
    d = pd.read_csv(os.path.join(R, csv))
    best = {p: d[(d.protocol == p) & d.model.isin([m for m, _ in rows])].RMSE.min() for p in protos}
    out = []
    for m, label in rows:
        cells = []
        for p in protos:
            sub = d[(d.protocol == p) & (d.model == m)]
            if sub.empty:
                cells.append("n/a")
            else:
                r = sub.iloc[0]
                # the reference model itself carries no marker
                if r.dRMSE_vs_ref == 0 and r.dRMSE_lo == 0 and r.dRMSE_hi == 0:
                    rm = num(r.RMSE)
                    rm = f"\\textbf{{{rm}}}" if abs(r.RMSE - best[p]) < 1e-9 else rm
                    cells.append(f"{rm} ({num(r.R2)})")
                else:
                    cells.append(cell(r, best[p]))
        out.append(f"{label} & " + " & ".join(cells))
    return " \\\\\n".join(out)


igt_rows = [("Heat balance (simple, 5)", "Heat balance (reference)"),
            ("Log-linear (5)", "Log-linear regression"),
            ("Power law in q, 1+M (5)", "Power law in $q$, $1{+}M$"),
            ("Quadratic ridge", "Quadratic ridge"),
            ("GP, linear mean", "GP, linear mean"),
            ("GP (default)", "GP (default)"),
            ("RF (tuned)", "Random Forest (tuned)"),
            ("XGBoost (tuned)", "XGBoost (tuned)"),
            ("XGBoost (tuned, monotone)", "XGBoost (tuned, monotone)"),
            ("MLP (tuned)", "MLP (tuned)")]
igt_protos = ["A. Normal (5-fold x10)", "C. New high heat flux", "D. New low heat flux",
              "E. New high moisture"]
open(os.path.join(HERE, "tab_igt_body.tex"), "w", newline="\n").write(
    body("IGT_robust_results.csv", igt_protos, igt_rows) + "\n")

fgt_rows = [("Law, species n (11)", "Power law, species $n_s$ (11)"),
            ("Law, shared n (8)", "Power law, shared $n$ (8)$^{a}$"),
            ("Law, one n (5)", "Power law, one $n$ (5)$^{a}$"),
            ("Quadratic ridge", "Quadratic ridge"),
            ("GP, linear mean", "GP, linear mean"),
            ("GP (default)", "GP (default)"),
            ("RF (tuned)", "Random Forest (tuned)"),
            ("XGBoost (tuned)", "XGBoost (tuned)"),
            ("XGBoost (tuned, monotone)", "XGBoost (tuned, monotone)"),
            ("MLP (tuned)", "MLP (tuned)")]
fgt_protos = ["A. Normal (5-fold x10)", "B. New species", "C. New temperature-wind pair",
              "D. Species seen at 2 mm only"]
# one row for the power law per column: merge the three law variants by availability
d = pd.read_csv(os.path.join(R, "FGT_robust_results.csv"))
best = {p: d[d.protocol == p][~d[d.protocol == p].model.str.startswith("Log-linear")].RMSE.min()
        for p in fgt_protos}
lines = []
for m, label in fgt_rows:
    cells = []
    for p in fgt_protos:
        sub = d[(d.protocol == p) & (d.model == m)]
        if sub.empty:
            cells.append("n/a")
            continue
        r = sub.iloc[0]
        if r.dRMSE_lo == 0 and r.dRMSE_hi == 0 and r.dRMSE_vs_ref == 0:
            rm = num(r.RMSE)
            rm = f"\\textbf{{{rm}}}" if abs(r.RMSE - best[p]) < 1e-9 else rm
            cells.append(f"{rm} ({num(r.R2)})")
        else:
            cells.append(cell(r, best[p]))
    lines.append(f"{label} & " + " & ".join(cells))
open(os.path.join(HERE, "tab_fgt_body.tex"), "w", newline="\n").write(" \\\\\n".join(lines) + "\n")

# leakage table
L = pd.read_csv(os.path.join(R, "leakage_results.csv"))
names = [("Physics", "Physics model"), ("Log-linear", "Log-linear regression"),
         ("GP (default)", "GP"), ("RF (default)", "Random Forest"),
         ("XGBoost (default)", "XGBoost")]
lines = []
for m, label in names:
    cells = []
    for ds in ("IGT", "FGT"):
        for sch in ("Random rows (leaky)", "By condition"):
            r = L[(L.dataset == ds) & (L.scheme == sch) & (L.model == m)].iloc[0]
            cells.append(f"{num(r.RMSE)} ({num(r.R2)})")
    lines.append(f"{label} & " + " & ".join(cells))
open(os.path.join(HERE, "tab_leak_body.tex"), "w", newline="\n").write(" \\\\\n".join(lines) + "\n")
print(open(os.path.join(HERE, "tab_igt_body.tex")).read())
print(open(os.path.join(HERE, "tab_fgt_body.tex")).read())
print(open(os.path.join(HERE, "tab_leak_body.tex")).read())
