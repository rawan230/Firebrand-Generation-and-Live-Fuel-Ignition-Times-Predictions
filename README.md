# Firebrand Generation and Live-Fuel Ignition Times: Predictions

Code and results for physics-based and machine-learning models of firebrand
generation time (FGT) and live-shrub-leaf ignition time (IGT), evaluated at the
level of experimental conditions (not raw rows).

## Data
- `FGT_Edited.csv`, `IGT_Edited.csv`: each condition is stored as three rows (mean, mean ± SD).

## Scripts (run from this folder, Python 3.12 with numpy, pandas, scikit-learn, xgboost, torch, scipy, joblib)
- `igt_graybox.py`: heat-balance model of IGT, gray-box GP correction -> `igt_graybox_outputs/`
- `igt_pinn.py`: PINN for the heat-balance equation -> `igt_graybox_outputs/`
- `fgt_physics.py`: power-law model of FGT -> `fgt_physics_outputs/`
- `robust_eval.py [igt|fgt|leak|all]`: ten-baseline comparison, extrapolation tests,
  bootstrap intervals, leakage of random splits -> `robust_outputs/`
- `SocProS_submission/make_figures*.py`, `make_tables.py`: figures and LaTeX tables from the outputs
  (they write into `SocProS_submission/`, which holds only these scripts here).

Set `OMP_NUM_THREADS=1` before running `robust_eval.py`.
