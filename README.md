# sthyb — Spatio-Temporal Hybrid residual correction for crime forecasting

Probabilistic residual-correction hybrids on top of spatio-temporal neural
backbones (STGCN/SAEA, Graph-WaveNet, STHSL), evaluated on three crime datasets
(São Paulo, Porto Alegre, Bahía) with a unified discrete log score (ALS),
PAI / hit-rate, Moran's I and cross-sectional-dependence diagnostics.

**This README is the replication guide.** For the code architecture and the
module ↔ method mapping, see [ARCHITECTURE.md](ARCHITECTURE.md). For the
methods and experimental protocol in short-paper form (with formulas), see
[docs/method_hybrids.md](docs/method_hybrids.md) and
[docs/method_guardia.md](docs/method_guardia.md). For the full results tables
and critical analysis, see [docs/RESULTS.md](docs/RESULTS.md).

---

## 1. Setup (~5 min)

```bash
conda create -n sthyb python=3.9 -y && conda activate sthyb
pip install -r requirements.txt
pip install -e .                       # makes `import sthyb` resolve
```

All commands below are run **from the repo root** (paths are relative to it).
TensorFlow 2.10 runs in `tf.compat.v1` (graph) mode; GPU is optional — CPU works
for everything except retraining the backbones.

## 2. Data

The datasets are **not tracked in git** (see `.gitignore`); place them under
`data/` as `{CITY}_{V,W,W2}.csv` (+ `{CITY}_mask{,2}.npy` for SAEA-structural):
`V` = daily counts (T days × N cells), `W` = spatial adjacency (row-normalizable),
`W2` = second-order weights. Cities: `SP_CRIME` (N=1445), `POA_CRIME` (94),
`BA_LESIONES` (74). Download: _TBD (release/Zenodo)_. Rebuild from raw sources:
see `data_prep/`.

## 3. Checkpoints

Trained backbones (~380 MB, final training step only) go under `checkpoints/`
(git-ignored). Download: _TBD (release/Zenodo)_. Or retrain (GPU, hours/model):

```bash
python scripts/train_stgcn.py    --dataset SP_CRIME --n_route 1445 --n_his 7 --n_pred 1 --batch_size 8 --epoch 300 --kt 2
python scripts/train_gwavenet.py --dataset SP_CRIME --n_route 1445 --n_his 7 --n_pred 1 --batch_size 8 --loss mse
python scripts/train_sthsl.py    --dataset SP_CRIME --n_route 1445 --n_his 7 --batch_size 8 --epoch 300 --loss mse
```

## 4. Smoke test (~15 s) — verify the pipeline before committing to a full run

```bash
SMOKE=1 python scripts/run_probabilistic.py          # POA_CRIME / stgcn only
```
PowerShell: `$env:SMOKE=1; python scripts/run_probabilistic.py; $env:SMOKE=$null`

## 5. Reproduce the results (command → artifact)

| Command | Output | What it feeds | Time (CPU) |
|---|---|---|---|
| `python scripts/run_probabilistic.py` | `results/probabilistic/als_master.csv` | main results table (ALS/MAE/RMSE/MI/PAI per method) | ~11 min |
| | `diag_residuals.csv` | residual diagnostics (ACF, ST-AR coefficients, robust Chow) | (same run) |
| | `gw_dm_tests.csv` | significance tests (Giacomini-White / Diebold-Mariano, HAC) | (same run) |
| | `calibration.csv` | PIT histograms + 80/95% coverage | (same run) |
| | `per_node.csv` | per-cell errors, GUARD IA doses & gates (maps) | (same run) |
| `python scripts/run_spatial_diag.py` | `spatial_diag.csv` + `figs/*.png` | Pesaran CD, hop correlograms, eigenvalue/ECM figures | ~5 min |

A `README.md` describing every column is written next to the CSVs.
Methods scored (18): `hybrid-{train,val,pooled}` × {Anscombe, additive(`-add`),
log1p(`-log`)}, `base+{Poisson,NB,Gauss}`, `guardia{,-nodec,-lisac}{,+NB}`.

## 6. Determinism

- All stochastic steps are seeded (`np.random.seed(0)` for the randomized PIT;
  permutation nulls in `spatial_diag` likewise).
- Estimators (OLS, GLM-IRLS, grid MLE) are deterministic given the data.
- Caveat: multi-threaded BLAS / parallel GLM fitting can flip floating-point
  summation order; consecutive runs are byte-identical in our testing, but
  across machines expect agreement to ~1e-9 — far below the 3–4 significant
  digits reported. For bit-exact runs set `OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1` and `GUARDIA_NJOBS=1` in `sthyb/config.py` (slower).

## 7. Configuration

Everything lives in **`sthyb/config.py`**: cities, backbones, train/val/test
split sizes (110/110 days), ST-AR lags ({1,7,14} own / {1,7} spatial — weekly
seasonality), NB dispersion grid, GUARD IA gate loss, and all paths.
Changing an experiment = editing that one file.

## 8. Tests (~10 s)

```bash
pip install pytest && pytest tests/ -q
```
Six fast unit tests of the numerical invariants the paper relies on: transform
round-trip, PMF mass ≈ 1, PIT uniformity under correct specification, NB-MLE
dispersion recovery, per-node LISA consistency, HAC test power.
