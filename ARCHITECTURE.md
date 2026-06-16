# sthyb — Architecture

How the code is organized, which equation lives in which module, and how to
extend it. For installation and replication commands, see [README.md](README.md).

## 1. Data flow

```
data/{CITY}_V.csv ──► sthyb.data.data_utils ──► z-scored windows (train/val/test)
                                 │
checkpoints/ ──► sthyb.models.infer ──► backbone predictions ŷ  (frozen weights)
                                 │
        ┌────────────────────────┴───────────────────────────┐
        │  residual hybrids                                   │  GUARD IA
        │  sthyb.hybrid.transforms  (space: level/log1p/Ans.) │  sthyb.hybrid.glm
        │  sthyb.hybrid.star        (ST-AR fit + projection)  │  sthyb.hybrid.guardia
        │  sthyb.hybrid.predictive  (discrete distributions)  │  (dose sweep + gate)
        └────────────────────────┬───────────────────────────┘
                                 │
                  sthyb.eval.probabilistic  (scores all methods)
                  sthyb.eval.metrics        (ALS/MAE/RMSE/Moran/PAI/HAC)
                  sthyb.eval.spatial_diag   (CD / correlogram / ECM)
                                 ▼
                  results/probabilistic/*.csv + figs/
```

## 2. Module ↔ method map

### `sthyb/config.py` — the experiment
Single source of truth: cities, backbones, split sizes, ST-AR lags, NB grid,
GUARD IA settings, paths. `SMOKE=1` (env var) restricts to POA/stgcn for fast
regression checks.

### `sthyb/models/` — backbones
- `base_model.py`, `layers.py`, `trainer.py`, `tester.py` — STGCN and the SAEA
  variants (`--saea none|sparse|structural`), TF1-style graphs.
- Graph-WaveNet and STHSL are self-contained in `scripts/train_*.py`.
- `infer.py` — the backbone catalogue (`_MODEL_SPECS`: checkpoint prefix +
  output tensor per model) and frozen-weight inference. **Adding a backbone =
  registering it here** + a checkpoint dir following `_DEFAULT_SUBDIR`.

### `sthyb/hybrid/` — the correction methods
- **`transforms.py`** — variance-stabilizing spaces as data
  (`Transform(tag, res, fwd, inv, mean, ppf_edge)`):
  `LEVEL` (identity), `LOG1P`, `ANSCOMBE` (g(y)=2√(y+3/8)).
  The hybrid models residuals e = fwd(y) − fwd(ŷ); `mean` is the
  Jensen-correct E[y] back-transform. **Adding a space = one `Transform`.**
- **`star.py`** — the seasonal ST-AR (Shoesmith 2013) on the residual panel:

      e_{i,t} = c_i + Σ_j φ_j e_{i,t−j} + Σ_l ψ_l (We)_{i,t−l} + u_{i,t}

  `within_ols` (per-region FE + pooled OLS; classical and Driscoll-Kraay
  covariances — the DK one feeds the robust Chow test), `predict_correction`
  (test-time projection from observed val→test lags only — the anti-leak
  boundary), `shrink_var` (per-node predictive variance, shrunk toward the
  panel mean), `nb_alpha_mle` (NB2 dispersion by grid MLE).
- **`predictive.py`** — every method is scored as a discrete distribution over
  integer counts. `Predictive` (shared: pmf via cdf differences, log score,
  randomized PIT, central-interval coverage); `TransformPredictive`
  (Gaussian in a Transform space, discretized at half-integer edges
  fwd(k+0.5)); `CountPredictive` (native Poisson/NB2).
- **`glm.py`** — the GUARD IA engine: per-cell Poisson calibration GLM

      log E[y_it] = β0 + α·log(ŷ_it) + β1·ε_{i,t−1} + β2·(Wε)_{i,t−1}

  (ŷ as a *free* covariate: α≠1 absorbs systematic bias). Training data only.
- **`guardia.py`** — dose and gate on top of the GLM: sweep c ∈ [0,2] scaling
  β2, record per-node validation loss L[c,i] and |LISA| A[c,i]; select c by
  `global` (Pareto knee on aggregate loss×|LISA|), `pernode` (argmin loss) or
  `lisapareto` (per-node Pareto knee); per-node gate keeps the backbone
  wherever calibration does not beat it on validation.

### `sthyb/eval/` — scoring and diagnostics
- **`metrics.py`** — Moran's I (+ analytical p per step, t-test across steps),
  LISA (mean and per-node), PAI@k, MAE/RMSE, Newey-West HAC test (the
  Giacomini-White / Diebold-Mariano statistic).
- **`probabilistic.py`** — the orchestrator: loads each backbone, builds the
  residual panels per transform space, fits the ST-AR per fit split
  (train/val/pooled), runs GUARD IA, and scores the 18 methods on the unified
  discrete log score + point/spatial metrics. Writes the 5 CSVs.
  Sanity gates abort the run if any PMF loses mass or a log score is non-finite.
- **`spatial_diag.py`** — residual cross-sectional dependence: Pesaran CD,
  correlogram by graph hop with a node-permutation null band, λ_max vs the
  Marchenko-Pastur edge, and error-correlation-matrix figures.
- **`plotting.py`** — NeurIPS figure style (`set_style()`), one color per
  method family (`PALETTE`, `method_color()`), `save_fig()` (PDF+PNG).

### `scripts/` — thin entry points
Each is ≤ 10 lines: parse nothing, import the package, run. The training
scripts (`train_stgcn/gwavenet/sthsl`) are the original SAEA-protocol trainers.

### `tests/` — numerical invariants
Six fast tests (no TF, no checkpoints): transform round-trip, PMF mass,
PIT uniformity, NB-MLE recovery, LISA consistency, HAC power.

## 3. Design decisions

- **Methods as data, not branches.** Transform spaces are registry entries;
  the evaluation loop is branch-free. Same for the backbone catalogue.
- **One config.** No hyperparameter lives outside `config.py`; the paper ↔
  code mapping is auditable in one screen.
- **Anti-leak by construction.** Estimation consumes train (or train+val for
  `pooled`); test enters only through `predict_correction` /
  `guardia_predict`, whose test-time regressors are lagged observed values.
  The boundaries are marked with comments at each site.
- **Unified probabilistic ruler.** Every method — Gaussian-in-transform,
  Poisson, NB — is reduced to a PMF over the integers before scoring, so log
  scores are comparable across families (no density/PMF mixing).
- **Golden-regression development.** Every refactor of this repo was verified
  byte-identical on the 5 output CSVs (smoke + full 9-cell) before landing.

## 4. How to extend

| Goal | Touch |
|---|---|
| New city | `data/{NAME}_{V,W,W2}.csv` + one line in `config.CITIES` |
| New backbone | train script + `_MODEL_SPECS`/`_DEFAULT_SUBDIR` in `models/infer.py` + `config.BACKBONES` |
| New transform space | one `Transform` in `hybrid/transforms.py` (+ add to `TRANSFORMS`) |
| New count distribution | subclass `Predictive` (provide `cdf`, `ppf`) |
| New metric | function in `eval/metrics.py`, add to the row dict in `eval/probabilistic.py` |
| Different lags / NB grid / gate loss | `config.py` |
