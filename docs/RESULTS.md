# Results

Evaluation of the residual hybrids and GUARD IA on three crime datasets
(São Paulo, Porto Alegre, Bahía) × three MSE-trained backbones (STGCN,
Graph-WaveNet, STHSL), held-out test = last 110 days. Reproduce with
`python scripts/run_probabilistic.py`. All numbers below are from the full
9-cell run (`results/probabilistic/als_master.csv`).

Metrics: **ALS** = mean discrete log score (lower = better, the headline
probabilistic metric); **MAE/RMSE** point error; **PAI@10%** hotspot
concentration; **I** = Moran's I of test residuals (×10³). For each
city×backbone we report the best variant of each model family (by ALS):
*base* = single model + Poisson; *additive/log1p/Anscombe* = ST-AR hybrid in
that transform space; *guardia* = GUARD IA (best of the 6 dose×distribution
variants).

## 1. Headline — best method per cell (by ALS)

| City/Backbone | Best method | ALS |
|---|---|---|
| SP / stgcn | `guardia-lisac+NB` | **0.384** |
| SP / gwavenet | `guardia-lisac+NB` | **0.367** |
| SP / sthsl | `guardia-nodec+NB` | **0.368** |
| POA / stgcn | `guardia-nodec+NB` | **1.659** |
| POA / gwavenet | `guardia-lisac+NB` | **1.615** |
| POA / sthsl | `guardia-lisac+NB` | **1.617** |
| BA / stgcn | `hybrid-add-pooled` (additive) | **0.664** |
| BA / gwavenet | `guardia-nodec` | **0.598** |
| BA / sthsl | `hybrid-train` (Anscombe) | **0.605** |

GUARD IA wins ALS in **7/9** cells (6 of them under the NB distribution); the
hybrid wins the two remaining BA cells.

## 2. Full metrics (best variant per family)

ALS / MAE / RMSE / PAI@10 / I(×10³); **bold** = best ALS in the cell.

| Cell | Model | RMSE | MAE | PAI10 | I·10³ | ALS |
|---|---|---|---|---|---|---|
| **SP/stgcn** | base | 0.706 | 0.236 | 5.54 | +1.23 | 0.433 |
| | additive | 0.713 | 0.234 | 5.56 | +1.00 | 0.414 |
| | log1p | 0.712 | 0.212 | 5.54 | +1.93 | 0.393 |
| | Anscombe | 0.708 | 0.213 | 5.55 | +1.57 | 0.393 |
| | **guardia-lisac+NB** | 0.703 | 0.234 | 5.54 | +1.13 | **0.384** |
| **SP/gwavenet** | base | 0.713 | 0.259 | 5.62 | +1.75 | 0.399 |
| | additive | 0.713 | 0.234 | 5.61 | +1.04 | 0.414 |
| | log1p | 0.721 | 0.209 | 5.60 | +1.76 | 0.391 |
| | Anscombe | 0.713 | 0.210 | 5.60 | +1.44 | 0.391 |
| | **guardia-lisac+NB** | 0.680 | 0.236 | 5.54 | +2.17 | **0.367** |
| **SP/sthsl** | base | 0.712 | 0.247 | 5.49 | +1.97 | 0.382 |
| | additive | 0.708 | 0.239 | 5.55 | +2.19 | 0.411 |
| | log1p | 0.717 | 0.210 | 5.52 | +3.05 | 0.390 |
| | Anscombe | 0.710 | 0.211 | 5.52 | +2.75 | 0.389 |
| | **guardia-nodec+NB** | 0.732 | 0.241 | 5.50 | +1.73 | **0.368** |
| **POA/stgcn** | base | 2.113 | 1.371 | 3.497 | +5.49 | 1.781 |
| | additive | 2.111 | 1.365 | 3.497 | +4.62 | 1.717 |
| | log1p | 2.118 | 1.339 | 3.497 | +3.95 | 1.683 |
| | Anscombe | 2.127 | 1.351 | 3.497 | +3.21 | 1.670 |
| | **guardia-nodec+NB** | 2.104 | 1.356 | 3.497 | +5.02 | **1.659** |
| **POA/gwavenet** | base | 2.012 | 1.333 | 3.490 | +4.08 | 1.681 |
| | additive | 2.007 | 1.325 | 3.477 | +3.73 | 1.672 |
| | log1p | 2.011 | 1.296 | 3.485 | +4.06 | 1.647 |
| | Anscombe | 2.003 | 1.301 | 3.485 | +3.14 | 1.633 |
| | **guardia-lisac+NB** | 1.972 | 1.301 | 3.477 | +2.14 | **1.615** |
| **POA/sthsl** | base | 1.999 | 1.337 | 3.497 | +3.39 | 1.670 |
| | additive | 1.980 | 1.308 | 3.497 | +1.69 | 1.663 |
| | log1p | 2.005 | 1.285 | 3.497 | +1.65 | 1.638 |
| | Anscombe | 1.984 | 1.285 | 3.497 | +0.72 | 1.624 |
| | **guardia-lisac+NB** | 1.983 | 1.308 | 3.497 | +0.12 | **1.617** |
| **BA/stgcn** | base | 0.603 | 0.380 | 2.873 | −14.39 | 0.872 |
| | **additive** | 0.602 | 0.374 | 2.873 | −14.42 | **0.664** |
| | log1p | 0.603 | 0.354 | 2.873 | −14.30 | 0.669 |
| | Anscombe | 0.601 | 0.356 | 2.873 | −14.33 | 0.665 |
| | guardia-lisac+NB | 0.601 | 0.373 | 2.873 | −14.43 | 0.785 |
| **BA/gwavenet** | base | 0.568 | 0.373 | 2.873 | −14.48 | 0.629 |
| | additive | 0.563 | 0.358 | 2.873 | −14.50 | 0.619 |
| | log1p | 0.575 | 0.339 | 2.873 | −14.36 | 0.615 |
| | Anscombe | 0.572 | 0.340 | 2.873 | −14.41 | 0.612 |
| | **guardia-nodec** | 0.559 | 0.364 | 2.873 | −14.49 | **0.598** |
| **BA/sthsl** | base | 0.618 | 0.467 | 2.856 | −12.60 | 0.719 |
| | additive | 0.556 | 0.366 | 2.873 | −14.45 | 0.617 |
| | log1p | 0.567 | 0.345 | 2.873 | −14.35 | 0.607 |
| | **Anscombe** | 0.564 | 0.345 | 2.873 | −14.39 | **0.605** |
| | guardia-lisac+NB | 0.567 | 0.362 | 2.665 | −14.48 | 0.608 |

## 3. Average rank across the 9 cells (1 = best)

| Model | RMSE | MAE | PAI | \|I\| | ALS | **mean** |
|---|---|---|---|---|---|---|
| **Anscombe** | 2.78 | 2.00 | 2.72 | **2.44** | 2.22 | **2.43** |
| guardia | **1.89** | 3.22 | 3.83 | 3.11 | **1.56** | 2.72 |
| log1p | 4.22 | **1.11** | 2.83 | 3.00 | 2.78 | 2.79 |
| additive | 2.44 | 3.67 | **2.50** | 3.11 | 3.89 | 3.12 |
| base | 3.67 | 5.00 | 3.11 | 3.33 | 4.56 | 3.93 |

Per-metric winner: RMSE → guardia, MAE → log1p, PAI → additive,
|I| → Anscombe, ALS → guardia.

## 4. Mean gain vs the single model (%)

(+ = improvement; RMSE/MAE/ALS/|I| = reduction, PAI = increase)

| Model | RMSE | MAE | PAI | \|I\| | ALS |
|---|---|---|---|---|---|
| additive | +1.3 | +4.9 | +0.2 | +11.9 | +4.2 |
| log1p | +0.4 | **+10.6** | +0.1 | **−4.9** | +6.7 |
| Anscombe | +1.0 | +10.2 | +0.1 | +9.0 | +7.1 |
| guardia | **+1.7** | +4.9 | **−0.9** | **+14.9** | **+7.5** |

## 5. Significance (Giacomini–White on ALS, Newey–West HAC)

Best Anscombe hybrid vs `guardia-lisac+NB` (t > 0 ⇒ GUARD IA better):

| Cell | t | p | |
|---|---|---|---|
| SP/stgcn | +4.13 | <0.001 | GUARD IA ✓ |
| SP/gwavenet | +11.51 | <0.001 | GUARD IA ✓ |
| SP/sthsl | +14.17 | <0.001 | GUARD IA ✓ |
| POA/gwavenet | +4.45 | <0.001 | GUARD IA ✓ |
| POA/stgcn | +2.30 | 0.021 | GUARD IA ✓ |
| POA/sthsl | +1.74 | 0.082 | n.s. |
| BA/gwavenet | +4.76 | <0.001 | GUARD IA ✓ |
| BA/stgcn | −8.42 | <0.001 | **hybrid ✓** |
| BA/sthsl | −0.68 | 0.496 | n.s. |

The hybrids beat `base+Poisson` significantly almost everywhere (the one
exception is SP/sthsl, where the bare Poisson is already competitive).

## 6. Calibration (central-interval coverage; nominal 80% / 95%)

| Method | cov80 | cov95 |
|---|---|---|
| base+Poisson | 0.93 | 0.98 |
| guardia | 0.93 | 0.98 |
| guardia+NB | 0.95 | 0.99 |
| hybrid-val | 0.94 | 0.98 |

All methods are **over-covered** (intervals slightly too wide), i.e.
conservative rather than overconfident — partly a discreteness artifact of
low-count central intervals. The randomized-PIT histograms (`calibration.csv`)
are close to uniform for the well-specified Poisson.

## 7. Residual spatial dependence (Moran's I)

Moran's I of the residuals is **significant only in São Paulo** (p ≈ 0.001–0.003);
in Porto Alegre it is positive but non-significant (p ≈ 0.24–0.33) and in Bahía
it is slightly negative and non-significant (p = 1.000). Even in SP, no method
reaches spatial independence — the additive hybrid attains the lowest |I|.
The `spatial_diag` Pesaran CD test, which is more powerful, does flag
cross-sectional dependence in POA/BA that Moran's I misses.

## 8. Ablation — distribution vs. correction

We elect Anscombe as the best hybrid (§2–4); this ablation explains **what
drives its gain**. Each hybrid adds two things over the single model + Poisson
baseline: a **distribution** (the transform-Gaussian PMF) and a **correction**
(the ST-AR). We isolate them with `base+<g>` = single model + transform-Gaussian
with **no correction** (point unchanged), giving a 3-step ladder:

`base+Poisson` → `base+Anscombe` (distribution) → `hybrid-Anscombe` (correction).

**Key fact:** `base+Poisson` and `base+Anscombe` share the same point forecast,
so they have **identical MAE/RMSE/PAI** — the distribution can only move the
**ALS**. Hence the decomposition is non-trivial only for ALS; for the point
metrics the entire gain is the correction.

Anscombe, averaged over the 9 cells:

| Metric | base+Pois | base+Ansc | hybrid | **distribution** | **correction** |
|---|---|---|---|---|---|
| ALS | 0.952 | 0.909 | 0.887 | **+0.043 (≈⅔)** | +0.022 (≈⅓) |
| MAE | 0.667 | 0.667 | 0.621 | +0.000 | **+0.046 (100%)** |
| RMSE | 1.116 | 1.116 | 1.106 | +0.000 | +0.010 (100%) |
| PAI@10 | 3.969 | 3.969 | 3.980 | +0.000 | +0.011 (100%) |

(Positive = improvement.) So the "is Anscombe just a better distribution?"
concern is **specific to ALS**: there ≈⅔ of the gain is the variance-stabilized
PMF and ≈⅓ the ST-AR correction. On **MAE/RMSE/PAI the gain is 100% the
correction** — no distributional confound. The additive baseline is degenerate
here (its level-Gaussian PMF is *worse* than Poisson, −0.020 ALS, so its whole
ALS gain comes from the correction); log1p mirrors Anscombe (≈⅓ correction).

## 9. Takeaways

1. **GUARD IA + NB is the strongest probabilistic model.** It wins ALS in 7/9
   cells and is significantly better than the best hybrid on all of SP and most
   of POA. The **NB wrapper is essential**: on count data the native Poisson is
   overconfident, and `guardia` → `guardia+NB` improves ALS consistently.
2. **The per-node dose (`-nodec`, `-lisac`) is faithful but barely moves the
   numbers.** The three GUARD IA dose modes give near-identical ALS even though
   the selected per-cell doses differ in 60–80% of cells — the spatial-lag
   signal in the residuals is too weak to reward heterogeneous dosing. This is a
   genuine (negative) finding, consistent with the weak/insignificant Moran's I.
3. **Different metrics, different champions** — coherent with loss geometry:
   ALS → GUARD IA+NB, MAE → log1p hybrid, |I| → Anscombe, PAI → additive. PAI
   barely moves for anyone (granularity + small N).
4. **The single model is dominated** on every metric (mean rank 3.93, last).
5. **Anscombe is the most balanced corrector** (best mean rank, 2.43): never the
   worst, leads |I|, second on MAE and ALS.
6. **BA/stgcn is the exception** where the additive hybrid beats GUARD IA — worth
   a closer look (the GUARD IA Poisson calibration underperforms there).

> Reproducibility note: results are deterministic up to ~1e-9 floating-point
> reassociation (multi-threaded BLAS); see the README determinism section.

---

# Part A — Hybrids vs. single models (no GUARD IA)

## A1. Statistical significance — best hybrid vs. single model
Giacomini–White (ALS) / Diebold–Mariano (absolute, squared error), Newey–West HAC,
on the per-day loss differentials over the 9 cells:

| Metric | significant (p<0.05) | favouring the hybrid |
|---|---|---|
| ALS (probabilistic) | **9/9** | 8/9 (only SP/sthsl favours base) |
| absErr (MAE-level) | **9/9** | **9/9** |
| sqErr (RMSE-level) | 4/9 | 2/9 (mixed) |

→ The hybrid's improvement is **significant and unanimous on MAE and ALS**, but
**not significant on RMSE** (the ~1% RMSE change is noise) — the loss-geometry story.

## A2. Average rank — single + hybrids only (1 = best, 9 cells)

| Model | RMSE | MAE | PAI | ALS | **mean** |
|---|---|---|---|---|---|
| **Anscombe** | 2.00 | 1.89 | 2.50 | **1.44** | **1.96** |
| log1p | 3.44 | **1.11** | 2.61 | 2.00 | 2.29 |
| additive | **1.78** | 3.00 | **2.22** | 3.00 | 2.50 |
| single model | 2.78 | 4.00 | 2.67 | 3.56 | 3.25 |

The single model is **last**; Anscombe is the best hybrid (also wins ALS in 5/9
cells, log1p in 2, additive/single 1 each).

## A3. Conclusions — hybrids
1. **Correction always helps**: the single model is dominated; it loses ALS in 8/9.
2. **Gains concentrate in MAE and ALS, not RMSE** (mean gain over the single model:
   MAE additive +4.9% / log1p +10.6% / Anscombe +10.2%; ALS +4.2 / +6.7 / +7.1%;
   RMSE only +0.4–1.3%). The backbones are MSE-trained → already good at RMSE, with
   slack only on MAE/distribution. **The hybrid fixes the metric the backbone did
   not optimise.** Sell it primarily as a **probabilistic corrector**.
3. **The transform space is the contribution** (additive is the baseline): Anscombe
   and log1p beat additive on ALS/MAE. **Anscombe is the best variant** (best mean
   rank, best ALS); log1p is the MAE specialist.

---

# Part B — GUARD IA (residual correction)

## B1. SAEA vs. GUARD IA on the STGCN backbone (gains over STGCN base)
ΔMAE% / ΔRMSE% / ΔPAI% (positive = improvement) and ΔI (Moran; negative = less
clustering). Red/negative = worse.

| Model | SP ΔMAE% | SP ΔRMSE% | SP ΔPAI% | SP ΔI | POA ΔMAE% | POA ΔRMSE% | POA ΔPAI% | POA ΔI | BA ΔMAE% | BA ΔRMSE% | BA ΔPAI% | BA ΔI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +SAEA-sparse | −1.56 | −0.69 | −0.18 | −0.0004 | +0.80 | +0.56 | −0.34 | −0.0050 | +1.46 | +0.77 | 0.00 | 0.0000 |
| +SAEA-structural | −77.36 | −22.89 | −1.26 | +0.0027† | +1.05 | +2.49 | −0.34 | −0.0127 | −9.86 | −8.45 | −0.59 | +0.0001 |
| **+GUARD IA** | **+4.50** | **+0.95** | −0.07 | 0.0000 | **+1.80** | **+0.44** | 0.00 | −0.0023 | **+4.88** | **+0.53** | 0.00 | −0.0001 |

† structural SP: Moran's I significance rose from ** to *** (more spatial
autocorrelation). GUARD IA SP PAI: minimal −0.07% loss.
→ SAEA (autocorrelation built into training) is unstable (structural collapses on
SP/BA); **GUARD IA gives consistent positive gains** as a post-hoc corrector.

## B2. GUARD IA gains over the single model (mean over 9 cells)
| Metric | mean gain |
|---|---|
| ALS | **+7.5%** |
| MAE | +5.0% |
| RMSE | +1.9% |
| PAI | −0.9% |

(GUARD IA improves everything except hotspot PAI.)

## B3. GUARD IA win counts — vs. best hybrid and vs. single model (out of 9 cells)
| Metric | GUARD IA > **hybrids** | GUARD IA > **single model** |
|---|---|---|
| ALS | **7/9** | **9/9** |
| MAE | **0/9** | 9/9 |
| RMSE | 5/9 | 8/9 |
| PAI | 0/9 | 2/9 |

→ vs single model: GUARD IA wins almost everything (except hotspot PAI).
→ vs hybrids: GUARD IA **wins on ALS (7/9) and RMSE (5/9)** but **loses on MAE
(0/9) and PAI (0/9)** — it is the probabilistic/RMSE specialist (MSE-gated, mean
prediction), while the hybrids (log1p) win the median-style MAE and hotspots.

## B4. Residual spatial dependence — GUARD IA whitens the residuals
Among {base, additive, Anscombe, GUARD IA}, which removes the most residual
cross-sectional dependence (9 cells):

| Diagnostic | GUARD IA wins |
|---|---|
| nearest-neighbour residual correlation (correlogram, corr_h1) | **8/9** |
| Pesaran CD (global cross-sectional dependence) | **7/9** |
| residual correlation matrix (ECM, by \|CD\|) | **7/9** |

The reduction is dramatic in SP (high N, strong dependence): GUARD IA cuts |CD|
≈3–4× (STGCN 369→101, GWaveNet 212→47, STHSL 343→97) while the hybrids barely
move it (369→318). Figures: `figs/correlogram_*` and `figs/ecm_*`
(e.g. `correlogram_SP_CRIME_sthsl_mse`, `ecm_SP_CRIME_sthsl_mse`).

## B5. Conclusions — GUARD IA
1. **GUARD IA is the only method that genuinely attacks correlated residuals** —
   it reduces the residual spatial correlation in 8/9 cells (correlogram) and the
   correlation matrix in 7/9 (ECM); the hybrids leave it essentially intact.
2. **It improves every metric except PAI in almost all cases** (vs single model:
   9/9 ALS, 9/9 MAE, 8/9 RMSE; only PAI 2/9).
3. **vs the hybrids**: it **loses on MAE and PAI** but **wins on ALS (7/9) and
   RMSE (5/9)** — the probabilistic/squared-error specialist.
4. **Limitation / open question**: PAI (hotspot) vs correlated error — making
   errors in clusters may also help predicting the cluster; the link between
   residual spatial structure and hotspot accuracy (PAI) needs investigation.
