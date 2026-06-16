"""
config.py — central experiment configuration for sthyb.

Single source of truth for datasets, backbones, lags, grids and paths.
Add a city / backbone / change a hyperparameter HERE — not in the scripts.
Set the env var SMOKE=1 to restrict to a fast POA_CRIME/stgcn smoke run.
"""
import os
import numpy as np

# ── datasets: (name, n_nodes) ────────────────────────────────────────────────
CITIES = [('SP_CRIME', 1445), ('POA_CRIME', 94), ('BA_LESIONES', 74)]

# ── backbones (MSE-trained → mean-targeting) ─────────────────────────────────
BACKBONES = ['stgcn', 'gwavenet_mse', 'sthsl_mse']

# ── train/val/test split sizes and inference batch ───────────────────────────
N_VAL, N_TEST, BATCH = 110, 110, 50

# ── residual-correction lags (own / spatial) ────────────────────────────────
OWN_LAGS, SP_LAGS = (1, 7, 14), (1, 7)

# ── NB2 dispersion MLE grid ──────────────────────────────────────────────────
ALPHA_GRID = np.logspace(-4, 1, 60)

# ── GUARD IA: EB pool min nodes, parallel jobs, gate/Pareto loss ─────────────
EB_MIN, GUARDIA_NJOBS, GUARDIA_GATE = 20, 4, 'mse'

# ── Anscombe offset ──────────────────────────────────────────────────────────
A_ANS = 0.375

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = './data'
CKPT_DIR  = './checkpoints'
OUT_DIR = './results/probabilistic'

# ── smoke test: isolated POA_CRIME/stgcn for fast golden-regression checks ───
if os.environ.get('SMOKE'):
    CITIES, BACKBONES = [('POA_CRIME', 94)], ['stgcn']
