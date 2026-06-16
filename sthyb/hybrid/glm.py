"""glm.py — per-cell Poisson calibration GLM (the GUARD IA engine).

For each cell i, fit by MLE (statsmodels IRLS) the calibration model

    log E[y_it] = beta0 + alpha * log(yhat_it) + beta1 * eps_{i,t-1} + beta2 * (W eps)_{i,t-1}

where yhat is the backbone prediction (entered as a FREE covariate, not a fixed
offset — alpha != 1 absorbs systematic over/under-prediction bias) and
eps = y - yhat is the backbone residual. Both residual regressors are
time-lagged (no contemporaneous neighbour values).

Anti-leak: estimation uses TRAINING data only; the validation/test design
matrices are merely returned, for the gating and dose sweep in
`sthyb.hybrid.guardia`.

`fit_one_node` returns (nid, nd) where nd holds:
  beta_calib    (4,) MLE params, or None if the GLM failed / params invalid
  X_va_calib, X_te_calib   design matrices for val/test re-prediction
  off_va        log(max(yhat_va, EPS)) — base val prediction in log space
  pred_te_base  uncorrected backbone test prediction (per-cell fallback)
"""
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

EPS = 1e-3   # floor for log(yhat) — keeps the log link finite on zero predictions


def _valid(b):
    return b is not None and not np.any(np.isnan(b)) and not np.any(np.isinf(b))


def _make_X_calib(log_p, eps_own, sp_lag):
    """Calibration design matrix: [1, log(yhat), eps_{t-1}, (W eps)_{t-1}]."""
    return np.column_stack([np.ones(len(log_p)), log_p, eps_own, sp_lag])


def fit_one_node(nid,
                 y_tr, y_va, y_te,
                 p_tr, p_va, p_te,
                 e_own_tr, e_own_va, e_own_te,
                 e_sp_tr,  e_sp_va,  e_sp_te):
    """Fit the calibration GLM for one cell (TRAINING data only)."""
    base_te = np.maximum(p_te, 0.0).astype(np.float32)
    off_tr = np.log(np.maximum(p_tr, EPS))
    off_va = np.log(np.maximum(p_va, EPS))
    off_te = np.log(np.maximum(p_te, EPS))
    y_f64 = y_tr.astype(np.float64)

    nd = dict(pred_te_base=base_te, off_va=off_va,
              beta_calib=None, X_va_calib=None, X_te_calib=None)
    try:
        Xc_tr = _make_X_calib(off_tr, e_own_tr, e_sp_tr).astype(np.float64)
        Xc_va = _make_X_calib(off_va, e_own_va, e_sp_va).astype(np.float64)
        Xc_te = _make_X_calib(off_te, e_own_te, e_sp_te).astype(np.float64)
        # design matrices are stored even if the fit fails (population-mean fallback)
        nd.update(X_va_calib=Xc_va, X_te_calib=Xc_te)
        m = GLM(y_f64, Xc_tr, family=families.Poisson()).fit(disp=False, maxiter=200)
        if _valid(m.params):
            nd['beta_calib'] = m.params.copy()
    except Exception:
        pass
    return nid, nd
