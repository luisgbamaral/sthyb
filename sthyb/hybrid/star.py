"""star.py — seasonal ST-AR residual model (Shoesmith, 2013), fit by within-FE OLS.

Model on the backbone residual e (in a chosen transform space, see transforms.py):

    e_{i,t} = c_i + sum_{j in OWN_LAGS} phi_j * e_{i,t-j}
                  + sum_{l in SP_LAGS}  psi_l * (W e)_{i,t-l}  + u_{i,t}

c_i is a per-region fixed effect (within demeaning); phi/psi are global and
estimated by pooled OLS on the demeaned panel. Spatial terms are TIME-LAGGED
(never contemporaneous). Default lags {1,7,14}/{1,7} target the weekly cycle.
"""
from scipy.stats import nbinom
import numpy as np
from sthyb.config import ALPHA_GRID, OWN_LAGS, SP_LAGS


def within_ols(eg, sg, own_lags=OWN_LAGS, sp_lags=SP_LAGS):
    """Within-FE OLS of the ST-AR model on residual panel eg (T,N), sg = (W e).

    Returns dict: beta (phi then psi), se, cov (classical), cov_dk
    (Driscoll-Kraay HAC — robust to cross-sectional dependence, used by the
    robust Chow test), c (per-region intercepts), s2_node (per-node residual
    variance — the predictive variance of the hybrid), resid_mean.
    """
    T, N = eg.shape
    m = max(max(own_lags), max(sp_lags))
    Te = T - m
    Y   = eg[m:]
    own = [eg[m - j:T - j] for j in own_lags]
    spl = [sg[m - l:T - l] for l in sp_lags]
    dmc = lambda Aa: Aa - Aa.mean(0)                       # demeaned (Te,N)
    Yc  = dmc(Y)
    cols = [dmc(a) for a in own] + [dmc(s) for s in spl]   # list of (Te,N)
    P = len(cols)
    Xc = np.stack(cols, axis=2)                            # (Te, N, P)
    Xflat, Yflat = Xc.reshape(-1, P), Yc.reshape(-1)
    XtXi = np.linalg.pinv(Xflat.T @ Xflat)
    beta = XtXi @ (Xflat.T @ Yflat)
    resid = (Yc - (Xc @ beta))                             # (Te, N)
    dof = max(Te * N - N - P, 1)
    sigma2 = float((resid ** 2).sum()) / dof
    cov_ols = sigma2 * XtXi
    se = np.sqrt(np.maximum(np.diag(cov_ols), 0.0))
    # Driscoll-Kraay: HAC over cross-sectional score sums h_t = Σ_i x_it u_it (P,)
    H = np.einsum('tnp,tn->tp', Xc, resid)                 # (Te, P)
    L = int(np.floor(Te ** (1 / 3)))
    S = (H.T @ H) / Te
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (H[l:].T @ H[:-l]) / Te
        S += w * (G + G.T)
    cov_dk = XtXi @ (S * Te) @ XtXi
    # recover c_i and predictive variance
    no = len(own_lags)
    c = Y.mean(0).copy()
    for k in range(no):           c -= beta[k]      * own[k].mean(0)
    for k in range(len(sp_lags)): c -= beta[no + k] * spl[k].mean(0)
    s2_node = (resid ** 2).mean(0)
    return dict(beta=beta, se=se, cov=cov_ols, cov_dk=cov_dk, c=c,
                s2_node=s2_node, resid_mean=Y.mean(0))

def shrink_var(s2_node, T_split):
    w = T_split / (T_split + 30.0)
    return np.maximum(w * s2_node + (1 - w) * float(np.mean(s2_node)), 1e-6)

def pooled_acf(eg, lags):
    Eo = eg - eg.mean(0); g0 = float((Eo * Eo).mean()) + 1e-12
    return {h: float((Eo[h:] * Eo[:-h]).mean()) / g0 for h in lags}

def predict_correction(c, beta, e_va, e_te, s_va, s_te, own_lags=OWN_LAGS, sp_lags=SP_LAGS):
    """ε̂ on test from observed val->test lags only.  Returns (T_te, N)."""
    T_va, T_te = e_va.shape[0], e_te.shape[0]
    ce, cs = np.vstack([e_va, e_te]), np.vstack([s_va, s_te])
    no = len(own_lags)
    corr = np.tile(c, (T_te, 1))
    for k, j in enumerate(own_lags): corr += beta[k]      * ce[T_va - j:T_va - j + T_te]
    for k, l in enumerate(sp_lags):  corr += beta[no + k] * cs[T_va - l:T_va - l + T_te]
    return corr

def nb_alpha_mle(y_int, mu, grid=ALPHA_GRID):
    """α* maximizing NB val log-likelihood (grid search)."""
    mu = np.maximum(mu, 1e-6).reshape(-1); yv = y_int.reshape(-1)
    best_a, best_ll = grid[0], -np.inf
    for a in grid:
        n = 1.0 / a; pp = n / (n + mu)
        ll = float(nbinom.logpmf(yv, n, pp).sum())
        if ll > best_ll: best_ll, best_a = ll, a
    return float(best_a)
