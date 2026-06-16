"""metrics.py — spatial-dependence metrics (Moran/LISA/PAI), point errors, GW/DM HAC test."""
from scipy.stats import norm
import numpy as np


def _mi_one(z_raw, Wr, N_, S0, S1, S2):
    """Moran's I + analytical p-value for one time step."""
    z  = z_raw - z_raw.mean()
    ss = float(z @ z)
    if ss < 1e-15:
        return 0.0, np.nan
    I  = float(N_ / S0 * (z @ Wr @ z) / ss)
    EI = -1.0 / (N_ - 1)
    m2 = ss / N_
    m4 = float((z ** 4).sum()) / N_
    b2 = m4 / m2 ** 2
    A  = N_ * ((N_**2 - 3*N_ + 3)*S1 - N_*S2 + 3*S0**2)
    B  = b2  * ((N_**2 - N_)*S1  - 2*N_*S2 + 6*S0**2)
    C  = (N_-1) * (N_-2) * (N_-3) * S0**2
    VI = (A - B) / C - EI**2
    if VI <= 0:
        return I, np.nan
    zscore = (I - EI) / np.sqrt(VI)
    pval   = 2.0 * (1.0 - norm.cdf(abs(zscore)))
    return I, pval

def mean_lisa_abs(actual, pred, Wr):
    """Mean absolute local Moran's I (LISA) across windows.

    LISA_i,t = ε_i,t_c * Σ_j w_ij ε_j,t_c  (local spatial autocorrelation)
    Returns mean over (t, i) of |LISA_i,t| — minimise to remove local clusters.
    """
    resid = actual - pred
    vals  = []
    for t in range(len(resid)):
        z   = resid[t] - resid[t].mean()
        wz  = Wr @ z                          # spatial lag of centred residual
        lisa = z * wz                         # local MI (unnormalised)
        vals.append(float(np.mean(np.abs(lisa))))
    return float(np.mean(vals))

def mean_morans_i_sig(actual, pred, Wr):
    """Returns (mean_I, pct_sig, ttest_p) across all test steps.

    pct_sig  : fraction of steps with per-step p < 0.05 (interpretable: how
               often are residuals spatially autocorrelated?)
    ttest_p  : one-sample t-test on the T observed I-values against mu=0;
               tests whether the *mean* I is significantly different from 0
               across windows, robust to inter-step correlation.
    """
    from scipy.stats import ttest_1samp
    resid = actual - pred
    N_    = resid.shape[1]
    S0    = float(Wr.sum())
    S1    = float(0.5 * ((Wr + Wr.T) ** 2).sum())
    S2    = float(((Wr.sum(1) + Wr.T.sum(1)) ** 2).sum())
    Is, ps = [], []
    for t in range(len(resid)):
        I, p = _mi_one(resid[t], Wr, N_, S0, S1, S2)
        Is.append(I)
        if not np.isnan(p):
            ps.append(p)
    Is_arr  = np.array(Is)
    pct_sig = float(np.mean(np.array(ps) < 0.05)) if ps else np.nan
    # t-test: is mean(I) significantly > 0?
    if len(Is_arr) >= 2 and Is_arr.std() > 1e-15:
        _, ttest_p = ttest_1samp(Is_arr, popmean=0.0, alternative='greater')
    else:
        ttest_p = np.nan
    return float(Is_arr.mean()), pct_sig, ttest_p

def pai_at_k(actual_sum, pred_sum, k=0.10):
    """Predictive Accuracy Index at k: (share of crime captured in the top-k%
    predicted cells) / k. PAI = HitRate/k; PAI = 1 is a uniform-random map."""
    n_sel = max(1, int(len(actual_sum) * k))
    total = actual_sum.sum()
    if total == 0: return 0.0
    top = np.argsort(pred_sum)[::-1][:n_sel]
    return float(actual_sum[top].sum() / total) / k

def lisa_abs_per_node(actual, pred, Wr):
    """Per-node mean |LISA| over time — per-node form of mean_lisa_abs:
       LISA_{i,t} = z_{i,t}·(W z_t)_i,  z_t = resid_t − mean_i(resid_t).
       lisa_abs_per_node(...).mean() == mean_lisa_abs(...)  (asserted in the sweep)."""
    z = actual - pred
    z = z - z.mean(1, keepdims=True)             # centre per time over nodes
    lisa = z * (z @ Wr.T)                        # (W z_t)_i = (z @ Wr.T)[t,i]
    return np.abs(lisa).mean(0)

def mae(a, b): return float(np.mean(np.abs(a - b)))

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def spatial_metrics(y_te, pred, Wr):
    mi, pct, pv = mean_morans_i_sig(y_te, pred, Wr)
    a, p = y_te.sum(0), pred.sum(0)
    d = {'MI': float(mi), 'MI_pct_sig': float(pct), 'MI_pval': float(pv)}
    for kk in (0.01, 0.05, 0.10, 0.25): d[f'PAI{int(kk*100)}'] = float(pai_at_k(a, p, kk))
    return d

def hac_test(d_t):
    """Newey-West HAC t-test of mean(d_t)=0 (DM/GW).  Returns (mean,t,p)."""
    d_t = np.asarray(d_t, float); T = len(d_t); m = d_t.mean(); e = d_t - m
    L = int(np.floor(T ** (1 / 3)))
    S = float((e * e).mean())
    for l in range(1, L + 1):
        S += 2 * (1 - l / (L + 1)) * float((e[l:] * e[:-l]).mean())
    se = np.sqrt(max(S, 0) / T)
    t = m / se if se > 0 else 0.0
    return float(m), float(t), float(2 * norm.sf(abs(t)))
