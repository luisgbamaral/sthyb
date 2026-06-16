"""
spatial_diag.py — residual spatial-correlation diagnostics (test residuals).

For each city x backbone x method m in {base, hybrid-add-pooled, hybrid-pooled,
guardia}, on TEST residuals eps = y_te - pt_m (T x N):
  1) Pesaran CD test of cross-sectional dependence.
  2) Correlogram of pairwise residual correlation by graph hop (1..5, far) with
     a node-label-permutation null band (200 perms).
  3) lambda_max of R vs Marchenko-Pastur edge (1+sqrt(N/T))^2 and # eigs above.
  4) ECM: residual correlation matrix reordered by node order (+ 200x200 top-count
     block for SP), saved as .npy and imshow PNG (base vs methods side by side).

Outputs: results/probabilistic/spatial_diag.csv, figs/ecm_*.png, figs/correlogram_*.png
Run:  python scripts/run_spatial_diag.py   (GUARD IA is the slow part: per-cell GLMs)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.stats import norm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from sthyb.config import CITIES, BACKBONES, OWN_LAGS, SP_LAGS, OUT_DIR, DATA_DIR
from sthyb.eval.probabilistic import load_backbone
from sthyb.hybrid.star import within_ols, predict_correction
from sthyb.hybrid.guardia import guardia_predict
from sthyb.hybrid.transforms import g, ginv
from sthyb.eval.plotting import set_style

set_style()
FIG_DIR = pjoin(OUT_DIR, 'figs')
METHODS = ['base', 'hybrid-add-pooled', 'hybrid-pooled', 'guardia']
N_PERM = 200
np.random.seed(0)


# ── 1) Pesaran CD ────────────────────────────────────────────────────────────
def pesaran_cd(eps):
    """eps (T,N). Drops near-constant columns. Returns (CD, p, R, keep_mask)."""
    keep = eps.var(0) > 1e-12
    e = eps[:, keep]
    T, N = e.shape
    R = np.corrcoef(e, rowvar=False)
    iu = np.triu_indices(N, 1)
    s = float(np.nansum(R[iu]))
    CD = np.sqrt(2.0 * T / (N * (N - 1))) * s
    p = 2.0 * (1.0 - norm.cdf(abs(CD)))
    return float(CD), float(p), R, keep


# ── 2) graph hops + correlogram with null band ───────────────────────────────
def graph_dist(W):
    A = csr_matrix((W > 0).astype(float))
    return shortest_path(A, method='D', unweighted=True, directed=False)   # inf if disconnected


def correlogram(R, Dfull, keep, n_perm=N_PERM):
    Dk = Dfull[np.ix_(keep, keep)]
    Nk = R.shape[0]
    ii, jj = np.triu_indices(Nk, 1)
    rvals = R[ii, jj]
    dvals = Dk[ii, jj]
    res = {}
    for h in range(1, 6):
        mk = dvals == h
        res[f'corr_h{h}'] = float(np.nanmean(rvals[mk])) if mk.any() else np.nan
        res[f'se_h{h}'] = float(np.nanstd(rvals[mk]) / np.sqrt(max(mk.sum(), 1))) if mk.any() else np.nan
    res['corr_far'] = float(np.nanmean(rvals[dvals > 5])) if (dvals > 5).any() else np.nan
    # null band: permute node labels of D (R fixed)
    null = {h: [] for h in range(1, 6)}
    for _ in range(n_perm):
        perm = np.random.permutation(Nk)
        dp = Dk[perm[ii], perm[jj]]
        for h in range(1, 6):
            mk = dp == h
            null[h].append(np.nanmean(rvals[mk]) if mk.any() else np.nan)
    for h in range(1, 6):
        arr = np.asarray(null[h], float)
        res[f'null_lo_h{h}'] = float(np.nanpercentile(arr, 2.5))
        res[f'null_hi_h{h}'] = float(np.nanpercentile(arr, 97.5))
    return res


# ── 3) eigenvalues vs Marchenko-Pastur ───────────────────────────────────────
def mp_eigen(R, T):
    N = R.shape[0]
    lam = np.linalg.eigvalsh(R)
    edge = (1.0 + np.sqrt(N / T)) ** 2
    return float(lam[-1]), float(edge), int((lam > edge).sum())


# ── recompute the four method test predictions ───────────────────────────────
def _pooled_corr(y_tr, p_tr, y_va, p_va, y_te, p_te, Wr, tf):
    """within-OLS on pooled (train+val) residuals in transform tf; project to test."""
    e_tr, e_va, e_te = tf(y_tr) - tf(p_tr), tf(y_va) - tf(p_va), tf(y_te) - tf(p_te)
    s_tr, s_va, s_te = e_tr @ Wr.T, e_va @ Wr.T, e_te @ Wr.T
    r = within_ols(np.vstack([e_tr, e_va]), np.vstack([s_tr, s_va]))
    return predict_correction(r['c'], r['beta'], e_va, e_te, s_va, s_te)


def method_preds(ds, N, bk, Wr):
    data = load_backbone(ds, N, bk)
    if data is None:
        return None
    (y_tr, p_tr), (y_va, p_va), (y_te, p_te) = data['train'], data['val'], data['test']
    preds = {'base': p_te}
    preds['hybrid-pooled'] = ginv(g(p_te) + _pooled_corr(y_tr, p_tr, y_va, p_va, y_te, p_te, Wr, g))
    preds['hybrid-add-pooled'] = np.maximum(
        p_te + _pooled_corr(y_tr, p_tr, y_va, p_va, y_te, p_te, Wr, lambda x: x), 0.0)
    preds['guardia'] = np.maximum(
        guardia_predict(y_tr, p_tr, y_va, p_va, y_te, p_te, Wr, N)['global'][0], 0.0)
    return y_te, preds


# ── 4) ECM figures ───────────────────────────────────────────────────────────
def _full_R(eps):
    keep = eps.var(0) > 1e-12
    R = np.full((eps.shape[1], eps.shape[1]), np.nan)
    sub = np.corrcoef(eps[:, keep], rowvar=False)
    idx = np.where(keep)[0]
    R[np.ix_(idx, idx)] = sub
    return R


def ecm_figure(ds, bk, y_te, preds):
    Rs = {m: _full_R(y_te - preds[m]) for m in METHODS}
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4))
    for ax, m in zip(axes, METHODS):
        im = ax.imshow(Rs[m], cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='nearest')
        ax.set_title(m, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        np.save(pjoin(FIG_DIR, f'ecm_{ds}_{bk}_{m}.npy'), Rs[m].astype(np.float32))
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(f'{ds} / {bk} — residual correlation matrix (node order)')
    fig.savefig(pjoin(FIG_DIR, f'ecm_{ds}_{bk}.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)
    # SP: 200x200 block of the highest-count nodes
    if y_te.shape[1] > 200:
        top = np.argsort(y_te.sum(0))[::-1][:200]; top.sort()
        fig, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4))
        for ax, m in zip(axes, METHODS):
            ax.imshow(Rs[m][np.ix_(top, top)], cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='nearest')
            ax.set_title(m, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{ds} / {bk} — top-200 count nodes')
        fig.savefig(pjoin(FIG_DIR, f'ecm_{ds}_{bk}_top200.png'), dpi=110, bbox_inches='tight')
        plt.close(fig)


def correlogram_figure(ds, bk, curves, nullband):
    """curves[m] = dict corr_h*/se_h*; nullband = base method's null lo/hi per hop."""
    hops = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(6, 4))
    lo = [nullband[f'null_lo_h{h}'] for h in hops]
    hi = [nullband[f'null_hi_h{h}'] for h in hops]
    ax.fill_between(hops, lo, hi, color='0.8', label='null 95% (perm)')
    for m in METHODS:
        y = [curves[m][f'corr_h{h}'] for h in hops]
        e = [curves[m][f'se_h{h}'] for h in hops]
        ax.errorbar(hops, y, yerr=e, marker='o', ms=4, capsize=2, label=m)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('graph hop'); ax.set_ylabel('mean residual corr'); ax.set_xticks(hops)
    ax.set_title(f'{ds} / {bk} — residual correlogram'); ax.legend(fontsize=8)
    fig.savefig(pjoin(FIG_DIR, f'correlogram_{ds}_{bk}.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    for ds, N in CITIES:
        W = pd.read_csv(pjoin(DATA_DIR, f'{ds}_W.csv'), header=None).values.astype(np.float64)
        rs = W.sum(1); rs[rs == 0] = 1.0; Wr = W / rs[:, None]
        Dfull = graph_dist(W)
        for bk in BACKBONES:
            print(f"\n=== {ds} / {bk} ===")
            mp = method_preds(ds, N, bk, Wr)
            if mp is None:
                continue
            y_te, preds = mp
            curves = {}
            for m in METHODS:
                eps = y_te - preds[m]
                CD, CD_p, R, keep = pesaran_cd(eps)
                cg = correlogram(R, Dfull, keep)
                lam_max, mp_edge, n_above = mp_eigen(R, eps.shape[0])
                row = dict(city=ds, backbone=bk, method=m, CD=CD, CD_p=CD_p,
                           lam_max=lam_max, mp_edge=mp_edge, n_above_mp=n_above, **cg)
                rows.append(row); curves[m] = cg
                print(f"  {m:18s} CD={CD:8.1f} p={CD_p:.3f} | corr_h1={cg['corr_h1']:.4f} "
                      f"far={cg['corr_far']:.4f} | lam_max={lam_max:.1f} vs MP {mp_edge:.1f} "
                      f"(#>MP={n_above})")
            correlogram_figure(ds, bk, curves, curves['base'])
            ecm_figure(ds, bk, y_te, preds)

    df = pd.DataFrame(rows)
    cols = (['city', 'backbone', 'method', 'CD', 'CD_p']
            + [f'corr_h{h}' for h in range(1, 6)] + ['corr_far']
            + [f'se_h{h}' for h in range(1, 6)]
            + [f'null_lo_h{h}' for h in range(1, 6)] + [f'null_hi_h{h}' for h in range(1, 6)]
            + ['lam_max', 'mp_edge', 'n_above_mp'])
    df[cols].to_csv(pjoin(OUT_DIR, 'spatial_diag.csv'), index=False)
    print(f"\nSaved spatial_diag.csv + figs/ -> {OUT_DIR}")


if __name__ == '__main__':
    main()
