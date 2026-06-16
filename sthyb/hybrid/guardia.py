"""guardia.py — GUARD IA: per-cell Poisson-GLM calibration with gated spatial-lag dose."""
from joblib import Parallel, delayed
import numpy as np
from sthyb.config import EB_MIN, GUARDIA_GATE, GUARDIA_NJOBS
from sthyb.eval.metrics import lisa_abs_per_node, mean_lisa_abs
from sthyb.hybrid.glm import fit_one_node


def guardia_predict(y_tr, p_tr, y_va, p_va, y_te, p_te, Wr, N,
                    gate_loss=GUARDIA_GATE, njobs=GUARDIA_NJOBS):
    """GUARD IA: per-cell Poisson-GLM calibration with a gated spatial-lag dose.

    Pipeline (the heavy per-cell GLM fit runs ONCE, shared by all three modes):
      1. Fit the calibration GLM per cell (see sthyb.hybrid.glm) on TRAINING only.
      2. Sweep a dose c in [0, 2] that scales the spatial-lag coefficient beta2;
         on VALIDATION record, per candidate c, the per-node gated loss curve
         L[c, i] and the per-node |LISA| curve A[c, i].
      3. Select c three ways:
         'global'     — one best_c for all cells: Pareto knee on the aggregate
                        (loss, mean |LISA|) frontier over the gated grid;
         'pernode'    — c*_i = argmin_c of the per-node gated val loss;
         'lisapareto' — c*_i = per-node Pareto knee on (L[:, i], A[:, i])
                        (loss vs local spatial autocorrelation, per cell).
      4. Per-node gate (all modes): the calibrated prediction replaces the
         backbone for cell i only if it beats the backbone's val loss; s_i = 1.
    gate_loss in {'mae','mse'} sets both the gate criterion and the Pareto loss.

    Returns {'global'|'pernode'|'lisapareto': (mu_te, mu_va, s_i),
             'best_c': float, 'cstar_loss': (N,), 'cstar_lisa': (N,)}.
    Anti-leak: residual lags feeding val/test come only from the observed past.
    """
    eps_tr, eps_va, eps_te = y_tr - p_tr, y_va - p_va, y_te - p_te
    sp_tr, sp_va, sp_te = eps_tr @ Wr.T, eps_va @ Wr.T, eps_te @ Wr.T
    z = np.zeros((1, N))
    eo_tr = np.vstack([z, eps_tr[:-1]]); eo_va = np.vstack([eps_tr[-1:], eps_va[:-1]]); eo_te = np.vstack([eps_va[-1:], eps_te[:-1]])
    es_tr = np.vstack([z, sp_tr[:-1]]);  es_va = np.vstack([sp_tr[-1:], sp_va[:-1]]);  es_te = np.vstack([sp_va[-1:], sp_te[:-1]])
    tasks = [(i, y_tr[:, i], y_va[:, i], y_te[:, i], p_tr[:, i], p_va[:, i], p_te[:, i],
              eo_tr[:, i], eo_va[:, i], eo_te[:, i], es_tr[:, i], es_va[:, i], es_te[:, i])
             for i in range(N)]
    raw = (Parallel(n_jobs=njobs, prefer='processes')(delayed(fit_one_node)(*t) for t in tasks)
           if njobs > 1 else [fit_one_node(*t) for t in tasks])
    nd = {nid: d for nid, d in raw}

    _loss1 = ((lambda a, b: float(np.mean((a - b) ** 2))) if gate_loss == 'mse'
              else (lambda a, b: float(np.mean(np.abs(a - b)))))            # per-node scalar
    _lossN = ((lambda A2, B2: ((A2 - B2) ** 2).mean(0)) if gate_loss == 'mse'
              else (lambda A2, B2: np.abs(A2 - B2).mean(0)))                # (N,) per-node
    conv = [i for i in range(N) if nd[i].get('beta_calib') is not None]
    if len(conv) < EB_MIN:
        base = (p_te.copy(), p_va.copy(), np.zeros(N))
        return {'global': base, 'pernode': base, 'lisapareto': base,
                'best_c': 0.0, 'cstar_loss': np.zeros(N), 'cstar_lisa': np.zeros(N)}
    mu_beta = np.mean([nd[i]['beta_calib'] for i in conv], 0); cset = set(conv)
    beta_raw = {i: (nd[i]['beta_calib'].copy() if i in cset else mu_beta.copy()) for i in range(N)}
    base_va = np.column_stack([np.exp(nd[i]['off_va']) for i in range(N)])  # (T_va, N)
    base_ref = _lossN(y_va, base_va)                                        # (N,) base val loss
    Xva_ok = np.array([nd[i].get('X_va_calib') is not None for i in range(N)])
    scales = np.round(np.arange(0.0, 2.05, 0.10), 2); K = len(scales)

    def vpred(i, c):
        b = beta_raw[i].copy(); b[3] *= c
        return np.maximum(np.exp(np.clip(nd[i]['X_va_calib'] @ b, -30, 30)), 0.0)
    def tpred(i, c):
        b = beta_raw[i].copy(); b[3] *= c
        return np.maximum(np.exp(np.clip(nd[i]['X_te_calib'] @ b, -30, 30)), 0.0)

    # ── sweep ONCE: per-node GATED loss curve L and per-node |LISA| curve A ──
    L = np.zeros((K, N)); A = np.zeros((K, N))
    for k, c in enumerate(scales):
        pv = base_va.copy()
        for i in range(N):
            if not Xva_ok[i]: continue
            v = vpred(i, c)
            if _loss1(y_va[:, i], v) < base_ref[i]: pv[:, i] = v
        L[k] = _lossN(y_va, pv)
        A[k] = lisa_abs_per_node(y_va, pv, Wr)
        if k == 0:
            assert np.isclose(A[0].mean(), mean_lisa_abs(y_va, pv, Wr), rtol=1e-5), \
                "lisa_abs_per_node inconsistent with mean_lisa_abs"

    # ── GLOBAL best_c: Pareto knee on aggregate (loss, |LISA|) derived from L,A ──
    losses = np.sqrt(L.mean(1)) if gate_loss == 'mse' else L.mean(1)
    mis = A.mean(1)
    par = np.ones(K, bool)
    for a in range(K):
        for b in range(K):
            if a != b and losses[b] <= losses[a] and mis[b] <= mis[a]: par[a] = False; break
    pidx = np.where(par)[0]
    ln = (losses[pidx] - losses.min()) / max(losses.max() - losses.min(), 1e-12)
    mn = (mis[pidx] - mis.min()) / max(mis.max() - mis.min(), 1e-12)
    best_c = float(scales[pidx[np.argmin(np.sqrt(ln ** 2 + mn ** 2))]])

    # ── PER-NODE (loss-only): c*_i = argmin_c L[:,i] (GATED curve) ──
    cstar_loss = scales[np.argmin(L, axis=0)]

    # ── PER-NODE Pareto (dissertation Eq.): knee on (L[:,i], A[:,i]) ──
    cstar_lisa = np.zeros(N)
    for i in range(N):
        li, ai = L[:, i], A[:, i]
        lr, ar = float(li.max() - li.min()), float(ai.max() - ai.min())
        if (not Xva_ok[i]) or (lr < 1e-12 and ar < 1e-12):
            cstar_lisa[i] = best_c; continue
        Pi = []
        for a in range(K):
            dom = False
            for b in range(K):
                if a != b and li[b] <= li[a] and ai[b] <= ai[a] and (li[b] < li[a] or ai[b] < ai[a]):
                    dom = True; break
            if not dom: Pi.append(a)
        if not Pi: Pi = [int(np.argmin(li))]
        lni = (li[Pi] - li.min()) / max(lr, 1e-12)
        ani = (ai[Pi] - ai.min()) / max(ar, 1e-12)
        cstar_lisa[i] = scales[Pi[int(np.argmin(np.sqrt(lni ** 2 + ani ** 2)))]]

    # ── assemble per c-mode (per-node gate kept as safety net, NO cap) ──
    def build(c_of):
        mte = np.zeros((y_te.shape[0], N)); mva = base_va.copy(); si = np.zeros(N)
        for i in range(N):
            ndi = nd[i]
            if (not Xva_ok[i]) or ndi.get('X_te_calib') is None:
                mte[:, i] = ndi['pred_te_base']; continue
            c = c_of(i); v = vpred(i, c)
            if _loss1(y_va[:, i], v) < base_ref[i]:
                mte[:, i] = tpred(i, c); mva[:, i] = v; si[i] = 1.0
            else:
                mte[:, i] = ndi['pred_te_base']
        return mte, mva, si

    return {'global':     build(lambda i: best_c),
            'pernode':    build(lambda i: cstar_loss[i]),
            'lisapareto': build(lambda i: cstar_lisa[i]),
            'best_c': best_c, 'cstar_loss': cstar_loss, 'cstar_lisa': cstar_lisa}
