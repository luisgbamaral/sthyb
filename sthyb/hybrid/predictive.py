"""predictive.py — discrete predictive distributions over integer counts."""
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson
import numpy as np


class Predictive:
    """Base: subclasses provide cdf(k)=P(Y<=k) and ppf(q); the rest is shared."""
    def __init__(self, point): self.point = point
    def cdf(self, k): raise NotImplementedError
    def ppf(self, q): raise NotImplementedError
    def mass_ok(self, Kbig=4000): return bool(np.all(self.cdf(Kbig) >= 0.999))
    def pmf(self, k):
        k = np.asarray(k)
        lower = np.where(k >= 1, self.cdf(np.maximum(k - 1, 0)), 0.0)
        return np.clip(self.cdf(k) - lower, 0.0, 1.0)
    def logscore(self, y):
        return -np.log(np.maximum(self.pmf(y), 1e-12))
    def pit(self, y):
        """Randomized PIT for a discrete dist: u = F(y-1) + V*P(Y=y)."""
        below = np.where(y >= 1, self.cdf(np.maximum(y - 1, 0)), 0.0)
        py = np.clip(self.cdf(y) - below, 0.0, 1.0)
        return below + np.random.uniform(size=y.shape) * py
    def coverage(self, y, level):
        lo, hi = self.ppf((1 - level) / 2), self.ppf((1 + level) / 2)
        return float(np.mean((y >= lo) & (y <= hi)))

class TransformPredictive(Predictive):
    """Gaussian N(mu, sig^2) in a Transform space, discretized to integer counts."""
    def __init__(self, tf, point, mu, sig):
        super().__init__(point); self.tf, self.mu, self.sig = tf, mu, sig
    def cdf(self, k):
        return norm.cdf((self.tf.fwd(np.asarray(k, dtype=np.float64) + 0.5) - self.mu) / self.sig)
    def ppf(self, q):
        return self.tf.ppf_edge(self.mu, self.sig, norm.ppf(q))

class CountPredictive(Predictive):
    """Native count distribution (Poisson / NB2) — normalized by construction."""
    def __init__(self, kind, point, **prm):
        super().__init__(point); self.kind, self.p = kind, prm
    def cdf(self, k):
        k = np.asarray(k, dtype=np.float64)
        return (poisson.cdf(k, self.p['lam']) if self.kind == 'poisson'
                else nbinom.cdf(k, self.p['n'], self.p['pp']))
    def ppf(self, q):
        return (poisson.ppf(q, self.p['lam']) if self.kind == 'poisson'
                else nbinom.ppf(q, self.p['n'], self.p['pp']))
    def mass_ok(self, Kbig=4000): return True
