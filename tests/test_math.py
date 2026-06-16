"""
test_math.py — fast unit tests for the numerical core (no TF / no checkpoints).

Formalizes the ad-hoc checks used during development: transform round-trip and
PMF mass, randomized-PIT uniformity, NB-MLE recovery, LISA per-node consistency,
and the HAC (GW/DM) test. Run:  pytest tests/ -q
"""
import numpy as np
from scipy.stats import nbinom

from sthyb.hybrid.transforms import TRANSFORMS
from sthyb.hybrid.predictive import TransformPredictive, CountPredictive
from sthyb.hybrid.star import nb_alpha_mle
from sthyb.eval.metrics import mean_lisa_abs, lisa_abs_per_node, hac_test

rng = np.random.default_rng(0)


def test_transform_roundtrip():
    """inv(fwd(y)) == y for non-negative counts, in every transform space."""
    y = rng.poisson(2.0, size=(30, 10)).astype(float)
    for tf in TRANSFORMS:
        assert np.allclose(tf.inv(tf.fwd(y)), y, atol=1e-9), tf.tag


def test_transform_pmf_mass():
    """Discretized Gaussian-in-transform PMF integrates to ~1 (cdf at large K)."""
    level = rng.uniform(0.0, 5.0, size=(40, 15))
    sig = np.full_like(level, 0.8)
    for tf in TRANSFORMS:
        pr = TransformPredictive(tf, point=None, mu=tf.fwd(level), sig=sig)
        assert np.all(pr.cdf(4000) >= 0.999), tf.tag


def test_pit_uniform_for_well_specified_poisson():
    mu = rng.uniform(0.5, 3.0, size=(200, 40))
    y = rng.poisson(mu).astype(int)
    pr = CountPredictive('poisson', point=mu, lam=np.maximum(mu, 1e-6))
    np.random.seed(0)
    h, _ = np.histogram(pr.pit(y), bins=10, range=(0, 1))
    assert h.std() / h.mean() < 0.15        # ~uniform


def test_nb_alpha_mle_recovers_dispersion():
    mu = rng.uniform(0.5, 3.0, size=(300, 40))
    a_true = 0.5
    n = 1.0 / a_true
    y = nbinom.rvs(n, n / (n + np.maximum(mu, 1e-6)), random_state=0).astype(int)
    assert 0.3 < nb_alpha_mle(y, mu) < 0.8


def test_lisa_per_node_matches_mean():
    N, T = 40, 50
    W = rng.random((N, N)); np.fill_diagonal(W, 0.0); W /= W.sum(1, keepdims=True)
    a, p = rng.random((T, N)), rng.random((T, N))
    assert np.isclose(lisa_abs_per_node(a, p, W).mean(), mean_lisa_abs(a, p, W), rtol=1e-5)


def test_hac_detects_nonzero_mean():
    d = rng.standard_normal(200) * 0.1 + 0.1
    m, t, pv = hac_test(d)
    assert m > 0 and pv < 0.05
