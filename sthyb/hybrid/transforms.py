"""transforms.py — variance-stabilizing spaces (level / log1p / Anscombe).

The residual hybrid corrects the backbone in a transform space:
e = fwd(y) - fwd(yhat) is modelled by the ST-AR (star.py) and the corrected
prediction is inv(fwd(yhat) + e_hat). The Gaussian predictive in transform
space is discretized to integer counts via cdf edges at fwd(k + 0.5)
(predictive.py). Adding a new space = registering one Transform below.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sthyb.config import A_ANS


def g(y):
    """Anscombe transform g(y) = 2*sqrt(y + 3/8) — variance-stabilizing for Poisson."""
    return 2.0 * np.sqrt(np.maximum(y, 0.0) + A_ANS)

def ginv(z):
    """Inverse Anscombe (point back-transform), clamped at 0."""
    return np.maximum((z / 2.0) ** 2 - A_ANS, 0.0)


@dataclass
class Transform:
    """A variance-stabilizing space for the residual correction. Drives both the
    point/mean back-transform and the discretized Gaussian predictive. Adding a
    space = register a Transform; nothing else changes."""
    tag: str                  # method-name prefix in the master table
    res: str                  # residual-space key
    fwd: Callable             # level -> transform space (applied to ŷ and to k+0.5)
    inv: Callable             # transform space -> level point (clamped >= 0)
    mean: Callable            # (mu, s2) -> Jensen-correct E[y] in level (clamped >= 0)
    ppf_edge: Callable        # (mu, sig, z) -> smallest integer k with cdf(k) >= Phi(z)

LEVEL = Transform('hybrid-add', 'lvl',
    fwd=lambda y: y,
    inv=lambda z: np.maximum(z, 0.0),
    mean=lambda mu, s2: np.maximum(mu, 0.0),
    ppf_edge=lambda mu, sig, z: np.maximum(np.ceil(mu + sig * z - 0.5), 0.0))

LOG1P = Transform('hybrid-log', 'log',
    fwd=lambda y: np.log1p(y),
    inv=lambda z: np.maximum(np.expm1(z), 0.0),
    mean=lambda mu, s2: np.maximum(np.exp(mu + s2 / 2.0) - 1.0, 0.0),       # lognormal mean
    ppf_edge=lambda mu, sig, z: np.maximum(np.ceil(np.expm1(mu + sig * z) - 0.5), 0.0))

ANSCOMBE = Transform('hybrid', 'ans',
    fwd=g, inv=ginv,
    mean=lambda mu, s2: np.maximum((mu ** 2 + s2) / 4.0 - A_ANS, 0.0),      # exact E[y] under N(mu,s2) in g-space
    # 0.875 = A_ANS + 0.5: inverts the half-integer cdf edge g(k + 0.5)
    ppf_edge=lambda mu, sig, z: np.maximum(np.ceil((np.maximum(mu + sig * z, 0.0) / 2.0) ** 2 - 0.875), 0.0))

TRANSFORMS = (ANSCOMBE, LEVEL, LOG1P)
