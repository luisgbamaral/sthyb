"""math_graph.py — graph Laplacian and Chebyshev polynomial bases for STGCN."""
import numpy as np
from scipy.sparse.linalg import eigs


def scaled_laplacian(W):
    """Rescaled normalized graph Laplacian (eigenvalues in [-1, 1]).
    :param W: np.ndarray [N, N], weighted adjacency matrix.
    :return: np.matrix [N, N].
    """
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    """Chebyshev polynomial approximation of the graph convolution kernel.
    :param L: np.matrix [N, N], scaled Laplacian.
    :param Ks: int, Chebyshev order (spatial kernel size).
    :param n: int, number of nodes.
    :return: np.ndarray [N, Ks*N].
    """
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))
    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: spatial kernel size must be > 1, received "{Ks}".')
