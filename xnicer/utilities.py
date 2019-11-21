"""Utilities for the XNICER code.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13
"""

import numpy as np

# pylint: disable=invalid-name

def log1mexp(x):
    """Compute log(1 - exp(x)) robustly for any value of x.

    It uses the algorithm from

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    r = np.empty_like(x)
    m = -x < np.log(2)
    w = np.where(m)
    r[w] = np.log(-np.expm1(x[w]))
    w = np.where(~m)
    r[w] = np.log1p(-np.exp(x[w]))
    return r


def cho_solve(L, X):
    """Perform a forward substitution to solve a linear system.

    Parameters
    ----------
    L: array_like
        A lower triangular matrix, or an array of such matrices.
        A typical use is when L is the result of Cholesky decomposition.

    X: array_like
        A vector or an array of vectors

    Returns
    -------
    Y: array_like
        The solution of the linear system :math:`X = L Y`.

    Note
    ----
    The typical use involves setting `L = np.linalg.cholesky(M)` and then
    a call `Y = cho_solve(L, X)`.

    """
    if X.shape[-1] != L.shape[-2]:
        raise ValueError("Shapes X and L not aligned: %d (dim %d) != %d (dim %d)" %
                         (X.shape, L.shape, X.ndim-1, L.ndim-2))
    Y = np.zeros(np.broadcast(X, L[..., 0]).shape)
    n = Y.shape[-1]  # pylint: disable=unsubscriptable-object
    for i in range(n):
        Y[..., i] = (X[..., i] - np.sum(L[..., i, 0:i] *
                                        Y[..., 0:i], axis=-1)) / L[..., i, i]
    return Y


def cho_matrix_solve(L, X):
    """Perform a forward substitution to solve a linear system.

    Parameters
    ----------
    L: array_like
        An array with the Cholesky decomposition of a matrix or of an
        array of matrices

    X: array_like
        A matrix.

    Returns
    -------
    Y: array_like
        The solution of the linear system :math:`X = L Y`.

    Note
    ----
    The typical use involves setting `L = np.linalg.cholesky(M)` and then
    a call `Y = cho_solve(L, X)`.

    """
    if X.shape[-1] != L.shape[-2]:
        raise ValueError("Shapes X and L not aligned: %d (dim %d) != %d (dim %d)" %
                         (X.shape, L.shape, X.ndim-1, L.ndim-2))
    Y = np.zeros(np.broadcast(X[..., 0], L[..., 0]).shape + X.shape[-1:])
    m = Y.shape[-1]  # pylint: disable=unsubscriptable-object
    n = Y.shape[-2]  # pylint: disable=unsubscriptable-object
    for j in range(m):
        for i in range(n):
            Y[..., i, j] = (X[..., i, j] - np.sum(L[..., i, 0:i] *
                                                  Y[..., 0:i, j], axis=-1)) / L[..., i, i]
    return Y
