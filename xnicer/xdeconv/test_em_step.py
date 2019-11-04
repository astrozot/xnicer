import numpy as np
from scipy.special import logsumexp
import warnings
from .em_step import log_likelihoods, py_e_single_step, em_step  # pylint: disable=no-name-in-module
from .em_step import FIX_NONE, FIX_AMP, FIX_CLASS, FIX_MEAN, FIX_COVAR, FIX_ALL # pylint: disable=no-name-in-module
from . import xdeconv, scores

def py_log_likelihoods(deltas, covars, results=None):
    nobjs = deltas.shape[0]
    if results is None:
        results = np.zeros(nobjs)
    for n in range(nobjs):
        C = covars[n,:,:]
        C_1 = np.linalg.inv(C)
        results[n] = -0.5 * np.log(np.linalg.det(C)) - \
            0.5 * np.dot(np.dot(C_1, deltas[n]), deltas[n]) - \
            0.5 * deltas.shape[1] * np.log(2.0*np.pi)


def generate_test_log_likelihoods(n, r, seed=1):
    np.random.seed(seed)
    deltas = np.random.rand(n, r)
    C = np.random.rand(n, r, r)
    covars = np.einsum('...ij,...ik->...jk', C, C)
    results1 = np.zeros(n)
    results2 = np.zeros(n)
    log_likelihoods(deltas, covars, results1)
    py_log_likelihoods(deltas, covars, results2)
    assert np.allclose(results1, results2)


def test_log_likelihoods():
    for n in (1, 5, 20, 100):
        for r in (1, 2, 3, 5, 10):
            for seed in range(1, 6):
                generate_test_log_likelihoods(n, r, seed=seed)


def py_em_step(w, S, alpha, alphaclass, m, V, logweights, logclasses, Rt=None,
               fixpars=None, regularization=0.0):
    """Perform an EM step using a pure Python code
    
    Parameters
    ----------
    Note: all arrays parametersare expected to be provided as Fortran
    contiguous arrays.
    
    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions
        
    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.
    
    alpha: array-like, shape (k)
        Array with the statistical weight of each Gaussian. Updated at the
        exit with the new weights.
    
    alphaclass: array-like, shape (c, k)
        Array with the statistical weight per class of each Gaussian. Updated
        at the exit with the new weights. Runs over the k clusters and the c
        classes.

    m: array-like, shape (d, k)
        Centers of multivariate Gaussians, updated at the exit with the new
        centers.
    
    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians, updated 
        at the exit with the new covariance matrices.
        
    logweights: array-like, shape (n,) 
        Log-weights for each observation, or None
        
    logclasses: array-like, shape (c, n) 
        Log-probabilities that each observation belong to a given cluster.
        
    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into 
        the observed r-dimensional vector. If None, it is assumed that r=d 
        and that no project is performed (equivalently: R is an array if 
        identity matrices).
                
    fixpars: array-like, shape (k)
        Array of bitmasks with the FIX_* combinations. Currently ignored.
    
    regularization: double, default=0
        Regularization parameter (use 0 to prevent the regularization).
    """
    n = w.shape[1]
    d = m.shape[0]
    k = m.shape[1]
    q = np.zeros(1)
    b = np.zeros(d)
    B = np.zeros((d, d))
    qs = np.zeros((n, k), order='F')
    bs = np.zeros((n, k, d), order='F')
    Bs = np.zeros((n, k, d, d), order='F')
    for i in range(n):
        if Rt is None:
            Ri = np.identity(d)
        else:
            Ri = Rt[:, :, i].T
        for j in range(k):
            py_e_single_step(w[:, i], np.ascontiguousarray(Ri), S[:, :, i].T, 
                             m[:, j], V[:, :, j].T, q, b, B)
            qs[i, j] = np.sum(alphaclass[:,j] * np.exp(logclasses[:,i])) * \
                alpha[j] * np.exp(q[0]) 
            bs[i, j, :] = m[:, j] + b
            Bs[i, j, :] = B
    # Normalize qs
    # qs = np.exp(qs)
    qs /= np.sum(qs, axis=1)[:, np.newaxis]
    # M-step
    alphaclass[:, :] = np.sum(qs[:, np.newaxis, :] * 
                              np.exp(logclasses.T[:, :, np.newaxis]), axis=0) / n
    alpha[:] = np.sum(alphaclass, axis=0) 
    alphaclass /= alpha[np.newaxis, :]
    # Now collapse all class coordinates
    qj = np.sum(qs, axis=0)
    m[:, :] = (np.sum(qs[:, :, np.newaxis]*bs, axis=0) / qj[:, np.newaxis]).T
    V[:, :, :] = np.sum(qs[:, :, np.newaxis, np.newaxis] *
                        (np.einsum('...i,...j->...ij', m.T - bs, m.T - bs) + Bs), axis=0).T / \
        qj[np.newaxis, np.newaxis, :]


def generate_test_single_e_step(r, d, seed=1):
    """Generate a random E-step test.
    
    This procedure generates a single random point, and runs an E-step on it
    using both the Cython code and a pure Python one. It then compares the
    results.
    
    Parameters
    ----------
    r: int
        Dimensionality of the observed data.
        
    d: int
        Dimensionality of the original parameters.
        
    seed: int, default=1
        Seed for the random number generator.
    """
    np.random.seed(seed)
    w = np.random.rand(r)
    R = np.random.rand(r, d)
    S = np.random.rand(r, r)
    S = np.dot(S.T, S)
    m = np.random.rand(d)
    V = np.random.rand(d, d)
    V = np.dot(V.T, V)
    q = np.zeros(1)
    b = np.zeros(d)
    B = np.zeros((d, d))
    py_e_single_step(w, R, S, m, V, q, b, B)

    w_Rm = w - np.dot(R, m)
    RV = np.dot(R, V)
    T = np.dot(RV, R.T) + S
    T_1 = np.linalg.inv(T)
    q_ = -0.5 * np.log(np.linalg.det(T)) - 0.5 * \
        np.dot(w_Rm, np.dot(T_1, w_Rm))
    b_ = np.dot(np.dot(RV.T, T_1), w_Rm)
    B_ = V - np.dot(np.dot(RV.T, T_1), RV)

    assert np.allclose(q, q_)
    assert np.allclose(b, b_)
    assert np.allclose(B, B_)

    
def test_single_e_step():
    """Run a series of tests using `generate_test_single_e_step`."""
    for d in range(1, 8):
        for r in range(1, d+1):
            for iter in range(10):
                generate_test_single_e_step(r, d, seed=iter)


def generate_test_single_e_step_proj(d, seed=1):
    """Generate a random E-step test.
    
    This procedure generate a single random point, and runs an E-step on it
    using the Cython codes with and without projection. It then compares the
    results.
    
    Parameters
    ----------
    d: int
        Dimensionality of the data
        
    seed: int, default=1
        Seed for the random number generator.
    """
    np.random.seed(seed)
    w = np.random.rand(d)
    R = np.identity(d)
    S = np.random.rand(d, d)
    S = np.dot(S.T, S)
    m = np.random.rand(d)
    V = np.random.rand(d, d)
    V = np.dot(V.T, V)

    q1 = np.zeros(1)
    b1 = np.zeros(d)
    B1 = np.zeros((d, d))
    py_e_single_step(w, R, S, m, V, q1, b1, B1)

    q2 = np.zeros(1)
    b2 = np.zeros(d)
    B2 = np.zeros((d, d))
    py_e_single_step(w, None, S, m, V, q2, b2, B2)

    assert np.allclose(q1, q2)
    assert np.allclose(b1, b2)
    assert np.allclose(B1, B2)


def test_single_e_step_proj():
    """Make sure that the standard/projection e-steps are compatible.""" 
    for d in range(1, 8):
        for iter in range(10):
            generate_test_single_e_step_proj(d, seed=iter)


def generate_test_single_em_step(d, r, n, k, c, seed=1):
    """Generate a random EM-step test.

    This procedure generate a set of random points, and runs an EM-step on 
    them using both the Cython code and a pure Python one. It then finally
    compare the results.

    Parameters
    ----------
    d: int
        Dimensionality of the original parameters.

    r: int
        Dimensionality of the observed data.

    n: int
        Number of points to generate
        
    k: int
        Number of Gaussian components to have
        
    c: int
        Number of classes to have
        
    seed: int, default=1
        Seed for the random number generator.
    """
    np.random.seed(seed)
    w = np.asfortranarray(np.random.rand(r, n))
    Rt = np.asfortranarray(np.random.rand(d, r, n))
    S = np.random.rand(r, r, n)
    S = np.asfortranarray(np.einsum('ij...,kj...->ik...', S, S))
    m = np.asfortranarray(np.random.rand(d, k))
    V = np.random.rand(d, d, k)
    V = np.asfortranarray(np.einsum('ij...,kj...->ik...', V, V))
    alpha = np.random.rand(k)
    alpha /= np.sum(alpha)
    alphaclass = np.asfortranarray(np.random.rand(c, k))
    alphaclass /= np.sum(alphaclass, axis=0)[np.newaxis, :]
    if c > 1:
        classes = np.asfortranarray(np.random.rand(c, n))
        classes /= np.sum(classes, axis=0)[np.newaxis, :]
        classes = np.log(classes)
    else:
        classes = np.zeros((c, n), dtype=np.float64, order='F') - np.log(c)
    weights = np.zeros(n, dtype=np.float64, order='F')

    alpha1 = alpha.copy('A')
    alphaclass1 = alphaclass.copy('A')
    m1 = m.copy('A')
    V1 = V.copy('A')
    alpha2 = alpha.copy('A')
    alphaclass2 = alphaclass.copy('A')
    m2 = m.copy('A')
    V2 = V.copy('A')
    weights = np.zeros(n, order='F')

    em_step(w, S, alpha1, alphaclass1, m1, V1, weights, classes, Rt=Rt)
    py_em_step(w, S, alpha2, alphaclass2, m2, V2, weights, classes, Rt=Rt)

    assert np.allclose(alpha1, alpha2)
    assert np.allclose(alphaclass1, alphaclass2)
    assert np.allclose(m1, m2)
    assert np.allclose(V1, V2)


def test_single_em_step():
    """Run a series of tests using `generate_test_single_em_step`."""
    for d in range(1, 5):
        for r in range(1, d+1):
            for k in range(1, 5):
                for c in range(1, 3):
                    for n in (5, 10, 20):
                        for iter in range(10):
                            generate_test_single_em_step(d, r, n, k, c, seed=iter)


def generate_test_em_likelihood(d, n, k, c, seed=1):
    """Generate a test to check the likelihood calculation

    This procedure generate a set of random points, and runs an EM-step on 
    them using the Cython code, and check if the likelihood returned is
    identical to the one computed directly.

    Parameters
    ----------
    d: int
        Dimensionality of the data and parameters.

    n: int
        Number of points to generate
        
    k: int
        Number of Gaussian components to have
        
    c: int
        Number of classes to have; currently only works with c=1.
        
    seed: int, default=1
        Seed for the random number generator.
    """
    np.random.seed(seed)
    w = np.asfortranarray(np.random.rand(d, n))
    S = np.random.rand(d, d, n)
    S = np.asfortranarray(np.einsum('ij...,kj...->ik...', S, S))
    m = np.asfortranarray(np.random.rand(d, k))
    V = np.random.rand(d, d, k)
    V = np.asfortranarray(np.einsum('ij...,kj...->ik...', V, V))
    alpha = np.random.rand(k)
    alpha /= np.sum(alpha)
    alphaclass = np.asfortranarray(np.random.rand(c, k))
    alphaclass /= np.sum(alphaclass, axis=0)[np.newaxis, :]
    if c > 1:
        classes = np.asfortranarray(np.random.rand(c, n))
        classes /= np.sum(classes, axis=0)[np.newaxis, :]
        classes = np.log(classes)
    else:
        classes = np.zeros((c, n), dtype=np.float64, order='F') - np.log(c)
    weights = np.zeros(n, dtype=np.float64, order='F')

    results = np.zeros((k, n))
    for c in range(k):
        log_likelihoods(w.T - m[:,c], S.T + V[:,:,c], results[c,:])

    lnlike1 = np.sum(logsumexp(results + np.log(alpha[:, np.newaxis]), axis=0))
    lnlike2 = em_step(w, S, alpha, alphaclass, m, V, weights, classes)

    assert np.allclose([lnlike1 / n], [lnlike2])


def test_em_likelihood():
    """Run a series of tests using `generate_test_single_em_step`."""
    for d in range(1, 5):
        for k in range(1, 5):
            for n in (1, 5, 10, 20):
                for iter in range(10):
                    generate_test_em_likelihood(d, n, k, 1, seed=iter)


def generate_data(xamp, xmean, xcovar, ycovar, npts=1000,
                  xclass=None, use_weight=False, use_projection=False,
                  use_classes=False, fixpars=None, seed=1,
                  confusion=0.01):
    """Generate a set of points for XD tests
    
    Parameters
    ----------
    xamp: array-like, shape (k,)
        Array with the statistical weight of each Gaussian.

    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians.

    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain
        the diagonal value of each component; if a 2D vector is provided, it
        is taken to be the identical covariance matrix of all compoments.

    ycovar: array-like, shape (npts, dy, dy)
        Array of covariance matrices for the data points.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain
        the diagonal value of each point; if a 2D vector is provided, it
        is taken to be the identical covariance matrix of all points.

    npts: int, default=1000
        Number of samples to generate.

    xclass: array-like, shape (k, c)
        Array with the statistical weight of each Gaussian for each class.

    use_weight: string, default=False
        Can be False (do not user weights), 'uniform' (use the same value
        for all weights), 'random' (use random distributed weights).

    use_projection: string, default=False
        Can be False (do not use any projection), 'identity' (use the
        identity matrix), 'random' (use a random matrix as projection),
        'alternating' (alternate between the coordinates; only if y has
        dimension 1).

    use_classes: string, default=False
        Indicate if additional information on the classification of the
        various objects should be given. Can be False (do not provide any
        additional information), 'exact' (associate each object with its
        true cluster), 'approximate' (associate each object with its true
        cluster with a 75% probability), 'random' (randomly assign class
        probabilities), 'uniform' (use uniform probabilities for all class
        associtaions).

    fixpars: array-like, shape (k,) or None
        Array of bitmasks with the FIX_* combinations.

    seed: integer, default=1
        The seed to use for the random number generator
    
    confusion: float, default=0.01
        A single parameter used to initialize the clusters with respect to the
        true parameters.  Confusion 0 indicate that the starting parameters
        are the true ones.
    """
    np.random.seed(seed)
    xamp = np.array(xamp)
    xamp /= np.sum(xamp)
    xmean = np.array(xmean)
    xcovar = np.array(xcovar)
    ycovar = np.array(ycovar)
    kdim = xamp.shape[0]
    if xclass is not None:
        xclass = np.array(xclass)
        cdim = xclass.shape[1]
    else:
        cdim = 1
        xclass = np.ones((kdim, cdim))
    xdim = xmean.shape[1]
    if ycovar.ndim >= 1:
        ydim = ycovar.shape[0]
    else:
        ydim = xdim

    if xcovar.ndim < 3:
        if xcovar.ndim == 0:
            xc = xcovar
            xcovar = np.zeros((kdim, xdim, xdim))
            for i in range(xdim):
                xcovar[:, i, i] = xc
        elif xcovar.ndim == 1:
            xc = xcovar
            xcovar = np.zeros((kdim, xdim, xdim))
            for i in range(xdim):
                xcovar[:, i, i] = xc[i]
        elif xcovar.ndim == 2:
            xcovar = np.tile(xcovar, (kdim, 1, 1))

    if ycovar.ndim < 3:
        if ycovar.ndim == 0:
            cov = ycovar
            ycovar = np.zeros((npts, ydim, ydim))
            for i in range(ydim):
                ycovar[:, i, i] = cov
        elif ycovar.ndim == 1:
            cov = ycovar
            ycovar = np.zeros((npts, ydim, ydim))
            for i in range(ydim):
                ycovar[:, i, i] = cov[i]
        elif ycovar.ndim == 2:
            ycovar = np.tile(ycovar, (npts, 1, 1))

    sqrt_xcovar = np.linalg.cholesky(xcovar)
    c, cc = divmod(np.searchsorted(np.cumsum(xamp * xclass), 
                                   np.random.rand(npts)), cdim)
    xdata = np.random.randn(npts, xdim)
    xdata = np.einsum("...ij,...j->...i", sqrt_xcovar[c, :, :], xdata) + \
        xmean[c, :]
    if use_projection:
        if use_projection == 'identity':
            assert xdim == ydim
            projection = np.tile(np.identity(xdim), (npts, 1, 1))
        elif use_projection == 'random':
            projection = np.random.randn(npts, ydim, xdim)
            if xdim == ydim:
                projection[np.random.rand(npts) < 0.1, :, :] = np.identity(xdim)
        elif use_projection == 'alternating':
            assert ydim == 1
            projection = np.zeros((npts, ydim, xdim))
            n = np.arange(npts)
            for i in range(xdim):
                projection[n % xdim == i, 0, i] = 1
        else:
            raise ValueError("Unknown use_projection value")
        ydata = np.einsum("...ij,...j", projection, xdata)
    else:
        ydata = xdata
        projection = None

    if use_weight:
        if use_weight == 'uniform':
            weight = np.zeros(npts)
        elif use_weight == 'random':
            weight = np.log(np.random.rand(npts))
        else:
            raise ValueError("Unkwnon use_weights value")
    else:
        weight = None

    if use_classes:
        classes = np.zeros((npts, cdim))
        if use_classes == 'exact':
            classes[np.arange(npts), cc] = 1
        elif use_classes == 'approximate':
            success = 0.75
            classes[:, :] = (1.0 - success) / (cdim - 1)
            goods = np.random.rand(npts) < success
            classes[np.arange(npts)[goods], cc[goods]] = success
            classes[np.arange(npts)[~goods], (cc[~goods]+1) % cdim] = success
        elif use_classes == 'random':
            classes = np.random.rand(npts, cdim)
            classes /= np.sum(classes, axis=1)[:, np.newaxis]
        elif use_classes == 'uniform':
            classes[:, :] = 1 / cdim
        else:
            raise ValueError("Unknown use_classes value")
        with np.errstate(divide='ignore'):
            classes = np.log(classes)
    else:
        classes = None

    sqrt_ycovar = np.linalg.cholesky(ycovar)
    yerr = np.random.randn(npts, ydim)
    ydata += np.einsum("...ij,...j->...i", sqrt_ycovar, yerr)

    # Determine the starting parameters. Use the true ones in case the 
    # corresponding parameter has been fixed.
    if fixpars is not None:
        xamp_t = xamp.copy()
        free = (fixpars & FIX_AMP == 0)
        if np.any(free):
            xamp_t[free] += confusion * np.random.rand(np.sum(free))
            xamp_t[free] *= (1.0 - np.sum(xamp_t[~free])) / \
                np.sum(xamp_t[free])
        xclass_t = xclass.copy()
        free = (fixpars & FIX_CLASS == 0)
        if np.any(free):
            xclass_t[free, :] += confusion * np.random.rand(np.sum(free), cdim)
            xclass_t[free, :] /= np.sum(xclass_t[free, :], axis=1)[:, np.newaxis]
        free = (fixpars & FIX_MEAN == 0)
        xmean_t = xmean.copy()
        if np.any(free):
            xmean_t[free, :] += confusion * \
                np.diagonal(xcovar, axis1=1, axis2=2)[free] * np.random.randn(np.sum(free), xdim)
        free = (fixpars & FIX_COVAR == 0)
        xcovar_t = xcovar.copy()
        if np.any(free):
            xcovar_t[free, :, :] *= (1 + confusion * np.random.randn(np.sum(free), xdim, xdim))
    else:
        xamp_t = xamp + confusion * np.random.rand(kdim)
        xamp_t /= np.sum(xamp_t)
        xclass_t = xclass + confusion * np.random.rand(kdim, cdim)
        xclass_t /= np.sum(xclass_t, axis=1)[:, np.newaxis]
        xmean_t = xmean + confusion * \
            np.diagonal(xcovar, axis1=1, axis2=2) * np.random.randn(kdim, xdim)
        xcovar_t = xcovar * (1 + confusion * np.random.randn(*xcovar.shape))

    # Data are now ready, proceed with the real test
    return {
        'ydata': ydata, 'ycovar': ycovar, 'xcovar_orig': xcovar,
        'xamp': xamp_t, 'xmean': xmean_t, 'xcovar': xcovar_t, 
        'xclass': xclass_t, 'projection': projection, 'weight': weight,
        'classes': classes, 'fixpars': fixpars
    }


def generate_test_scores(xamp, xmean, xcovar, ycovar, npts=1000,
                         xclass=None, use_weight=False, use_projection=False,
                         use_classes=False, fixpars=None, seed=1,
                         confusion=0.01, silent=True, **kw):
    """Create a full test for the extreme deconvolution algorithm.
    
    This procedure works by creating an artificial set of random samples 
    following a Gaussian mixture model (GMM), then checking that the extreme
    deconvolution is able to recover the original parameters.
    
    Parameters
    ----------
    xamp: array-like, shape (k,)
        Array with the statistical weight of each Gaussian.
        
    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians.
    
    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain 
        the diagonal value of each component; if a 2D vector is provided, it
        is taken to be the identical covariance matrix of all compoments.
        
    ycovar: array-like, shape (npts, dy, dy)
        Array of covariance matrices for the data points.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain 
        the diagonal value of each point; if a 2D vector is provided, it
        is taken to be the identical covariance matrix of all points.

    npts: int, default=1000
        Number of samples to generate.
       
    xclass: array-like, shape (k, c)
        Array with the statistical weight of each Gaussian for each class.
        
    use_weight: string, default=False
        Can be False (do not user weights), 'uniform' (use the same value
        for all weights), 'random' (use random distributed weights).
        
    use_projection: string, default=False
        Can be False (do not use any projection), 'identity' (use the 
        identity matrix), 'random' (use a random matrix as projection),
        'alternating' (alternate between the coordinates; only if y has
        dimension 1).
    
    use_classes: string, default=False
        Indicate if additional information on the classification of the 
        various objects should be given. Can be False (do not provide any
        additional information), 'exact' (associate each object with its
        true cluster), 'approximate' (associate each object with its true
        cluster with a 75% probability), 'random' (randomly assign class
        probabilities), 'uniform' (use uniform probabilities for all class
        associtaions).
        
    fixpars: array-like, shape (k,) or None
        Array of bitmasks with the FIX_* combinations.
    
    seed: integer, default=1
        The seed to use for the random number generator
        
    confusion: float, default=0.01
        A single parameter used to initialize the clusters with respect to the
        true parameters.  Confusion 0 indicate that the starting parameters
        are the true ones.
        
    silent: bool, default=True
        If True, will not print warning messages.
        
    All extra keyword parameters are directly passed to xdeconv.
    """
    data = generate_data(xamp, xmean, xcovar, ycovar, npts=npts,
                         xclass=xclass, use_weight=use_weight,
                         use_projection=use_projection,
                         use_classes=use_classes, fixpars=fixpars,
                         seed=seed, confusion=confusion)
    # Fix the input arrays
    xamp = np.array(xamp)
    xmean = np.array(xmean)
    xcovar = data.pop('xcovar_orig')
    # Updata the argument of xdeconv
    data.update(kw)
    data.pop('xamp')
    data.pop('weight')
    data.pop('fixpars')
    # Data are now ready, proceed with the real test
    with warnings.catch_warnings():
        if silent:
            warnings.simplefilter('ignore')
        q = scores(**data)  # pylint: disable=unexpected-keyword-arg
        
    # Now the same with Python code
    n_components = data['xmean'].shape[0]
    Y = data['ydata'][:, np.newaxis, :]
    Yerr = data['ycovar']
    projection = data['projection']
    Yerr = Yerr[:, np.newaxis, :, :]
    if projection is None:
        T = Yerr + data['xcovar']
        delta = Y - data['xmean']
    else:
        P = projection[:, np.newaxis, :, :]
        V = data['xcovar'][np.newaxis, :, :, :]
        mu = data['xmean'][np.newaxis, :, :]
        T = Yerr + np.einsum('...ij,...jk,...lk->...il', P, V, P)
        delta = Y - np.einsum('...ik,...k->...i', P, mu)
    tmp = np.empty(Y.shape[0])
    result = np.empty((Y.shape[0], n_components))
    for c in range(n_components):
        log_likelihoods(delta[:, c, :], T[:, c, :, :], tmp)
        result[:, c] = tmp 
        if use_classes:
            result[:, c] += logsumexp(np.log(data['xclass'][np.newaxis,c,:]) \
                            + data['classes'], axis=1)
    assert np.allclose(result, q)
    

def test_scores():
    """Test the score calculations in various configurations"""
    for iter in range(3):
        npts = [100, 300, 1000][iter]
        generate_test_scores([0.5, 0.5], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                             npts=npts, seed=iter, silent=True)
        generate_test_scores([0.25, 0.25, 0.5], [[0.0], [4.0], [-3.0]], 
                             [[[0.8]], [[1.2]], [[1.0]]], 0.5,
                             npts=npts, seed=1, silent=True)
        generate_test_scores([0.25, 0.75], 
                             [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                             npts=npts, seed=iter, silent=True)
        generate_test_scores([0.25, 0.25, 0.5], 
                             [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], 
                             [0.8, 1.2, 1.0], 1,
                             use_projection='random', npts=npts, seed=iter, 
                             silent=True)
        generate_test_scores([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], 
                             [1.0, 0.5], [1.0, 1.0],
                             xclass=[[0.8, 0.2], [0.4, 0.6]],
                             use_projection=False, npts=npts, seed=iter, 
                             silent=True, use_classes='approximate')


def generate_test_pyxc(xamp, xmean, xcovar, ycovar, npts=1000,
                       xclass=None, use_weight=False, use_projection=False, 
                       use_classes=False, fixpars=None, seed=1, 
                       confusion=0.01, silent=True, **kw):
    """Create a full test for the extreme deconvolution algorithm.
    
    This procedure works by creating an artificial set of random samples 
    following a Gaussian mixture model (GMM), then checking that the extreme
    deconvolution is able to recover the original parameters.
    
    Parameters
    ----------
    xamp: array-like, shape (k,)
        Array with the statistical weight of each Gaussian.
        
    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians.
    
    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain 
        the diagonal value of each component; if a 2D vector is provided, it
        is taken to be the identical covariance matrix of all compoments.
        
    ycovar: array-like, shape (npts, dy, dy)
        Array of covariance matrices for the data points.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain 
        the diagonal value of each point; if a 2D vector is provided, it
        is taken to be the identical covariance matrix of all points.

    npts: int, default=1000
        Number of samples to generate.
       
    xclass: array-like, shape (k, c)
        Array with the statistical weight of each Gaussian for each class.
        
    use_weight: string, default=False
        Can be False (do not user weights), 'uniform' (use the same value
        for all weights), 'random' (use random distributed weights).
        
    use_projection: string, default=False
        Can be False (do not use any projection), 'identity' (use the 
        identity matrix), 'random' (use a random matrix as projection),
        'alternating' (alternate between the coordinates; only if y has
        dimension 1).
    
    use_classes: string, default=False
        Indicate if additional information on the classification of the 
        various objects should be given. Can be False (do not provide any
        additional information), 'exact' (associate each object with its
        true cluster), 'approximate' (associate each object with its true
        cluster with a 75% probability), 'random' (randomly assign class
        probabilities), 'uniform' (use uniform probabilities for all class
        associtaions).
        
    fixpars: array-like, shape (k,) or None
        Array of bitmasks with the FIX_* combinations.
    
    seed: integer, default=1
        The seed to use for the random number generator
        
    confusion: float, default=0.01
        A single parameter used to initialize the clusters with respect to the
        true parameters.  Confusion 0 indicate that the starting parameters
        are the true ones.
        
    silent: bool, default=True
        If True, will not print warning messages.
        
    All extra keyword parameters are directly passed to xdeconv.
    """
    data = generate_data(xamp, xmean, xcovar, ycovar, npts=npts,
                         xclass=xclass, use_weight=use_weight, 
                         use_projection=use_projection, 
                         use_classes=use_classes, fixpars=fixpars, 
                         seed=seed, confusion=confusion)
    # Fix the input arrays
    xamp = np.array(xamp)
    xmean = np.array(xmean)
    xcovar = data.pop('xcovar_orig')
    # Updata the argument of xdeconv
    data.update(kw)
    # Data are now ready, proceed with the real test
    with warnings.catch_warnings():
        if silent:
            warnings.simplefilter('ignore')
        xdeconv(**data) # pylint: disable=unexpected-keyword-arg
    # extreme_deconvolution(ydata, ycovar, xamp_t, xmean_t, xcovar_t, 
    # projection=projection, weight=weight)

    # Estimate the expected errors
    eff_npts = npts
    eff_covar = xcovar.copy()
    if use_projection == 'random':
        eff_npts /= 2
        eff_covar += np.mean(data['ycovar'], axis=0)
    elif use_projection == 'alternating':
        eff_npts /= xamp.shape[0]
        eff_covar += np.mean(data['ycovar'], axis=0)
    else:
        eff_covar += np.mean(data['ycovar'], axis=0)
    if use_weight == 'random':
        eff_npts /= 0.75
    # We add a numerical stability term to all error estimates
    eps = 1e-8
    # From a Multinomial distribution...
    amp_err = np.sqrt(xamp*(1-xamp) / eff_npts) + eps
    # From a Multivariate Normal distribution
    eff_var = np.diagonal(eff_covar, axis1=1, axis2=2)
    mean_err = np.sqrt(eff_var / (eff_npts * xamp[:, np.newaxis])) + eps
    # From a Whishart distribution...
    cov_err = np.sqrt((eff_covar**2 + np.einsum('...i,...j->...ij', eff_var, eff_var)) /
                      ((eff_npts - 1) * xamp[:, np.newaxis, np.newaxis])) + eps
    return ((data['xamp'], (data['xamp'] - xamp) / amp_err),
            (data['xmean'], (data['xmean'] - xmean) / mean_err),
            (data['xcovar'], (data['xcovar'] - xcovar) / cov_err))


def test_pyxc_1d():
    """Test using 1D data w/ 2 or 3 clusters"""
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1, 
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"

    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"
        
    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0], [4.0], [-3.0]], [[[0.8]], [[1.2]], [[1.0]]], 0.5,
                                     npts=npts, seed=1, silent=True)
        assert np.all(np.abs(a[1]) < 9), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 9"
        assert np.all(np.abs(b[1]) < 9), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 9"
        assert np.all(np.abs(c[1]) < 9), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 9"


def test_pyxc_1d_fixed():
    """Test using 1D data w/ randomly fixed parameters and 2 or 3 clusters"""
    repeats = 3
    for iter in range(5*repeats):
        npts = [100, 300, 1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(2) < 0.3) * FIX_AMP + 
                   (np.random.rand(2) < 0.3) * FIX_MEAN +
                   (np.random.rand(2) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                     npts=npts, seed=iter, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"a[1][fixed] = {np.max(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]))} != 0" 
        assert np.all(np.abs(b[1]) < 4), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 4"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"b[1][fixed] = {np.max(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]))} != 0" 
        assert np.all(np.abs(c[1]) < 5), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 5"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"c[1][fixed] = {np.max(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]))} != 0" 

    for iter in range(5*repeats):
        npts = [100, 300, 1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(2) < 0.3) * FIX_AMP +
                   (np.random.rand(2) < 0.3) * FIX_MEAN +
                   (np.random.rand(2) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                     npts=npts, seed=iter, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"a[1][fixed] = {np.max(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]))} != 0"
        assert np.all(np.abs(b[1]) < 4), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 4"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"b[1][fixed] = {np.max(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]))} != 0"
        assert np.all(np.abs(c[1]) < 5), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 5"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"c[1][fixed] = {np.max(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]))} != 0"

    for iter in range(3*repeats):
        npts = [1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(3) < 0.3) * FIX_AMP +
                   (np.random.rand(3) < 0.3) * FIX_MEAN +
                   (np.random.rand(3) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0], [4.0], [-3.0]], [[[0.8]], [[1.2]], [[1.0]]], 0.5,
                                     npts=npts, seed=1, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 9), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 9"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"a[1][fixed] = {np.max(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]))} != 0"
        assert np.all(np.abs(b[1]) < 9), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 9"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"b[1][fixed] = {np.max(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]))} != 0"
        assert np.all(np.abs(c[1]) < 9), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 9"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"c[1][fixed] = {np.max(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]))} != 0"
            

def test_pyxc_2d():
    """Test using 2D data w/ 2 or 3 clusters"""
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 4), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 4"
        assert np.all(np.abs(b[1]) < 4), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 4"
        assert np.all(np.abs(c[1]) < 4), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 4"
        
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 4), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 4"
        assert np.all(np.abs(b[1]) < 4), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 4"
        assert np.all(np.abs(c[1]) < 4), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 4"
        
    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], 
                                     [0.8, 1.2, 1.0], [1.0, 1.0],
                                     npts=npts, seed=1, silent=True)
        assert np.all(np.abs(a[1]) < 5), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 5"
        assert np.all(np.abs(b[1]) < 5), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 5"
        assert np.all(np.abs(c[1]) < 5), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 5"


def test_pyxc_2d_projection():
    """Test using 2D data in projection w/ 2 or 3 clusters"""
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     use_projection='random', npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"

    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0],
                                     use_projection='alternating', npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(np.abs(c[1]) < 6), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 6"

    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], [0.8, 1.2, 1.0], 1,
                                     use_projection='identity', npts=npts, seed=1, silent=True)
        assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 5"
        assert np.all(np.abs(b[1]) < 5), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 5"
        assert np.all(np.abs(c[1]) < 4), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 4"


def test_pyxc_2d_classes():
    """Test using 2D data w/ classes and projections"""
    for use_classes in ['exact', 'random', 'uniform', 'approximate']:
        print(use_classes)
        for iter in range(5):
            npts = [100, 300, 1000, 3000, 10000][iter]
            print(npts)
            a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                         xclass=np.identity(2), use_projection=False, npts=npts, seed=iter, silent=True,
                                         use_classes=use_classes)
            assert np.all(np.abs(a[1]) < 4), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 4"
            assert np.all(np.abs(b[1]) < 4), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 4"
            assert np.all(np.abs(c[1]) < 4), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 4"

        for iter in range(5):
            npts = [100, 300, 1000, 3000, 10000][iter]
            print(npts)
            a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                         xclass=[[0.8, 0.2], [0.4, 0.6]],
                                         use_projection=False, npts=npts, seed=iter, silent=True,
                                         use_classes=use_classes)
            assert np.all(
                np.abs(a[1]) < 4), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 4"
            assert np.all(
                np.abs(b[1]) < 4), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 4"
            assert np.all(
                np.abs(c[1]) < 4), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 4"

        for iter in range(5):
            npts = [100, 300, 1000, 3000, 10000][iter]
            print(npts)
            a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                         xclass=np.identity(2),
                                         use_projection='random', npts=npts, seed=iter, silent=True,
                                         use_classes=use_classes)
            assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
            assert np.all(np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
            assert np.all(np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"

        for iter in range(5):
            npts = [100, 300, 1000, 3000, 10000][iter]
            print(npts)
            a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0],
                                         xclass=np.identity(2),
                                         use_projection='alternating', npts=npts, seed=iter, silent=True,
                                         use_classes=use_classes)
            assert np.all(np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
            assert np.all(np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
            assert np.all(np.abs(c[1]) < 6), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 6"

        for iter in range(3):
            npts = [1000, 3000, 10000][iter]
            print(npts)
            a, b, c = generate_test_pyxc([0.25, 0.25, 0.50],
                                         [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], [0.8, 1.2, 1.0], 1,
                                         xclass=np.identity(3),
                                         use_projection='identity', npts=npts, seed=1, silent=True,
                                         use_classes=use_classes)
            assert np.all(
                np.abs(a[1]) < 5), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 5"
            assert np.all(np.abs(b[1]) < 5), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 5"
            assert np.all(np.abs(c[1]) < 6), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 6"
        print("Done")


def test_pyxc_2d_fixed():
    """Test using 2D data w/ randomly fixed parameters and 2 or 3 clusters"""
    repeats = 3
    for iter in range(5*repeats):
        npts = [100, 300, 1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(2) < 0.3) * FIX_AMP + 
                   (np.random.rand(2) < 0.3) * FIX_MEAN +
                   (np.random.rand(2) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     npts=npts, seed=iter, silent=True, fixpars=fixpars)
    
    for iter in range(5*repeats):
        npts = [100, 300, 1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(2) < 0.3) * FIX_AMP + 
                   (np.random.rand(2) < 0.3) * FIX_MEAN +
                   (np.random.rand(2) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     npts=npts, seed=iter, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 4), f"a|[1]| = {np.max(np.abs(a[1])):.2f} > 4"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"a[1][fixed] = {np.max(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]))} != 0"
        assert np.all(np.abs(b[1]) < 4), f"b|[1]| = {np.max(np.abs(b[1])):.2f} > 4"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"b[1][fixed] = {np.max(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]))} != 0"
        assert np.all(np.abs(c[1]) < 4), f"c|[1]| = {np.max(np.abs(c[1])):.2f} > 4"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"c[1][fixed] = {np.max(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]))} != 0"

    for iter in range(3*repeats):
        npts = [1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(3) < 0.3) * FIX_AMP +
                   (np.random.rand(3) < 0.3) * FIX_MEAN +
                   (np.random.rand(3) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]],
                                     [0.8, 1.2, 1.0], [1.0, 1.0],
                                     npts=npts, seed=1, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 5), f"a|[1]| = {np.max(np.abs(a[1])):.2f} > 5"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"a[1][fixed] = {np.max(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]))} != 0"
        assert np.all(np.abs(b[1]) < 5), f"b|[1]| = {np.max(np.abs(b[1])):.2f} > 5"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"b[1][fixed] = {np.max(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]))} != 0"
        assert np.all(np.abs(c[1]) < 5), f"c|[1]| = {np.max(np.abs(c[1])):.2f} > 5"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"c[1][fixed] = {np.max(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]))} != 0"
