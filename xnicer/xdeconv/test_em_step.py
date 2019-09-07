import numpy as np
from scipy.special import logsumexp
from .em_step import log_likelihoods, py_e_single_step, em_step
from .em_step import FIX_NONE, FIX_AMP, FIX_MEAN, FIX_COVAR, FIX_ALL, em_step
from . import xdeconv

def py_log_likelihoods(deltas, covars, results=None):
    nobjs = deltas.shape[0]
    if results is None:
        results = np.zeros(nobjs)
    for n in range(nobjs):
        C = covars[n,:,:]
        C_1 = np.linalg.inv(C)
        results[n] = -0.5 * np.log(np.linalg.det(C)) - 0.5 * np.dot(np.dot(C_1, deltas[n]), deltas[n])
    return logsumexp(results)


def generate_test_log_likelihoods(n, r, seed=1):
    np.random.seed(seed)
    deltas = np.random.rand(n, r)
    C = np.random.rand(n, r, r)
    covars = np.einsum('...ij,...ik->...jk', C, C)
    results1 = np.zeros(n)
    results2 = np.zeros(n)
    result1 = log_likelihoods(deltas, covars, results1)
    result2 = py_log_likelihoods(deltas, covars, results2)
    assert np.allclose(results1, results2)
    assert np.allclose([result1], [result2])


def test_log_likelihoods():
    for n in (1, 5, 20, 100):
        for r in (1, 2, 3, 5, 10):
            for seed in range(1, 6):
                generate_test_log_likelihoods(n, r, seed=seed)
                

def py_em_step(w, S, alpha, m, V, Rt=None, logweights=None, classes=None,
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
        
    m: array-like, shape (d, k)
        Centers of multivariate Gaussians, updated at the exit with the new
        centers.
    
    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians, updated 
        at the exit with the new covariance matrices.
    
    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into 
        the observed r-dimensional vector. If None, it is assumed that r=d 
        and that no project is performed (equivalently: R is an array if 
        identity matrices).
        
    logweights: array-like, shape (n,) 
        Log-weights for each observation, or None
        
    classes: array-like, shape (n, k) 
        Log-probabilities that each observation belong to a given cluster.
        
    fixpars: array-like, shape (k)
        Array of bitmasks with the FIX_AMP, FIX_MEAN, and FIX_AMP combinations.
        Currently ignored.
    
    regularization: double, default=0
        Regularization parameter (use 0 to prevent the regularization).
    """
    r = w.shape[0]
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
            py_e_single_step(w[:, i], np.ascontiguousarray(Ri), S[:, :, i].T, m[:, j], V[:, :, j].T,
                             q, b, B)
            qs[i, j] = np.log(alpha[j]) + q[0]
            bs[i, j, :] = m[:, j] + b
            Bs[i, j, :] = B
    # Normalize qs
    qs = np.exp(qs)
    qs /= np.sum(qs, axis=1)[:, np.newaxis]
    # M-step
    qj = np.sum(qs, axis=0)
    alpha[:] = qj / n
    m[:, :] = (np.sum(qs[:, :, np.newaxis]*bs, axis=0) / qj[:, np.newaxis]).T
    V[:, :, :] = np.sum(qs[:, :, np.newaxis, np.newaxis] *
                        (np.einsum('...i,...j->...ij', m.T - bs, m.T - bs) + Bs), axis=0).T / \
        qj[np.newaxis, np.newaxis, :]


def generate_test_single_e_step(r, d, seed=1):
    """Generate a random E-step test.
    
    This procedure generate a single random point, and runs an E-step on it
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
                
                
def generate_test_single_em_step(d, r, n, k, seed=1):
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

    alpha1 = alpha.copy('A')
    m1 = m.copy('A')
    V1 = V.copy('A')
    alpha2 = alpha.copy('A')
    m2 = m.copy('A')
    V2 = V.copy('A')

    em_step(w, S, alpha1, m1, V1, Rt)
    py_em_step(w, S, alpha2, m2, V2, Rt)

    assert np.allclose(alpha1, alpha2)
    assert np.allclose(m1, m2)
    assert np.allclose(V1, V2)


def test_single_em_step():
    """Run a series of tests using `generate_test_single_em_step`."""
    for d in range(1, 5):
        for r in range(1, d+1):
            for k in range(1, 5):
                for n in (5, 10, 20):
                    for iter in range(10):
                        generate_test_single_em_step(d, r, n, k, seed=iter)
  

def generate_test_pyxc(xamp, xmean, xcovar, ycovar, npts=1000,
                       use_weight=False, use_projection=False, use_classes=False,
                       fixpars=None, seed=1, confusion=0.01, plot=False, **kw):
    """Create a full test for the extreme deconvolution algorithm.
    
    This procedure works by creating an artificial set of random samples 
    following a Gaussian mixture model (GMM), then checking that the extreme
    deconvolution is able to recover the original parameters.
    
    Parameters
    ----------
    xamp: array-like, shape (k)
        Array with the statistical weight of each Gaussian.
        
    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians.
    
    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians.
        If a simple scalar is provided, it is understood as the diagonal term
        of all gaussians; if a 1D vector is provided, it is taken to contain 
        the diagonal value of each component; if a 2D vector is provided, it
        is tken to be the identical covariance matrix of all copmoments.
        
    npts: int, default=1000
        Number of samples to generate.
        
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
        
    fixpars: array
    
    seed: integer, default=1
        The seed to use for the random number generator
        
    confusion: float, default=0.1
        A single parameter used to initialize the clusters with respect to the
        true parameters.  Confusion 0 indicate that the starting parameters
        are the true ones.
        
    All extra keyword parameters are directly passed to xdeconv.
    """

    np.random.seed(seed)
    xamp = np.array(xamp)
    xamp /= np.sum(xamp)
    xmean = np.array(xmean)
    xcovar = np.array(xcovar)
    ycovar = np.array(ycovar)
    kdim = xamp.shape[0]
    xdim = xmean.shape[1]
    if ycovar.ndim > 0:
        ydim = ycovar.shape[0]
    else:
        ydim = xdim

    if xcovar.ndim < 3:
        if xcovar.ndim == 0:
            c = xcovar
            xcovar = np.zeros((kdim, xdim, xdim))
            for i in range(xdim):
                xcovar[:, i, i] = c
        elif xcovar.ndim == 1:
            c = xcovar
            xcovar = np.zeros((kdim, xdim, xdim))
            for i in range(xdim):
                xcovar[:, i, i] = c[i]
        elif xcovar.ndim == 2:
            xcovar = np.tile(xcovar, (kdim, 1, 1))

    if ycovar.ndim < 3:
        if ycovar.ndim == 0:
            c = ycovar
            ycovar = np.zeros((npts, ydim, ydim))
            for i in range(ydim):
                ycovar[:, i, i] = c
        elif ycovar.ndim == 1:
            c = ycovar
            ycovar = np.zeros((npts, ydim, ydim))
            for i in range(ydim):
                ycovar[:, i, i] = c[i]
        elif ycovar.ndim == 2:
            ycovar = np.tile(ycovar, (npts, 1, 1))

    sqrt_xcovar = np.linalg.cholesky(xcovar)
    c = np.searchsorted(np.cumsum(xamp), np.random.rand(npts))
    xdata = np.random.randn(npts, xdim)
    xdata = np.einsum("...ij,...j->...i", sqrt_xcovar[c, :, :], xdata) + \
        xmean[c, :]
    if use_projection:
        if use_projection == 'identity':
            assert xdim == ydim
            projection = np.tile(np.identity(xdim), (npts, 1, 1))
        elif use_projection == 'random':
            projection = np.random.randn(npts, ydim, xdim)
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
        classes = np.zeros((npts, kdim))
        if use_classes == 'exact':
            classes[np.arange(npts), c] = 1
        elif use_classes == 'approximate':
            classes[:, :] = 0.25 / (kdim - 1)
            classes[np.arange(npts), c] = 0.75
        elif use_classes == 'random':
            classes = np.random.rand(npts, kdim)
            classes /= np.sum(classes, index=1)
        elif use_classes == 'uniform':
            classes[:, :] = 1 / kdim
        else:
            raise ValueError("Unkwnon use_classes value")
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
            xamp_t[free] *= (1.0 - np.sum(xamp_t[~free])) / np.sum(xamp_t[free])
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
        xmean_t = xmean + confusion * \
            np.diagonal(xcovar, axis1=1, axis2=2) * np.random.randn(kdim, xdim)
        xcovar_t = xcovar * (1 + confusion * np.random.randn(*xcovar.shape))

    # Data are now ready, proceed with the real test
    xdeconv(ydata, ycovar, xamp_t, xmean_t, xcovar_t, projection=projection, weight=weight,
            classes=classes, fixpars=fixpars, **kw)
    # extreme_deconvolution(ydata, ycovar, xamp_t, xmean_t, xcovar_t, projection=projection, weight=weight)

    # Estimate the expected errors
    eff_npts = npts
    eff_covar = xcovar.copy()
    if use_projection == 'random':
        eff_npts /= 2
    elif use_projection == 'alternating':
        eff_npts /= kdim
    else:
        eff_covar += np.mean(ycovar, axis=0)
    if use_weight == 'random':
        eff_npts /= 0.75
    amp_err = np.sqrt(xamp*(1-xamp) / eff_npts)
    eff_var = np.diagonal(eff_covar, axis1=1, axis2=2)
    mean_err = np.sqrt(eff_var / (eff_npts * xamp[:, np.newaxis]))
    # From a Whishart distribution...
    cov_err = np.sqrt((eff_covar**2 + np.einsum('...i,...j->...ij', eff_var, eff_var)) /
                      ((eff_npts - 1) * xamp[:, np.newaxis, np.newaxis]))
    if plot:
        data = []
        for k in range(kdim):
            data.append(ydata[c == k, 0])
        plt.hist(data, bins=100, stacked=True)
    return ((xamp_t, (xamp_t - xamp) / amp_err),
            (xmean_t, (xmean_t - xmean) / mean_err),
            (xcovar_t, (xcovar_t - xcovar) / cov_err))


def test_pyxc_1d():
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1, 
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 3), "Error: a[1] = {a[1]} > 3"
        assert np.all(np.abs(b[1]) < 3), "Error: b[1] = {b[1]} > 3"
        assert np.all(np.abs(c[1]) < 3), "Error: c[1] = {c[1]} > 3"

    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 3), "Error: a[1] = {a[1]} > 3"
        assert np.all(np.abs(b[1]) < 3), "Error: b[1] = {b[1]} > 3"
        assert np.all(np.abs(c[1]) < 3), "Error: c[1] = {c[1]} > 3"
        
    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0], [4.0], [-3.0]], [[[0.8]], [[1.2]], [[1.0]]], 0.5,
                                     npts=npts, seed=1, silent=True)
        assert np.all(np.abs(a[1]) < 9), "Error: a[1] = {a[1]} > 9"
        assert np.all(np.abs(b[1]) < 9), "Error: b[1] = {b[1]} > 9"
        assert np.all(np.abs(c[1]) < 9), "Error: c[1] = {c[1]} > 9"


def test_pyxc_1d_fixed():
    repeats = 3
    for iter in range(5*repeats):
        npts = [100, 300, 1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(2) < 0.3) * FIX_AMP + 
                   (np.random.rand(2) < 0.3) * FIX_MEAN +
                   (np.random.rand(2) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                     npts=npts, seed=iter, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 3), f"Error: a[1] = {a[1]} > 3"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"Error: a[1][fixed] = {a[1][fixpars & FIX_AMP == FIX_AMP]} != 0" 
        assert np.all(np.abs(b[1]) < 4), f"Error: b[1] = {b[1]} > 4"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"Error: b[1][fixed] = {b[1][fixpars & FIX_MEAN == FIX_MEAN]} != 0" 
        assert np.all(np.abs(c[1]) < 5), f"Error: c[1] = {c[1]} > 5"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"Error: c[1][fixed] = {c[1][fixpars & FIX_COVAR == FIX_COVAR]} != 0" 

    for iter in range(5*repeats):
        npts = [100, 300, 1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(2) < 0.3) * FIX_AMP +
                   (np.random.rand(2) < 0.3) * FIX_MEAN +
                   (np.random.rand(2) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                     npts=npts, seed=iter, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 3), f"Error: a[1] = {a[1]} > 3"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"Error: a[1][fixed] = {a[1][fixpars & FIX_AMP == FIX_AMP]} != 0"
        assert np.all(np.abs(b[1]) < 4), f"Error: b[1] = {b[1]} > 4"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"Error: b[1][fixed] = {b[1][fixpars & FIX_MEAN == FIX_MEAN]} != 0"
        assert np.all(np.abs(c[1]) < 5), f"Error: c[1] = {c[1]} > 5"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"Error: c[1][fixed] = {c[1][fixpars & FIX_COVAR == FIX_COVAR]} != 0"

    for iter in range(3*repeats):
        npts = [1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(3) < 0.3) * FIX_AMP +
                   (np.random.rand(3) < 0.3) * FIX_MEAN +
                   (np.random.rand(3) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0], [4.0], [-3.0]], [[[0.8]], [[1.2]], [[1.0]]], 0.5,
                                     npts=npts, seed=1, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 9), f"Error: a[1] = {a[1]} > 9"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"Error: a[1][fixed] = {a[1][fixpars & FIX_AMP == FIX_AMP]} != 0"
        assert np.all(np.abs(b[1]) < 9), f"Error: b[1] = {b[1]} > 9"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"Error: b[1][fixed] = {b[1][fixpars & FIX_MEAN == FIX_MEAN]} != 0"
        assert np.all(np.abs(c[1]) < 9), f"Error: c[1] = {c[1]} > 9"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"Error: c[1][fixed] = {c[1][fixpars & FIX_COVAR == FIX_COVAR]} != 0"
            

def test_pyxc_2d():
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 4), f"Error: a[1] = {a[1]} > 4"
        assert np.all(np.abs(b[1]) < 4), f"Error: b[1] = {b[1]} > 4"
        assert np.all(np.abs(c[1]) < 4), f"Error: c[1] = {c[1]} > 4"
        
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 4), f"Error: a[1] = {a[1]} > 4"
        assert np.all(np.abs(b[1]) < 4), f"Error: b[1] = {b[1]} > 4"
        assert np.all(np.abs(c[1]) < 4), f"Error: c[1] = {c[1]} > 4"
        
    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], 
                                     [0.8, 1.2, 1.0], [1.0, 1.0],
                                     npts=npts, seed=1, silent=True)
        assert np.all(np.abs(a[1]) < 5), f"Error: a[1] = {a[1]} > 5"
        assert np.all(np.abs(b[1]) < 5), f"Error: b[1] = {b[1]} > 5"
        assert np.all(np.abs(c[1]) < 5), f"Error: c[1] = {c[1]} > 5"


def test_pyxc_2d_projection():
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                     use_projection='random', npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 4), f"Error: a[1] = {a[1]} > 4"
        assert np.all(np.abs(b[1]) < 4), f"Error: b[1] = {b[1]} > 4"
        assert np.all(np.abs(c[1]) < 4), f"Error: c[1] = {c[1]} > 4"

    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0],
                                     use_projection='alternating', npts=npts, seed=iter, silent=True)
        assert np.all(np.abs(a[1]) < 6), f"Error: a[1] = {a[1]} > 6"
        assert np.all(np.abs(b[1]) < 6), f"Error: b[1] = {b[1]} > 6"
        assert np.all(np.abs(c[1]) < 12), f"Error: c[1] = {c[1]} > 12"

    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], [0.8, 1.2, 1.0], 1,
                                     use_projection='identity', npts=npts, seed=1, silent=True)
        assert np.all(np.abs(a[1]) < 5), f"Error: a[1] = {a[1]} > 5"
        assert np.all(np.abs(b[1]) < 5), f"Error: b[1] = {b[1]} > 5"
        assert np.all(np.abs(c[1]) < 5), f"Error: c[1] = {c[1]} > 5"


def test_pyxc_2d_fixed():
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
        assert np.all(np.abs(a[1]) < 4), f"Error: a[1] = {a[1]} > 4"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"Error: a[1][fixed] = {a[1][fixpars & FIX_AMP == FIX_AMP]} != 0"
        assert np.all(np.abs(b[1]) < 4), f"Error: b[1] = {b[1]} > 4"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"Error: b[1][fixed] = {b[1][fixpars & FIX_MEAN == FIX_MEAN]} != 0"
        assert np.all(np.abs(c[1]) < 4), f"Error: c[1] = {c[1]} > 4"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"Error: c[1][fixed] = {c[1][fixpars & FIX_COVAR == FIX_COVAR]} != 0"

    for iter in range(3*repeats):
        npts = [1000, 3000, 10000][iter // repeats]
        np.random.seed(10 + iter)
        fixpars = ((np.random.rand(3) < 0.3) * FIX_AMP +
                   (np.random.rand(3) < 0.3) * FIX_MEAN +
                   (np.random.rand(3) < 0.3) * FIX_COVAR)
        a, b, c = generate_test_pyxc([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]],
                                     [0.8, 1.2, 1.0], [1.0, 1.0],
                                     npts=npts, seed=1, silent=True, fixpars=fixpars)
        assert np.all(np.abs(a[1]) < 5), f"Error: a[1] = {a[1]} > 5"
        assert np.all(np.abs(a[1][fixpars & FIX_AMP == FIX_AMP]) < 1e-5), \
            f"Error: a[1][fixed] = {a[1][fixpars & FIX_AMP == FIX_AMP]} != 0"
        assert np.all(np.abs(b[1]) < 5), f"Error: b[1] = {b[1]} > 5"
        assert np.all(np.abs(b[1][fixpars & FIX_MEAN == FIX_MEAN]) < 1e-5), \
            f"Error: b[1][fixed] = {b[1][fixpars & FIX_MEAN == FIX_MEAN]} != 0"
        assert np.all(np.abs(c[1]) < 5), f"Error: c[1] = {c[1]} > 5"
        assert np.all(np.abs(c[1][fixpars & FIX_COVAR == FIX_COVAR]) < 1e-5), \
            f"Error: c[1][fixed] = {c[1][fixpars & FIX_COVAR == FIX_COVAR]} != 0"
