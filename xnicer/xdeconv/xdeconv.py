import numpy as np
from .em_step import FIX_NONE, FIX_AMP, FIX_MEAN, FIX_COVAR, FIX_ALL, em_step
import warnings


def check_numpy_array(name, arr, shapes):
    """Verify that an array has the correct shape.
    
    Parameters
    ----------
    name: string
        Name of the array. Used for error messages.
        
    arr: array-like
        The array to check.
        
    shapes: list
        Must be a list of touples, with each touple indicating an allowed
        shape. If an element of the touple is negative or null, it indicates
        that the corresponding dimension can have any size; otherwise, a 
        strict check on the dimension is performed.
        
    Returns
    -------
    arr.shape: touple
        The shape of the array, if it has passed all checks.
    """
    def shape_error():
        if len(shapes) == 1:
            s = f"{len(shapes[0])}D"
        elif len(shapes) == 2:
            s = f"{len(shapes[0])}D or {len(shapes[1])}D"
        else:
            s = ", ".join([str(len(shape)) + "D" for shape in shapes[:-1]]) + \
                f", or {len(shapes[-1])}D"
        return s
    if isinstance(shapes, tuple):
        shapes = [shapes]
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} should be a {shape_error()} array")
    for shape in shapes:
        if arr.ndim == len(shape):
            for n, d in enumerate(shape):
                if d > 0 and arr.shape[n] != d:
                    raise ValueError(
                        f"{name}.shape[{n}] should be equal to {d}")
            return arr.shape
    raise ValueError(f"{name} should be a {shape_error()} array")


def xdeconv(ydata, ycovar, xamp, xmean, xcovar,
            projection=None, weight=None, classes=None,
            fixpars=None, tol=1.e-6, maxiter=int(1e9),
            regular=0.0):
    """Perform a full extreme deconvolution.

    Parameters
    ----------
    ydata: array-like, shape (n, dy)
        Set of observations involving n data, each having r dimensions
        
    ycovar: array-like, shape (n, dy, dy)
        Array of covariances of the observational data ydata.
    
    xamp: array-like, shape (k)
        Array with the statistical weight of each Gaussian. Updated at the
        exit with the new weights.
        
    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians, updated at the exit with the new
        centers.
    
    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians, updated 
        at the exit with the new covariance matrices.
    
    Optional Parameters
    -------------------
    projection: array-like, shape (n, dy, dx)
        Array of projection matrices: for each datum (n), it is a matrix
        that transform the original d-dimensional vector into the observed
        r-dimensional vector. If None, it is assumed that r=d and that no
        project is performed (equivalently: R is an array if identity 
        matrices).
        
    weights: array-like, shape (n,) 
        Log-weights for each observation, or None
        
    classes: array-like, shape (n, k) 
        Log-probabilities that each observation belong to a given cluster.
        
    fixpars: integer or int array-like, shape (k,)
        Array of bitmasks with the FIX_AMP, FIX_MEAN, and FIX_AMP 
        combinations. If a single value is passed, it is used for all
        components.
        
    tol: double, default=1e-6
        Tolerance for the convergence: if two consecutive loglikelihoods 
        differ by less than tol, the procedure stops.
        
    maxiter: int, default=1e9
        Maximum number of iterations.
    
    regular: double, default=0
        Regularization parameter (use 0 to prevent the regularization).
    """

    nobjs, ydim = check_numpy_array("ydata", ydata, (-1, -1))
    w = np.asfortranarray(ydata.T, dtype=np.float64)

    check_numpy_array("ycovar", ycovar, [(nobjs, ydim), (nobjs, ydim, ydim)])
    if ycovar.ndim == 2:
        S = np.zeros((ydim, ydim, nobjs), order='F')
        for i in range(ydim):
            S[i, i, :] = ycovar[:, i]
    else:
        S = np.asfortranarray(ycovar.T, dtype=np.float64)

    kdim, = check_numpy_array("xamp", xamp, (-1,))
    alpha = np.asfortranarray(xamp.T, dtype=np.float64)

    _, xdim = check_numpy_array("xmean", xmean, (kdim, -1))
    m = np.asfortranarray(xmean.T, dtype=np.float64)

    check_numpy_array("xcovar", xcovar, (kdim, xdim, xdim))
    V = np.asfortranarray(xcovar.T, dtype=np.float64)

    if projection is not None:
        check_numpy_array("projection", projection, (nobjs, ydim, xdim))
        Rt = np.asfortranarray(projection.T, dtype=np.float64)
    else:
        Rt = None

    if weight is not None:
        check_numpy_array("weight", weight, (nobjs,))
        wgh = np.asfortranarray(weight.T, dtype=np.float64)
    else:
        wgh = None

    if classes is not None:
        check_numpy_array("classes", classes, (nobjs, kdim))
        clss = np.asfortranarray(classes.T, dtype=np.float64)
    else:
        clss = None

    if fixpars is not None:
        if isinstance(fixpars, int):
            fixpars = np.repeat(fixpars, kdim)
        check_numpy_array("fixpars", fixpars, (kdim,))
        fix = np.asfortranarray(fixpars.T, dtype=np.uint8)
    else:
        fix = None

    with np.errstate(divide='ignore'):
        oldloglike = em_step(w, S, alpha, m, V, Rt=Rt, logweights=wgh,
                             classes=clss, fixpars=fix, regularization=regular)
    decreased = False
    for iter in range(1, maxiter):
        with np.errstate(divide='ignore'):
            loglike = em_step(w, S, alpha, m, V, Rt=Rt, logweights=wgh,
                              classes=clss, fixpars=fix, regularization=regular)
        diff = loglike - oldloglike
        if diff < 0:
            decreased = True
        if abs(diff) < tol:
            break
        oldloglike = loglike
    if maxiter > 1:
        if iter == maxiter-1:
            warnings.warn(f"xdeconv did not converge after {maxiter} iterations",
                          RuntimeWarning)
        if decreased:
            warnings.warn(
                f"Log-likelihood decreased during the fitting procedure", RuntimeWarning)

        # Saves back all the data: sure this is necessary?
        xamp[:] = alpha
        xmean[:, :] = m.T
        xcovar[:, :, :] = V.T
    return oldloglike
