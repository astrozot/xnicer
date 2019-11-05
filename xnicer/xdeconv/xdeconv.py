import logging, warnings
import numpy as np
from scipy.special import logsumexp
from .em_step import em_step, FIX_NONE, FIX_AMP, FIX_CLASS  # pylint: disable=no-name-in-module
from .em_step import FIX_MEAN, FIX_COVAR, FIX_ALL, _scores  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

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
    arr.shape: tuple
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
            xclass=None, projection=None, weight=None, classes=None,
            fixpars=None, tol=1.e-6, maxiter=int(1e9),
            regular=0.0, splitnmerge=0):
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
    xclass: array-like, shape (k, c)
        Array with the statistical weight of each Gaussian for each class.
        Updated at the exit with the new weights. The sum of all classes for
        a single cluster k is unity.
    
    projection: array-like, shape (n, dy, dx)
        Array of projection matrices: for each datum (n), it is a matrix
        that transform the original d-dimensional vector into the observed
        r-dimensional vector. If None, it is assumed that r=d and that no
        project is performed (equivalently: R is an array if identity 
        matrices).
        
    weights: array-like, shape (n,) 
        Log-weights for each observation, or None
        
    classes: array-like, shape (n, c) 
        Log-probabilities that each observation belong to a given class.
        
    fixpars: integer or int array-like, shape (k,)
        Array of bitmasks with the FIX_AMP, FIX_MEAN, FIX_AMP, and FIX_CLASS
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
    if xclass is not None:
        kdim, cdim = check_numpy_array("xclass", xclass, (kdim, -1))
        alphaclass = np.asfortranarray(xclass.T, dtype=np.float)
        alphaclass /= np.sum(alphaclass, axis=0)
    else:
        cdim = 1
        alphaclass = np.ones((cdim, kdim), dtype=np.float64, order='F') / cdim

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
        wgh = np.zeros(nobjs, dtype=np.float64, order='F')

    if classes is not None:
        check_numpy_array("classes", classes, (nobjs, cdim))
        clss = np.asfortranarray(classes.T, dtype=np.float64)
    else:
        clss = np.zeros((cdim, nobjs), dtype=np.float64, order='F') - \
            np.log(cdim)

    if fixpars is not None:
        if isinstance(fixpars, int):
            fixpars = np.repeat(fixpars, kdim)
        check_numpy_array("fixpars", fixpars, (kdim,))
        fix = np.asfortranarray(fixpars.T, dtype=np.uint8)
    else:
        fix = None

    with np.errstate(divide='ignore'):
        oldloglike = em_step(w, S, alpha, alphaclass, m, V, wgh, clss, 
                             Rt=Rt, fixpars=fix, regularization=regular)
        logger.debug(f'Initial log-like={oldloglike}')
        decreased = False
        for iter in range(1, maxiter):
            loglike = em_step(w, S, alpha, alphaclass, m, V, wgh, clss, 
                              Rt=Rt, fixpars=fix, regularization=regular)
            diff = loglike - oldloglike
            oldloglike = loglike
            if diff < 0:
                decreased = True
            if abs(diff) < tol:
                break
    logger.debug(f'Loop exit after {iter} iterations')
    logger.debug(f'Final log-like={oldloglike}.')

    # Split and merge code
    while True:
        # Saves back all the data: this part of the code has to be executed
        # always!
        xamp[:] = alpha.T
        if xclass is not None:
            xclass[:, :] = alphaclass.T
        xmean[:, :] = m.T
        xcovar[:, :, :] = V.T

        # Shortcut: exit if no split and merge has to be performed
        if splitnmerge == 0:
            break
        
        # Comput the split and merge ranks (ijk is an iterator!)
        ijk = splitnmerge_rank(ydata, ycovar, xamp, xmean, xcovar,
                               xclass, projection, classes)
        sm_iter = 0
        eps = 1.0
        for i, j, k in ijk:
            sm_iter += 1
            if sm_iter > splitnmerge:
                # We have tried all allowed combinations for split & merge:
                # quit the loop
                break
            # Prepare the new set of parameters
            logger.debug(f'Performing split-and-merge ({i},{j},{k})')
            alpha1 = np.copy(alpha)
            alpha1[i] = alpha[i] + alpha[j]
            alpha1[j] = alpha1[k] = alpha[k] * 0.5
            m1 = np.copy(m)
            m1[:,i] = (alpha[i]*m[:,i] + alpha[j]*m[:,j]) / alpha1[i]
            det = np.linalg.det(V[:,:,k]) ** (1 / xdim)
            m1[:,j] = m[:,k] + eps * np.random.randn(xdim) * np.sqrt(det)
            m1[:,k] = m[:,k] + eps * np.random.randn(xdim) * np.sqrt(det)
            V1 = np.copy(V)
            V1[:,:,i] = (alpha[i]*V[:,:,i] + alpha[j]*V[:,:,j]) / alpha1[i]
            V1[:,:,j] = np.identity(xdim) * det
            V1[:,:,k] = np.copy(V1[:,:,j])
            alphaclass1 = np.copy(alphaclass)
            alphaclass1[:,i] = (alphaclass[:,i] + alphaclass[:,j]) * 0.5
            alphaclass1[:,j] = np.copy(alphaclass[:,k])
            alphaclass1[:,k] = np.copy(alphaclass1[:,j])
            # Now performs a minimization keeping the other parameters fixed.
            # This is the so-called partial EM procedure
            if fix is not None:
                fix1 = np.copy(fix)
            else:
                fix1 = np.zeros(kdim, dtype=np.uint8)
            fix1[:] = FIX_ALL
            fix1[i] = fix1[j] = fix1[k] = FIX_NONE
            with np.errstate(divide='ignore'):
                for piter in range(10):
                    em_step(w, S, alpha1, alphaclass1, m1, V1, wgh, clss,
                            Rt=Rt, fixpars=fix1, regularization=regular)
            iter += piter
            # Now perform again the full EM procedure
            if fix is not None:
                fix1 = np.copy(fix)
            else:
                fix1 = None
            with np.errstate(divide='ignore'):
                old_ll = em_step(w, S, alpha1, alphaclass1, m1, V1, wgh, clss,
                                 Rt=Rt, fixpars=fix1, regularization=regular)
                for piter in range(maxiter - iter):
                    ll = em_step(w, S, alpha1, alphaclass1, m1, V1, wgh, clss,
                                 Rt=Rt, fixpars=fix1, regularization=regular)
                    diff = ll - old_ll
                    old_ll = ll
                    if diff < 0:
                        decreased = True
                    if abs(diff) < tol:
                        break
            iter += piter
            # Regardless if we converged or not, check if we have increased
            # the log-likelihood wrt the original one. My check takes into
            # account (partially, by 50%) the fact that we have performed
            # piter more iterations: so it is natural to expect that we have
            # somewhat improved our log-likelihood.
            if ll - loglike > tol * piter * 0.5:
                # OK, we have done a good job, copy the final parameters
                logger.info(f'Success w/ split and merge ({i},{j},{k}): {ll} > {loglike}')
                loglike = oldloglike = ll
                alpha = alpha1
                m = m1
                V = V1
                alphaclass = alphaclass1
                # ...and try the split & merge again
                break
            else:
                logger.debug(f'Failure w/ split and merge: {ll} < {loglike}')

            # If we did not improve, we will try th next combination of
            # triplet for the split & merge.
        if sm_iter > splitnmerge:
            logger.debug(f'No more split and merge possibilities')
            # We have exhausted our possibilities for split and merge: exit
            break

    # Final checks: show warnings if necessary
    if maxiter > 1:
        if iter == maxiter-1:
            logger.warn(
                f"xdeconv did not converge after {maxiter} iterations")
            warnings.warn(
                f"xdeconv did not converge after {maxiter} iterations")
        if decreased:
            logger.warn(
                f"Log-likelihood decreased during the fitting procedure")

    return oldloglike


def scores(ydata, ycovar, xmean, xcovar,
           xclass=None, projection=None, classes=None):
    """Compute the scores (log-likelihoods) for all objects and components.

    Note that the computed log-likelihood does not include the term associated
    with the amplitude (statistical weight) of each Gaussian, but possibly it
    includes the amplitude of each class within each Gaussian.
    
    Parameters
    ----------
    ydata: array-like, shape (n, dy)
        Set of observations involving n data, each having r dimensions
        
    ycovar: array-like, shape (n, dy, dy)
        Array of covariances of the observational data ydata.
    
    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians.
    
    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians.
    
    Optional Parameters
    -------------------
    xclass: array-like, shape (k, c)
        Array with the statistical weight of each Gaussian for each class.
        Updated at the exit with the new weights. The sum of all classes for
        a single cluster k is unity.
    
    projection: array-like, shape (n, dy, dx)
        Array of projection matrices: for each datum (n), it is a matrix
        that transform the original d-dimensional vector into the observed
        r-dimensional vector. If None, it is assumed that r=d and that no
        project is performed (equivalently: R is an array if identity 
        matrices).
        
    classes: array-like, shape (n, c) 
        Log-probabilities that each observation belong to a given class.
        
    Returns
    -------
    scores: array-like, shape (n, k)
        The log-likelihoods for each point and each cluster.
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

    kdim, xdim = check_numpy_array("xmean", xmean, (-1, -1))
    m = np.asfortranarray(xmean.T, dtype=np.float64)

    if xclass is not None:
        kdim, cdim = check_numpy_array("xclass", xclass, (kdim, -1))
        alphaclass = np.asfortranarray(xclass.T, dtype=np.float)
        alphaclass /= np.sum(alphaclass, axis=0)
    else:
        cdim = 1
        alphaclass = np.ones((cdim, kdim), dtype=np.float64, order='F') / cdim

    check_numpy_array("xcovar", xcovar, (kdim, xdim, xdim))
    V = np.asfortranarray(xcovar.T, dtype=np.float64)

    if projection is not None:
        check_numpy_array("projection", projection, (nobjs, ydim, xdim))
        Rt = np.asfortranarray(projection.T, dtype=np.float64)
    else:
        Rt = None

    if classes is not None:
        check_numpy_array("classes", classes, (nobjs, cdim))
        clss = np.asfortranarray(classes.T, dtype=np.float64)
    else:
        clss = np.zeros((cdim, nobjs), dtype=np.float64, order='F') - \
            np.log(cdim)

    qs = np.zeros((kdim, nobjs), order='F')
    with np.errstate(divide='ignore'):
        _scores(w, S, alphaclass, m, V, clss, qs, Rt=Rt)

    return np.ascontiguousarray(qs.T)


def splitnmerge_rank(ydata, ycovar, xamp, xmean, xcovar, xclass=None, 
                     projection=None, classes=None, fixpars=None):
    """Computes the order for the split and merge algorithm.
    
    ydata: array-like, shape (n, dy)
        Set of observations involving n data, each having r dimensions
        
    ycovar: array-like, shape (n, dy, dy)
        Array of covariances of the observational data ydata.
    
    xamp: array-like, shape (k,)
        The amplitude of the Gaussians.
        
    xmean: array-like, shape (k, dx)
        Centers of multivariate Gaussians.
    
    xcovar: array-like, shape (k, dx, dx)
        Array of covariance matrices of the multivariate Gaussians.
    
    Optional Parameters
    -------------------
    xclass: array-like, shape (k, c)
        Array with the statistical weight of each Gaussian for each class.
        Updated at the exit with the new weights. The sum of all classes for
        a single cluster k is unity.
    
    projection: array-like, shape (n, dy, dx)
        Array of projection matrices: for each datum (n), it is a matrix
        that transform the original d-dimensional vector into the observed
        r-dimensional vector. If None, it is assumed that r=d and that no
        project is performed (equivalently: R is an array if identity 
        matrices).
        
    classes: array-like, shape (n, c) 
        Log-probabilities that each observation belong to a given class.
        
    fixpars: array-like, shape (k,) or None
        Array of bitmasks with the FIX_AMP, FIX_MEAN, FIX_AMP, and FIX_CLASS
        combinations. If a single value is passed, it is used for all
        components. All clusters with fixpars != FIX_NONE will not be used for 
        spitting and merging.
        
    Returns
    -------
    A generator that, each time it is used, will return a triplet (i,j,k):
    the i-th and j-th clusters are candidates for a merging, while the k-th
    cluster is a candidate for a splitting.
    """
    if fixpars is None:
        idx = np.arange(xamp.shape[0])
    else:
        if isinstance(fixpars, int):
            fixpars = np.repeat(fixpars, int)
        idx = np.where(fixpars == FIX_NONE)[0]
    # Compute j_merge = log(P P^T), where P is the probability that each point
    # belongs to a given cluster.
    s = scores(ydata, ycovar, xmean, xcovar,
               xclass=xclass, projection=projection, classes=classes)
    j_merge = logsumexp(s[:, :, np.newaxis] + s[:, np.newaxis, :], axis=0)
    j_merge = j_merge[np.ix_(idx, idx)]
    # Compute j_split, as the local Kullback divergence.
    q_ij = np.log(xamp[np.newaxis, :]) + s
    q_ij -= logsumexp(q_ij, axis=1)[:, np.newaxis]
    q_ij -= logsumexp(q_ij, axis=0)[np.newaxis, :]
    j_split = np.sum(np.exp(q_ij) * (q_ij - s), axis=0)
    j_split = j_split[idx]
    # Now sort the various triples: start with the best couples of j_merge
    i, j = np.tril_indices_from(j_merge, -1)
    s = np.argsort(-j_merge[i, j])
    i = idx[i[s]]
    j = idx[j[s]]
    # Same with j_split: sort the array against the best index
    k = idx[np.argsort(-j_split)]
    # Creates the final result
    for curr_i, curr_j in zip(i, j):
        for curr_k in k:
            if curr_k != curr_i and curr_k != curr_j:
                yield curr_i, curr_j, curr_k
