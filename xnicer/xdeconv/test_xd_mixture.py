import numpy as np
from scipy.special import logsumexp
import warnings
from .em_step import FIX_NONE, FIX_AMP, FIX_CLASS, FIX_MEAN, FIX_COVAR, FIX_ALL # pylint: disable=no-name-in-module
from .xd_mixture import XD_Mixture
from .test_em_step import generate_data


def generate_test_xdmix(xamp, xmean, xcovar, ycovar, npts=1000,
                        xclass=None, use_weight=False, use_projection=False,
                        use_classes=False, fixpars=None, seed=1,
                        confusion=0.01, silent=True, **kw):
    """Create a full test for the XD_Mixture class.
    
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
    xdm = XD_Mixture(xamp.shape[0])
    xdm.fit(data['ydata'], data['ycovar'], projection=data['projection'],
            log_weight=data['weight'], Yclass=data['classes'],
            fixpars=data['fixpars'])
    
    # Sort the results so that they are as similar as possible to the original
    # centers
    dist = np.sum((xmean[:, np.newaxis, :] - xdm.means_[np.newaxis, :, :])**2, 
                  axis=2)
    s = np.argmin(dist, axis=1)
    xmean_s = xdm.means_[s]
    xamp_s = xdm.weights_[s]
    xcovar_s = xdm.covariances_[s]

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
    
    return ((xamp_s, (xamp_s - xamp) / amp_err),
            (xmean_s, (xmean_s - xmean) / mean_err),
            (xcovar_s, (xcovar_s - xcovar) / cov_err))


def test_xdmix_1d():
    """Test using 1D data w/ 2 or 3 clusters"""
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_xdmix([0.5, 0.5], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1, 
                                      npts=npts, seed=iter, silent=True)
        assert np.all(
            np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(
            np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(
            np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"
        
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_xdmix([0.25, 0.75], [[3.0], [-3.0]], [[[1.0]], [[0.5]]], 1,
                                      npts=npts, seed=iter, silent=True)
        assert np.all(
            np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(
            np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(
            np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"

    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_xdmix([0.25, 0.25, 0.5], [[0.0], [5.0], [-4.0]], [[[0.8]], [[1.2]], [[1.0]]], 0.5,
                                      npts=npts, seed=1, silent=True)
        assert np.all(
            np.abs(a[1]) < 9), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 9"
        assert np.all(
            np.abs(b[1]) < 9), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 9"
        assert np.all(
            np.abs(c[1]) < 9), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 9"


def test_xdmix_2d_projection():
    """Test using 2D data in projection w/ 2 or 3 clusters"""
    for iter in range(5):
        npts = [100, 300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_xdmix([0.5, 0.5], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                      use_projection='random', npts=npts, seed=iter, silent=True)
        assert np.all(
            np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(
            np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(
            np.abs(c[1]) < 3), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 3"
        
    for iter in range(4):
        npts = [300, 1000, 3000, 10000][iter]
        a, b, c = generate_test_xdmix([0.25, 0.75], [[3.0, 1.0], [-3.0, -1.0]], [1.0, 0.5], [1.0, 1.0],
                                      use_projection='random', npts=npts, seed=iter, silent=True)
        assert np.all(
            np.abs(a[1]) < 3), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 3"
        assert np.all(
            np.abs(b[1]) < 3), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 3"
        assert np.all(
            np.abs(c[1]) < 6), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 6"

    for iter in range(3):
        npts = [1000, 3000, 10000][iter]
        a, b, c = generate_test_xdmix([0.25, 0.25, 0.5], [[0.0, 0.0], [4.0, 2.0], [-3.0, 1.0]], [0.8, 1.2, 1.0], 1,
                                      use_projection='identity', npts=npts, seed=1, silent=True)
        assert np.all(
            np.abs(a[1]) < 7), f"|a[1]| = {np.max(np.abs(a[1])):.2f} > 7"
        assert np.all(
            np.abs(b[1]) < 7), f"|b[1]| = {np.max(np.abs(b[1])):.2f} > 7"
        assert np.all(
            np.abs(c[1]) < 7), f"|c[1]| = {np.max(np.abs(c[1])):.2f} > 7"
