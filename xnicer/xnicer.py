"""Extreme decomposition and XNICER code.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13"""

# Author: Marco Lombardi <marco.lombardi@gmail.com>

# See https://numpydoc.readthedocs.io/en/latest/format.html

from __future__ import print_function, division
from typing import Union
import numpy as np
import warnings
import copy
from scipy.special import ndtri, logsumexp
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array, check_random_state
from extreme_deconvolution import extreme_deconvolution
from .utilities import log1mexp, cho_solve
from .catalogs import ExtinctionCatalogue


class XDCV(GaussianMixture):
    """Extreme decomposition using a Gaussian Mixture Model

    This class allows to estimate the parameters of a Gaussian mixture
    distribution from data with noise.

    Parameters
    ----------
    n_components : int or list of int, defaults to 1.
        The number of mixture components. If a list of integers is provided, 
        the best value is found using the BIC (see `bic_test`)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of:

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

        Currently only 'full' is implemented.

    tol : float, defaults to 1e-5.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 1e9.
        The maximum number of EM iterations to perform.

    regularization : float, defaults to 0.0
        The regularization parameter used by the extreme deconvolution.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'gmm', 'kmeans', 'random'} or XDCV, defaults to 'gmm'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'gmm'    : data are initialized from a quick GMM fit.
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
            XDCV     : instance of a XDCV already fitted.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_x_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                            if 'spherical',
            (n_x_features, n_x_features)               if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    splitnmerge : int, default to 0.
        The depth of the split and merge path (default = 0, i.e. no split and
        merge is performed).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_x_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                            if 'spherical',
            (n_features, n_x_features)                 if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                            if 'spherical',
            (n_features, n_x_features)                 if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                            if 'spherical',
            (n_features, n_x_features)                 if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    See Also
    --------
    GaussianMixture : Gaussian mixture model.
    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-5,
                 reg_covar=1e-06, max_iter=int(1e9), n_init=1, init_params='gmm',
                 weights_init=None, means_init=None, precisions_init=None,
                 splitnmerge=0, random_state=None, warm_start=False,
                 regularization=0.0, verbose=0, verbose_interval=10):
        super(XDCV, self).__init__(
            n_components=n_components, covariance_type=covariance_type, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.splitnmerge = splitnmerge
        self.weights_ = self.means_ = self.covariances_ = None
        self.converged_ = False
        self.lower_bound_ = -np.infty
        self.regularization = regularization

        # Other model parameters
        # FIXME: Check what to do here!
        self.logL_ = None
        self.n_samples_ = self.n_eff_samples_ = None
        self.n_features_ = None
        self.BIC_ = None

    def _check_parameters(self, Y):
        """Check the Gaussian mixture parameters are well defined."""
        if self.covariance_type not in ['full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['full']"
                             % self.covariance_type)
        init_params = self.init_params
        if self.init_params == 'gmm' or isinstance(self.init_params, XDCV):
            self.init_params = 'kmeans'
        super(XDCV, self)._check_parameters(Y)
        self.init_params = init_params

    def _check_Y_Yerr(self, Y, Yerr, projection=None, log_weight=None):
        """Check the input data Y together with its errors Yerr.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_y_features)
            Input data.

        Yerr: array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection: array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        log_weight: array_like, shape (n_samples,)
            Optional log weights for the various points.

        Returns
        -------
        Y : array-like, shape (n_samples, n_y_features)
            The converted Y array.

        Yerr: array_like, shape (n_samples, n_y_features[, n_y_features])
            The converted Yerr array.

        projection: array_like, shape (n_samples, n_y_features, n_x_features), or None
            The converted projection.

        log_weight: array_like, shape (n_samples,) or None
            The converted log weight.
        """
        Y = check_array(Y, dtype=[np.float64], order='C', estimator='XDCV')
        n_samples, n_y_features = Y.shape
        if n_samples < self.n_components:
            raise ValueError('Expected n_samples >= n_components '
                             'but got n_components = %d, n_samples = %d'
                             % (self.n_components, Y.shape[0]))
        if len(Yerr.shape) not in [2, 3]:
            raise ValueError('Expected a 2d or 3d array for Yerr')
        Yerr = check_array(Yerr, dtype=[
                           np.float64], order='C', ensure_2d=False, allow_nd=True, estimator='XDCV')
        if Yerr.shape[0] != n_samples:
            raise ValueError('Yerr must have the same number of samples as Y')
        if Yerr.shape[1] != n_y_features:
            raise ValueError('Yerr must have the same number of features as Y')
        if len(Yerr.shape) == 3 and Yerr.shape[1] != Yerr.shape[2]:
            raise ValueError(
                'Yerr must be of shape (n_samples, n_y_features, n_y_features)')
        if projection is not None:
            projection = check_array(projection, dtype=[np.float64], order='C', ensure_2d=False,
                                     allow_nd=True, estimator='XDCV')
            if len(projection.shape) != 3:
                raise ValueError(
                    'projection must be of shape (n_samples, n_y_features, n_x_features)')
            if projection.shape[0] != n_samples:
                raise ValueError(
                    'projection must have the same number of samples as Y')
            if projection.shape[1] != n_y_features:
                raise ValueError(
                    'projection must have the same number of features as Y')
        if log_weight is not None:
            log_weight = check_array(log_weight, dtype=[np.float64], order='C', ensure_2d=False,
                                     allow_nd=True, estimator='XDCV')
            if len(log_weight.shape) != 1:
                raise ValueError('log_weight must be a 1d array')
            if log_weight.shape[0] != n_samples:
                raise ValueError(
                    'log_weight must have the same number of samples as Y')
        return Y, Yerr, projection, log_weight

    def _initialize_parameters(self, Y, Yerr, random_state, projection=None, log_weight=None):
        """Initialize the model parameters.

        Parameters
        ----------
        Y: array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr: array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        random_state : RandomState
            A random number generator instance.

        projection: array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        log_weight: array_like, shape (n_samples,)
            Optional log weights for the various points.
        """
        if isinstance(self.init_params, XDCV):
            parameters = list(p.copy()
                              for p in self.init_params._get_parameters())
            self._set_parameters(parameters)
        else:
            if projection is not None:
                identity = np.zeros(shape=projection.shape[1:])
                j = range(max(identity.shape))
                identity[j, j] = 1
                mask = np.all(projection == identity, axis=(1, 2))
                n_x_features = projection.shape[2]
            else:
                mask = np.ones(self.n_samples_, dtype=np.bool)
                n_x_features = Y.shape[1]
            if log_weight is not None:
                mask &= np.log(np.random.rand(log_weight.shape[0])) < log_weight
            if np.sum(mask) < 2 * n_x_features ** 2:
                warnings.warn(
                    'Not enough simple datapoints. Using random values for the initial values.')
                self.means_ = np.random.randn(self.n_components, n_x_features)
                self.covariances_ = np.random.randn(
                    self.n_components, n_x_features, n_x_features)
                self.covariances_ = np.einsum(
                    '...ji,...jk->...ij', self.covariances_, self.covariances_)
                self.weights_ = np.random.rand(self.n_components)
                self.weights_ /= np.sum(self.weights_)
            else:
                init_params = self.init_params
                if init_params == 'gmm':
                    init_params = 'kmeans'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tmp_gmm = GaussianMixture(self.n_components, max_iter=30, covariance_type='full',
                                              init_params=init_params, random_state=self.random_state)
                if self.init_params == 'gmm':
                    tmp_gmm.fit(Y[mask])
                else:
                    tmp_gmm._initialize_parameters(
                        Y[mask], tmp_gmm.random_state)
                self.means_ = tmp_gmm.means_
                self.weights_ = tmp_gmm.weights_
                self.covariances_ = tmp_gmm.covariances_
            self.n_features_ = n_x_features

    def fit(self, Y, Yerr, projection=None, log_weight=None,
            fixamp=None, fixmean=None, fixcovar=None):
        """Fit the XD model to data

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        log_weight : array_like, shape (n_samples,)
            Optional log weights for the various points.

        fixamp : None, bool, or list of bools
            Do not change the amplitude for the selected features.

        fixmean : None, bool, or list of bools
            Do not change the mean for the selected features.

        fixcovar : None, bool, or list of bools
            Do not change the covariance for the selected features.

        The best fit parameters are directly saved in the object.
        """
        # If n_components is an iterable, perform BIC optimization
        if hasattr(self.n_components, '__iter__'):
            self.bic_test(Y, Yerr, self.n_components, projection=projection, log_weight=log_weight)
            return self

        # Temporarily change some of the parameters to perform a first
        # initialization
        Y, Yerr, projection, log_weight = self._check_Y_Yerr(
            Y, Yerr, projection, log_weight)
        self._check_initial_parameters(Y)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not ((self.warm_start or fixamp or fixmean or fixcovar)
                       and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        self.n_samples_, _ = Y.shape
        if log_weight is not None:
            weight = np.exp(log_weight)
            self.n_eff_samples_ = np.sum(weight)**2 / np.sum(weight**2)
        else:
            self.n_eff_samples_ = self.n_samples_

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(Y, Yerr, random_state,
                                            projection=projection, log_weight=log_weight)
                self.lower_bound_ = -np.infty

            self.lower_bound_ = extreme_deconvolution(Y, Yerr,
                                                      self.weights_, self.means_, self.covariances_,
                                                      tol=self.tol, maxiter=self.max_iter,
                                                      w=self.reg_covar, splitnmerge=self.splitnmerge,
                                                      weight=log_weight, logweight=True,
                                                      projection=projection,
                                                      fixamp=fixamp, fixmean=fixmean,
                                                      fixcovar=fixcovar)
            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                self.converged_ = True

        self._set_parameters(best_params)
        # Computes the BIC
        ndim = self.means_.shape[1]
        cov_params = self.n_components * ndim * (ndim + 1) / 2.0
        mean_params = ndim * self.n_components
        n_params = int(cov_params + mean_params + self.n_components - 1)
        self.bic_ = -2 * np.sum(self.lower_bound_) + \
            n_params * np.log(self.n_eff_samples_)
        return self

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.lower_bound_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.lower_bound_) = params

    def score_samples_components(self, Y, Yerr, projection=None):
        """Compute the log probabilities for each component

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        Returns
        -------
        logprob : array_like, shape (n_samples, n_components)
            Log probabilities of each data point in X.
        """
        Y, Yerr, projection, _ = self._check_Y_Yerr(
            Y, Yerr, projection=projection)
        n_y_features = Y.shape[1]
        Y = Y[:, np.newaxis, :]
        if Yerr.ndim == 2:
            tmp = np.zeros((Yerr.shape[0], Yerr.shape[1], Yerr.shape[1]))
            for k in range(Yerr.shape[1]):
                tmp[:, k, k] = Yerr[:, k]
            Yerr = tmp
        Yerr = Yerr[:, np.newaxis, :, :]
        if projection is None:
            T = Yerr + self.covariances_
            delta = Y - self.means_
        else:
            P = projection[:, np.newaxis, :, :]
            V = self.covariances_[np.newaxis, :, :, :]
            mu = self.means_[np.newaxis, :, :]
            T = Yerr + np.einsum('...ij,...jk,...lk->...il', P, V, P)
            delta = Y - np.einsum('...ik,...k->...i', P, mu)
        # Instead of computing the inverse of the full covariance matrix
        # does a Cholesky decomposition, an operation that is ~100 times
        # faster! Then uses forward substitution to invert it.
        Tc = np.linalg.cholesky(T)
        Y = cho_solve(Tc, delta)
        chi2 = np.sum(Y*Y, axis=2) / 2.0
        Tlogdet = np.sum(np.log(np.diagonal(Tc, axis1=-2, axis2=-1)), axis=-1)
        return - chi2 - Tlogdet - (n_y_features * np.log(2.0*np.pi) / 2.0)

    def score_samples(self, Y, Yerr, projection=None):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        log = self.score_samples_components(Y, Yerr, projection=projection)
        return logsumexp(np.log(self.weights_) + log, axis=-1)

    def score(self, Y, Yerr, projection=None, log_weight=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        log_weight : array_like, shape (n_samples,)
            Optional log weights for the various points.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        if log_weight is not None:
            weight = np.exp(log_weight)
            return np.sum(weight*self.score_samples(Y, Yerr, projection=projection)) / np.sum(weight)
        else:
            return np.mean(self.score_samples(Y, Yerr, projection=projection))

    def bic(self, Y=None, Yerr=None, projection=None, log_weight=None, ranks=None):
        """Compute Bayesian information criterion for current model and
        proposed data.

        Computed in the same way as the scikit-learn GMM model computes
            the BIC.

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        log_weight : array_like, shape (n_samples,)
            Optional log weights for the various points.

        Returns
        -------
        bic : float
            BIC for the model and data (lower is better).
        """
        if Y is None:
            score = self.lower_bound_
            n_samples = self.n_eff_samples_
        else:
            score = self.score(Y, Yerr, projection=projection, log_weight=log_weight)
            n_samples = Y.shape[0]
            if ranks is not None:
                n_samples = np.sum(ranks)
            elif projection is not None:
                ranks = np.empty(projection.shape[0], dtype=np.int64)
                for n in range(projection.shape[0]):
                    ranks[n] = np.linalg.matrix_rank(projection[n])
                n_samples = np.sum(ranks)
        return (-2 * score * n_samples +
                self._n_parameters() * np.log(n_samples))

    def aic(self, Y=None, Yerr=None, projection=None, log_weight=None, ranks=None):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        log_weight : array_like, shape (n_samples,)
            Optional log weights for the various points.

        Returns
        -------
        aic : float
            The lower the better.
        """
        if Y is None:
            score = self.lower_bound_
            n_samples = self.n_eff_samples_
        else:
            score = self.score(Y, Yerr, projection=projection, log_weight=log_weight)
            n_samples = Y.shape[0]
        return -2 * score * n_samples + 2 * self._n_parameters()

    def bic_test(self, Y, Yerr, components, projection=None,
                 no_err=False, **kw):
        """Compute Bayesian information criterion for a range of components.

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_y_features)
            Input data.

        Yerr : array_like, shape (n_samples, n_y_features[, n_y_features])
            (Co)variances on input data.

        components : array_like
            List of values to try for n_compments.

        projection : array_like (optional), shape (n_samples, n_y_features, n_x_features)
            An optional projection matrix, especially useful when there are
            missing data.

        no_err : bool (optional)
            Flag for whether to compute BIC using the error array included
            or not (default = False).

        kw : dictionary
            All other parameters are directly passed to `fit`.

        Returns
        -------
        bics : array_like, shape = (len(components),)
            BIC for each value of n_components.

        optimal_n_comp : float
            Number of components with lowest BIC score.

        lowest_bic : float
            Lowest BIC from the scores computed.
        """
        bics = np.array([])
        lowest_bic = np.infty
        lowest_par = None
        optimal_n_comp = 0
        if no_err:
            Yerr_zero = np.zeros_like(Yerr)
        for n_components in components:
            self.n_components = n_components
            self.fit(Y, Yerr, projection, **kw)
            if no_err:
                bics = np.append(bics, self.bic(Y, Yerr_zero, projection))
            else:
                bics = np.append(bics, self.bic_)
            if bics[-1] < lowest_bic:
                optimal_n_comp = n_components
                optimal_bic = bics[-1]
                optimal_par = self.get_params()
                optimal_res = self._get_parameters()
            print(n_components, bics[-1])
        if lowest_par:
            self.bic_ = optimal_bic
            self.set_params(**optimal_par)
            self._set_parameters(optimal_res)
        return bics, optimal_n_comp, optimal_bic


class XNicer(BaseEstimator):
    """Class XNicer, used to perform the XNicer extinction estimate.

    This class allows one to estimate the extinction from two color catalogues,
    a control field and a science field. It uses the extreme deconvolution
    provided by the XDCV class.

    Parameters
    ----------
    xdcv : XDCV
        An XDCV instance used to perform all necessary extreme deconvolutions.

    extinctions : array-like, shape (_,)
        A 1D vector of extinctions used to perform a selection correction.

    extinction_vec : array-like, shape (n_bands,)
        The extinction vector, that is A_band / A_ref, for each band.
    """

    def __init__(self, xdcv, extinctions=None, extinction_vec=None):
        self.xdcv = xdcv
        if extinctions is None:
            extinctions = [0.0]
        self.extinctions = np.array(extinctions)
        self.extinction_vec = np.array(extinction_vec)
        self.band = None
        self.log_weights_ = None
        self.calibration = None

    def fit(self, cat, band=None):
        """Initialize the class with control field data.

        Parameters
        ----------
        cat : PhotometricCatalogue
            The control field data, as a PhotometricCatalogue.

        band : int or None, default to None
            The band to be used for magnitude corrections. If None, no magnitude
            correction is performed.
        """
        assert cat.n_bands == len(self.extinction_vec), \
            "The number of bands does not match the length of extinction vector"
        if band is not None:
            # Compute the cumulative distribution and use it in the magnitude
            # mapping.
            eps = 1e-7
            w = np.where(cat.mag_errs[:, band] < cat.max_err)[0]
            mags = np.sort(cat.mags[w, band])
            # This function ensures that the distribution of the mapped
            # magnitudes follow a normal distribution N(0, 1).
            self.map = lambda _: ndtri(
                np.minimum(1 - eps,
                           np.maximum(eps,
                                      np.interp(_, mags, (np.arange(len(mags)) + 0.5) / len(mags)))))
        else:
            self.map = lambda _: _
        self.band = band
        for n, extinction in enumerate(self.extinctions):
            # Add the extinction, as requested
            cat_A = cat.extinguish(extinction * self.extinction_vec,
                                   apply_completeness=True, update_errors=False)
            cat_A.mags -= extinction * self.extinction_vec
            # FIXME Investigate if we need to use self.map or a translated
            # self.map
            cols_A = cat_A.get_colors(use_projection=True, band=band,
                                      map_mags=self.map)
                                      # map_mags=lambda m: self.map(m + extinction * self.extinction_vec[band]))
            if n == 0:
                self.xdcv.fit(cols_A.cols, cols_A.col_covs, cols_A.projections, 
                              log_weight=cols_A.log_probs)
            else:
                self.xdcv.fit(cols_A.cols, cols_A.col_covs, cols_A.projections,
                              fixmean=True, fixcovar=True, log_weight=cols_A.log_probs)
            if self.log_weights_ is None:
                # We could set this earlier in the __init__, but it does not
                # work in case the numbero of components for self.xdcv is an
                # array (this is possible if we request a BIC criterion)
                self.log_weights_ = np.zeros(
                    (len(self.extinctions), self.xdcv.n_components))
            self.log_weights_[n] = np.log(self.xdcv.weights_)


    def calibrate(self, cat, extinctions, **kw):
        """Perform a full calibration of the algorithm for a set of test extinctions.
        
        The calibration works by simulating in turn all extionctions provided,
        and by checking the final bias and (inverse) variances of the
        estimated data. This way we can predict and counter-balance the
        effects of a (differential) extinction, as in the Nicest algorithm. In
        fact, this implements the XNicest algorithm, a generalized Nicest
        algorithm that also works for general number counts distributions.
        Note that the algorithm resembles, in some sense, the Mack & Mueller
        non-parametric regression (Mack, Y.P. & and Mueller, H.-G., 1989a,
        "Derivative Estimation in Non-parametric Regression with Random
        Predictor Variables," Sankhya, Ser A, 51, 59-72).

        Parameters
        ----------
        cat : PhotometryCatalogue
            A PhotometryCatalogue with the science field data. For optimal
            results, cat should have associated log probabilities (see
            `PhotometricCalogue.add_log_probs`).

        extinctions : list of array of floats
            The list of extinctions to use for the calibration. All
            extinctions should be non-negative. The first extinction must be 0.

        kw : dictionary
            Additional keywords are directly passed to `predict`.
        """
        self.calibration = None
        biases = []
        ivars = []
        if cat.log_probs is None:
            warnings.warn('For best results add log probabilities to cat')
        for extinction in extinctions:
            cat_t = cat.extinguish(self.extinction_vec * extinction)
            ext_t = self.predict(cat_t, **kw)
            objweight = np.exp(ext_t.log_weight_) / ext_t.variance_
            ivar = np.sum(objweight)
            mean = np.sum(ext_t.mean_ * objweight) / ivar
            biases.append(mean - extinction)
            ivars.append(ivar)
        self.calibration = (np.array(extinctions), np.array(biases),
                            np.array(ivars) / ivars[0])
            
    def predict(self, cat, use_projection=True, use_band=None, n_iters=3):
        """Compute the extinction for each object of a PhotometryCatalogue

        Parameters
        ----------
        cat : PhotometryCatalogue
            A PhotometryCatalogue with the science field data.

        use_projection : boolean, default=True
            Parameter passed directly to `PhotometricCatalogue`.

        use_band : int or None, default=None
            If an int, the corresponding band is added to the catalogue colors
            to as an additional parameter to the extreme deconvolution. For
            this to work, the associated magnitude are mapped, so that the
            mapped values are distributed normally with vanishing mean and
            unity variance. The use of the band is often not necessary and
            similar results are obtained without it. Also, it slows down the
            computations, and therefore by default it is inhibited.

        n_iters : int, default=1
            Number of iterations to use during the fitting procedure. If
            n_iters=1, then no adjustment for the extinction selection
            effect is made.
        """
        # Check if we need to use the band
        band = self.band
        if use_band == None or use_band == False:
            band = None
        
        # Compute the colors: initially we ignore the self.band
        cols = cat.get_colors(use_projection=use_projection, band=None,
                              map_mags=self.map)

        # Allocate the result: note that we use cols.n_objs, in case the
        # original catalogue has the log_probs attribute. This would trigger
        # the generation of a different number of colors. Note also that, in
        # this case, we need to keep track of the original objects and of the
        # associated probabilities.
        res = ExtinctionCatalogue(cols.n_objs, self.xdcv.n_components,
                                  selection=cols.selection)

        # Compute the extinction vector; this is OK
        color_ext_vec = self.extinction_vec[:-1] - self.extinction_vec[1:]

        # We now perform a two-step process. First, we work ignoring the
        # band index, if this is used (band is not None); then we repeat
        # the same analysis by using the magnitude at the computed original
        # magnitude for each star. We keep computing the star magnitude as
        # necessary.
        for step in (0, 1):
            xdcv = copy.deepcopy(self.xdcv)
            if step == 0:
                if band is not None:
                    # "Flatten" the GMM along the magnidute axis
                    xdcv.n_features_ -= 1
                    xdcv.means_ = xdcv.means_[:, :-1]
                    xdcv.covariances_ = xdcv.covariances_[:, :-1, :-1]
                    # Compute the colors: initially we ignore the self.band
                    cols = cat.get_colors(use_projection=use_projection, band=None)
                else:
                    continue
            else:
                # Recompute the colors; uses magnitudes where one has taken
                # rid of the previously computed extinctions
                if band is not None:
                    cols = cat.get_colors(use_projection=use_projection, band=self.band,
                                          map_mags=self.map,
                                          extinctions=res.mean_ * self.extinction_vec[self.band])
                    # Add a null term to the color extinction vector
                    color_ext_vec = np.hstack((color_ext_vec, [0.0]))
                    # Now trace back all magnitudes to their original values
                else:
                    # Note: since band=None we have not computed cols above
                    cols = cat.get_colors(use_projection=use_projection, band=None)

            # Compute all parameters (except the weights) for the 1st deconvolution
            if not use_projection:
                T = cols.col_covs[:, np.newaxis, :, :] + xdcv.covariances_
                Tc = np.linalg.cholesky(T)
                d = cols.cols[:, np.newaxis, :] - xdcv.means_[np.newaxis, :, :]
                T_k = cho_solve(Tc, color_ext_vec)
            else:
                T = cols.col_covs[:, np.newaxis, :, :] + \
                    np.einsum('...ij,...jk,...lk->...il',
                              cols.projections[:, np.newaxis, :, :],
                              xdcv.covariances_[np.newaxis, :, :, :],
                              cols.projections[:, np.newaxis, :, :])
                Tc = np.linalg.cholesky(T)
                d = cols.cols[:, np.newaxis, :] - \
                    np.einsum('...ij,...j->...i', cols.projections[:, np.newaxis, :, :],
                              xdcv.means_[np.newaxis, :, :])
                T_k = cho_solve(Tc, np.einsum('...ij,...j->...i',
                                               cols.projections,
                                               color_ext_vec)[:, np.newaxis, :])
            T_d = cho_solve(Tc, d)
            Tlogdet = np.sum(np.log(np.diagonal(Tc, axis1=-2, axis2=-1)), axis=-1)
            sigma_k2 = 1.0 / np.sum(T_k * T_k, axis=-1)
            # We are now ready to compute the means, variances, and weights for the
            # GMM of the extinction. Each of these parameters has a shape
            # (n_objs, n_bands - 1). We can go on with the means and variances, but
            # the weights require more care, because they are linked to the xdcv
            # weights, and these in turn depend on the particular extinction of the
            # star. We therefore initially only compute log_weights0_, the log of
            # the weights of the extinction GMM that does not include any term related
            # to the xdcv weight.
            res.means_[:] = sigma_k2 * np.sum(T_k * T_d, axis=-1)
            # variances_ = np.sum(T_d * T_d, axis=-1) - self.ext*self.ext / sigma_k2
            res.variances_[:] = sigma_k2
            C_k = np.sum(T_d * T_d, axis=-1) - res.means_*res.means_ / sigma_k2
            log_weights0_ = - Tlogdet - \
                (cat.n_bands - 1) * np.log(2.0*np.pi) / 2.0 - \
                C_k / 2.0 + np.log(2.0*np.pi*sigma_k2) / 2.0
            # Fine, now need to perform two loops: one outer loop where we iterate
            # n_iters time, and an internal loop that computes the weights for each
            # extinction step. To proceed we define a new array, log_ext_weights,
            # that saves the log of the contribution of a particular extinction step
            # for each object; we initially assume that all objects only have a
            # contribution from the first extinction step.
            log_ext_weights = np.full(
                (cols.n_objs, len(self.extinctions)), -np.inf)
            log_ext_weights[:, 0] = 0.0
            for _ in range(n_iters):
                res.log_weights_ = log_weights0_ + \
                    logsumexp(self.log_weights_[
                              np.newaxis, :, :] + log_ext_weights[:, :, np.newaxis], axis=1)
                res.log_evidence_ = logsumexp(res.log_weights_, axis=-1)
                res.log_weights_ -= res.log_evidence_[..., np.newaxis]
                # Now we need to update the weights for the extinction steps according
                # to each object's average extinction
                for e, extinction in enumerate(self.extinctions):
                    log_ext_weights[:, e] = res.score_samples(
                        np.repeat(extinction, cols.n_objs), np.zeros(cols.n_objs))
                log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
            res.update_()

        # In case we have a calibrated object, perform the bias correction and
        # the XNicest estimates
        if self.calibration:
            log_ext_weights = np.empty((cols.n_objs, len(self.calibration[0])))
            # Perform a first pass with no bias corrected
            for e, extinction in enumerate(self.calibration[0]):
                log_ext_weights[:, e] = res.score_samples(
                        np.repeat(extinction, cols.n_objs), np.zeros(cols.n_objs))
            log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
            ext_weights = np.exp(log_ext_weights)
            # Now do the bias correction, based on the measured extinctions
            bias = np.sum(ext_weights * self.calibration[1], axis=1)
            res.mean_ -= bias
            res.means_ -= bias[:, np.newaxis]
            # Recompute the weights
            # FIXME: is this really necessary?
            for e, extinction in enumerate(self.calibration[0]):
                log_ext_weights[:, e] = res.score_samples(
                    np.repeat(extinction, cols.n_objs), np.zeros(cols.n_objs))
            log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
            ext_weights = np.exp(log_ext_weights)
            # Finally perform the XNicest evaluations
            res.xnicest_weight = np.sum(
                ext_weights / self.calibration[2], axis=1)
            res.xnicest_bias = np.sum(ext_weights * self.calibration[0]
                / self.calibration[2], axis=1) / res.xnicest_weight - np.sum(
                    ext_weights * self.calibration[0], axis=1)
        # Now, in case of use of log_probs, we need to correct the weights so
        # to include the original color weights.
        if cols.log_probs is not None:
            res.log_weights_ += cols.log_probs[:, np.newaxis]
            res.log_weight_ += cols.log_probs
        return res
