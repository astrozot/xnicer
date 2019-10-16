import warnings
import numpy as np
from scipy.special import ndtri, logsumexp
from sklearn.utils import check_array, check_random_state
from sklearn.mixture import GaussianMixture
# from extreme_deconvolution import extreme_deconvolution
from . import FIX_NONE, FIX_AMP, FIX_CLASS, FIX_MEAN, FIX_COVAR, FIX_ALL, \
    xdeconv, log_likelihoods


class XD_Mixture(GaussianMixture):
    """Extreme decomposition using a Gaussian Mixture Model

    This class allows to estimate the parameters of a Gaussian mixture
    distribution from data with noise.

    Parameters
    ----------
    n_components : int, tuple, list of int, or list of tuple, default=1. The
        number of mixture components. In case objects can be classified in
        multiple classes, it must be a tuple, indicating the number of
        components in each class. If a list is provided, the best value is
        found using the BIC (see `bic_test`)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to
        'full'. String describing the type of covariance parameters to use.
        Must be one of:

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

        Currently only 'full' is implemented.

    tol : float, defaults to 1e-5. The convergence threshold. EM iterations
        will stop when the lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6. Non-negative regularization added to
        the diagonal of covariance. Allows to assure that the covariance
        matrices are all positive.

    max_iter : int, defaults to 1e9. The maximum number of EM iterations to
        perform.

    regularization : float, defaults to 0.0 The regularization parameter used
        by the extreme deconvolution.

    n_init : int, defaults to 1. The number of initializations to perform. The
        best results are kept.

    init_params : {'gmm', 'kmeans', 'random'} or XD_Mixture, defaults to
        'gmm'. The method used to initialize the weights, the means and the
        precisions. Must be one of::

            'gmm'      : data are initialized from a quick GMM fit.
            'kmeans'   : responsibilities are initialized using kmeans.
            'random'   : responsibilities are initialized randomly.
            XD_Mixture : instance of a XD_Mixture already fitted.

    weights_init : array-like, shape (n_components, ), optional The
        user-provided initial weights, defaults to None. If it None, weights
        are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_x_features), optional The
        user-provided initial means, defaults to None, If it None, means are
        initialized using the `init_params` method.

    precisions_init : array-like, optional. The user-provided initial
        precisions (inverse of the covariance matrices), defaults to None. If
        it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                            if 'spherical',
            (n_x_features, n_x_features)               if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    splitnmerge : int, default to 0. The depth of the split and merge path
        (default = 0, i.e. no split and merge is performed).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False. If 'warm_start' is True, the solution
        of the last fitting is used as initialization for the next call of
        fit(). This can speed up convergence when fit is called several time
        on similar problems.

    verbose : int, default to 0. Enable verbose output. If 1 then it prints
        the current initialization and each iteration step. If greater than 1
        then it prints also the log probability and the time needed for each
        step.

    verbose_interval : int, default to 10. Number of iteration done before the
        next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,) The weights of each mixture
        components.

    means_ : array-like, shape (n_components, n_x_features) The mean of each
        mixture component.

    covariances_ : array-like The covariance of each mixture component. The
        shape depends on `covariance_type`::

            (n_components,)                            if 'spherical',
            (n_features, n_x_features)                 if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    precisions_ : array-like The precision matrices for each component in the
        mixture. A precision matrix is the inverse of a covariance matrix. A
        covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices
        makes it more efficient to compute the log-likelihood of new samples
        at test time. The shape depends on `covariance_type`::

            (n_components,)                            if 'spherical',
            (n_features, n_x_features)                 if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    precisions_cholesky_ : array-like The cholesky decomposition of the
        precision matrices of each mixture component. A precision matrix is
        the inverse of a covariance matrix. A covariance matrix is symmetric
        positive definite so the mixture of Gaussian can be equivalently
        parameterized by the precision matrices. Storing the precision
        matrices instead of the covariance matrices makes it more efficient to
        compute the log-likelihood of new samples at test time. The shape
        depends on `covariance_type`::

            (n_components,)                            if 'spherical',
            (n_features, n_x_features)                 if 'tied',
            (n_components, n_x_features)               if 'diag',
            (n_components, n_x_features, n_x_features) if 'full'

    converged_ : bool True when convergence was reached in fit(), False
        otherwise.

    lower_bound_ : float Log-likelihood of the best fit of EM.

    See Also
    --------
    GaussianMixture : Gaussian mixture model.
    """

    def __init__(self, n_components=1, n_classes=1, covariance_type='full', 
                 tol=1e-5, reg_covar=1e-06, max_iter=int(1e9), 
                 n_init=1, init_params='gmm', 
                 weights_init=None, means_init=None, precisions_init=None,
                 splitnmerge=0, random_state=None, warm_start=False,
                 regularization=0.0, verbose=0, verbose_interval=10):
        super(XD_Mixture, self).__init__(
            n_components=n_components, covariance_type=covariance_type, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.n_classes = n_classes
        self.splitnmerge = splitnmerge
        self.weights_ = self.classes_ = self.means_ = self.covariances_ = None
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
        if self.init_params == 'gmm' or isinstance(self.init_params, XD_Mixture):
            self.init_params = 'kmeans'
        super(XD_Mixture, self)._check_parameters(Y)
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
        Y = check_array(Y, dtype=[np.float64], order='C', estimator='XD_Mixture')
        n_samples, n_y_features = Y.shape
        if n_samples < self.n_components:
            raise ValueError('Expected n_samples >= n_components '
                             'but got n_components = %d, n_samples = %d'
                             % (self.n_components, Y.shape[0]))
        if len(Yerr.shape) not in [2, 3]:
            raise ValueError('Expected a 2d or 3d array for Yerr')
        Yerr = check_array(Yerr, dtype=[
                           np.float64], order='C', ensure_2d=False, allow_nd=True, estimator='XD_Mixture')
        if Yerr.shape[0] != n_samples:
            raise ValueError('Yerr must have the same number of samples as Y')
        if Yerr.shape[1] != n_y_features:
            raise ValueError('Yerr must have the same number of features as Y')
        if len(Yerr.shape) == 3 and Yerr.shape[1] != Yerr.shape[2]:
            raise ValueError(
                'Yerr must be of shape (n_samples, n_y_features, n_y_features)')
        if projection is not None:
            projection = check_array(projection, dtype=[np.float64], order='C', ensure_2d=False,
                                     allow_nd=True, estimator='XD_Mixture')
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
                                     allow_nd=True, estimator='XD_Mixture')
            if len(log_weight.shape) != 1:
                raise ValueError('log_weight must be a 1d array')
            if log_weight.shape[0] != n_samples:
                raise ValueError(
                    'log_weight must have the same number of samples as Y')
        return Y, Yerr, projection, log_weight

    def _initialize_parameters(self, Y, Yerr, random_state, projection=None, log_weight=None, log_class_prob=None):
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

        log_weight : array_like, shape (n_samples,)
            Optional log weights for the various points.
            
        log_class_prob : array_like, shape (n_samples, n_classes)
            Optional log probability for each point to belong to a given class.
        """
        if isinstance(self.init_params, XD_Mixture):
            parameters = list(p.copy()
                              for p in self.init_params._get_parameters())
            self._set_parameters(parameters)
        else:
            init_params = self.init_params
            if projection is not None:
                identity = np.zeros(shape=projection.shape[1:])
                j = range(min(identity.shape))
                identity[j, j] = 1
                mask = np.all(projection == identity, axis=(1, 2))
                n_x_features = projection.shape[2]
            else:
                mask = np.ones(self.n_samples_, dtype=np.bool)
                n_x_features = Y.shape[1]
            n_points = Y.shape[0]
            if log_weight is not None:
                mask &= np.log(np.random.rand(log_weight.shape[0])) < log_weight
            if init_params == 'gmm':
                init_params = 'kmeans'
            if log_class_prob is None:
                log_class_prob = np.zeros((n_points, self.n_classes))
            # OK, now everything is just the same: we have classes and class probabilities
            self.weights_ = np.empty(self.n_components)
            self.classes_ = np.empty((self.n_components, self.n_classes))
            self.means_ = np.empty((self.n_components, n_x_features))
            self.covariances_ = np.empty((self.n_components, n_x_features, n_x_features))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tmp_gmm = GaussianMixture(
                    self.n_components, max_iter=30, covariance_type='full',
                    init_params=init_params, random_state=self.random_state)
                if init_params != 'random':
                    if np.sum(mask) < self.n_components:
                        raise ValueError(
                            'Number of valid points smaller than number of components.')
                    tmp_gmm.fit(Y[mask])
                    resp = tmp_gmm.predict_proba(Y[mask])[:, :, np.newaxis] * \
                        np.exp(log_class_prob)[mask, np.newaxis, :]
                    xclass = np.sum(resp, axis=0)
                    xclass /= np.sum(xclass, axis=1)[:, np.newaxis] 
                else:
                    tmp_gmm._initialize_parameters(
                        Y[mask], tmp_gmm.random_state)
                    xclass = np.zeros((self.n_components, self.n_classes)) - \
                        np.log(self.n_classes)
                self.means_ = tmp_gmm.means_
                self.xclass_ = xclass
                self.weights_ = tmp_gmm.weights_ 
                self.covariances_ = tmp_gmm.covariances_
            # Renormalize the weights
            self.weights_ /= np.sum(self.weights_)
            self.n_features_ = n_x_features

    def fit(self, Y, Yerr, projection=None, log_weight=None, log_class_prob=None, fixpars=None):
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
            
        log_class_prob : array_lile, shape (n_samples, n_classes)
            The log probability for each point to belong to one of the classes. Only used
            if self.n_classes is larger than unity.

        fixpars : None, int, or int array_like, shape (self.n_components,)
            A combination of FIX_AMP, FIX_CLASS, FIX_MEAN, FIX_COVAR that 
            indicates the parameters to keep fixed for each component.  If a 
            scalar, the same value is used for all components.

        The best fit parameters are directly saved in the object.
        """
        # If n_components is a list, perform BIC optimization
        if isinstance(self.n_components, list):
            self.bic_test(Y, Yerr, self.n_components, projection=projection, log_weight=log_weight)
            return self
        
        if self.n_classes > 1:
            if log_class_prob is not None:
                if log_class_prob.shape[0] != Y.shape[0]:
                    raise ValueError(
                        "log_class_prob and Y do not have the same number of points")
                if log_class_prob.shape[1] != self.n_classes:
                    raise ValueError(
                        "log_class_prob has a number of probabilities different from n_classes")
        else:
            # Just in case it has been passed, ignore it!
            log_class_prob = None
        
        # Temporarily change some of the parameters to perform a first
        # initialization
        Y, Yerr, projection, log_weight = self._check_Y_Yerr(
            Y, Yerr, projection, log_weight)
        self._check_initial_parameters(Y)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not ((self.warm_start or fixpars is not None)
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
                                            projection=projection, log_weight=log_weight,
                                            log_class_prob=log_class_prob)
                self.lower_bound_ = -np.infty
            self.lower_bound_ = xdeconv(Y, Yerr, self.weights_, self.means_, self.covariances_,
                                        tol=self.tol, maxiter=self.max_iter,
                                        regular=self.reg_covar, # FIXME splitnmerge=self.splitnmerge,
                                        weight=log_weight, projection=projection,
                                        fixpars=fixpars, classes=log_class_prob)
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
        tmp = np.empty(Y.shape[0])
        result = np.empty((Y.shape[0], self.n_components))
        for c in range(self.n_components):
            log_likelihoods(delta[:, c, :], T[:, c, :, :], tmp)
            result[:, c] = tmp - (n_y_features * np.log(2.0*np.pi) / 2.0)
        return result

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
        if lowest_par:
            self.bic_ = optimal_bic
            self.set_params(**optimal_par)
            self._set_parameters(optimal_res)
        return bics, optimal_n_comp, optimal_bic


