"""Extreme decomposition and XNICER code.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13
"""

# Author: Marco Lombardi <marco.lombardi@gmail.com>

# See https://numpydoc.readthedocs.io/en/latest/format.html

from __future__ import print_function, division
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from astropy import table
from tqdm.auto import tqdm
from .xdeconv import FIX_MEAN, FIX_COVAR
from .xdeconv.em_step import predict_d  # pylint: disable=no-name-in-module
from .catalogs import PhotometricCatalogue, ColorCatalogue, ExtinctionCatalogue


class XNicer(BaseEstimator):
    """Class XNicer, used to perform the XNicer extinction estimate.

    This class allows one to estimate the extinction from two color catalogues,
    a control field and a science field. It uses the extreme deconvolution
    provided by the XDGaussianMixture class.

    Parameters
    ----------
    xdmix : XDGaussianMixture
        An XDGaussianMixture instance used to perform all necessary extreme
        deconvolutions.

    extinctions : array-like, shape (n_extinctions,)
        A 1D vector of extinctions used to perform a selection correction.

    log_weights_ : tuple of array-like, shape (n_extinctions, xdmix.n_components))
        The log of the weights of the extreme decomposition, for each class of
        objects, at each extintion value.

    calibration : tuple or None
        A tuple of arrays that saves a list of calibration parameters,
        computed through the `calibrate` method:

        - The extinctions for which the calibration has been computed
        - The biases associated to each extinction
        - The normalized sum of inverse variances

        Set to None if no calibration has been performed.

    """

    def __init__(self, xdmix, extinctions=None):
        """Constructor."""
        self.xdmix = xdmix
        if extinctions is None:
            extinctions = [0.0]
        self.extinctions = np.array(extinctions)
        self.log_weights_ = None
        self.log_classes_ = None
        self.calibration = None

    def fit(self, cat, update=False):
        """Initialize the class with control field data.

        Parameters
        ----------
        cat : PhotometricCatalogue
            The control field data, as a PhotometricCatalogue.

        update : bool
            If true, the first value of self.extinctions is skipped, which
            makes the whole procedure much faster.

        """
        # Basic check on the main parameter
        if not isinstance(cat, PhotometricCatalogue):
            raise TypeError(
                "Expecting a PhotoometricCatalogue as first argument")
        n_exts = len(self.extinctions)
        for n, extinction in enumerate(self.extinctions):
            # If update is set, skip the first iteration: this makes the
            # fitting procedure much faster
            if update and n == 0:
                continue
            # Add the extinction, as requested
            cat_ext = cat.extinguish(extinction, apply_completeness=True,
                                     update_errors=False)
            cat_ext['mags'] -= extinction * cat.meta['reddening_law']
            # Use double precision for the fit
            cols_ext = cat_ext.get_colors(use_projection=True,
                                          dtype=np.float64)
            if n == 0:
                self.xdmix.fit(cols_ext['cols'], cols_ext['col_covs'],
                               projection=cols_ext['projections'],
                               log_weight=cols_ext['log_probs'],
                               Yclass=cols_ext['log_class_probs'])
                # Only check the number of classes now
                use_classes = 'log_class_probs' in cat.colnames and \
                    self.xdmix.n_classes > 1
            else:
                self.xdmix.fit(cols_ext['cols'], cols_ext['col_covs'],
                               projection=cols_ext['projections'],
                               log_weight=cols_ext['log_probs'],
                               Yclass=cols_ext['log_class_probs'],
                               fixpars=FIX_MEAN | FIX_COVAR)
            if self.log_weights_ is None:
                # We could set this earlier in the __init__, but it does not
                # work in case the number of components for self.xdmix is an
                # array (this is possible if we request a BIC criterion)
                self.log_weights_ = np.zeros((n_exts, self.xdmix.n_components))
                if use_classes:
                    self.log_classes_ = np.zeros(
                        (n_exts, self.xdmix.n_components, self.xdmix.n_classes))
            with np.errstate(divide='ignore'):
                self.log_weights_[n] = np.log(self.xdmix.weights_)
                if use_classes:
                    self.log_classes_[n] = np.log(self.xdmix.classes_)

    def calibrate(self, cat, extinctions=None, progressbar=None,
                  apply_completeness=True, update_errors=True,
                  use_projection=True, **kw):
        """Perform a full calibration of the algorithm for a set of extinctions.

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

        extinctions : list of array of floats, default to self.extinctions
            The list of extinctions to use for the calibration. All
            extinctions should be non-negative. The first extinction must be 0.
            Can be different from self.extinction.

        apply_completeness : bool, default to True
            If True, the completeness function is taken into account, and
            random objects that are unlikely to be observable given the added
            extinction are dropped. This parameter is passed to `extinguish`.

        update_errors : bool, default to False
            If set, errors are also modified to reflect the fact that objects
            are now fainter.  This parameter is passed to `extinguish`.

        use_projection : bool, default to True
            If True, the color catalogue for the prediction is built using
            projections. One should set this parameter similarly to the way
            the analysis will be performed.

        kw : dictionary
            Additional keywords are directly passed to `predict`.

        progressbar : bool or None, default=None
            If True, show a progress bar using tqdm; if None (default), only
            show for TTY output.

        """
        # Basic check on the main parameter
        if not isinstance(cat, PhotometricCatalogue):
            raise TypeError(
                "Expecting a PhotoometricCatalogue as first argument")
        self.calibration = None
        if extinctions is None:
            extinctions = self.extinctions
        biases = []
        ivars = []
        if 'log_probs' not in cat.colnames:
            warnings.warn('For best results add log probabilities to cat')
        try:
            iterator = tqdm(extinctions, desc='calibration',
                            disable=progressbar, postfix={'bias': '---'})
            for extinction in iterator:
                cat_t = cat.extinguish(extinction,
                                       apply_completeness=apply_completeness,
                                       update_errors=update_errors)
                ext_t = self.predict(
                    cat_t.get_colors(use_projection=use_projection), **kw)
                objweight = np.exp(ext_t['log_weight']) / ext_t['variance_A']
                ivar = np.sum(objweight)
                mean = np.sum(ext_t['mean_A'] * objweight) / ivar
                biases.append(mean - extinction)
                ivars.append(ivar)
                iterator.set_postfix({'bias': mean - extinction})
        finally:
            iterator.close()
        self.calibration = (np.array(extinctions), np.array(biases),
                            np.array(ivars) / ivars[0])


    def predict(self, cols, use_classes=True, full=False, n_iters=3):
        """Compute the extinction for each object of a PhotometryCatalogue.

        Parameters
        ----------
        cols : ColorCatalogue
            A ColorCatalogue with the science field data.

        use_classes : bool, default=True
            If classes are available in the catalogue, and the deconvolution
            has been performed using classes, takes them into account for
            the extinction estimate. In this case, the final catalogue will
            also have a column called 'log_class_probs'.

        full : bool, default=False
            If True, the function returns complete extinction catalogue,
            including estimates of the intrinsic colors.

        n_iters : int, default=3
            Number of iterations to use during the fitting procedure. If
            n_iters=1, then no adjustment for the extinction selection
            effect is made.

        """
        # pylint: disable=invalid-name
        # Basic check on the main parameter
        if not isinstance(cols, ColorCatalogue):
            raise TypeError("Expecting a ColorCatalogue as first argument")
        # Check if we need to use classes
        if use_classes and 'log_class_probs' in cols.colnames and \
            self.xdmix.n_classes > 1:
            use_classes = True
        else:
            use_classes = False

        # Allocate the result: note that we use n_objs=len(cols), in case the
        # original catalogue has the log_probs columns. This would trigger
        # the generation of a different number of color objects. Note also
        # that, in this case, we need to keep track of the original objects and
        # of the associated probabilities.
        res = ExtinctionCatalogue()
        dtype = cols['cols'].dtype
        res.meta['n_components'] = self.xdmix.n_components
        res.add_column(table.Column(
            cols['idx'], name='idx',
            description='index with respect to the original catalogue',
            format='%d'))
        res.meta['n_colors'] = (cols.meta['n_bands']-1) if full else 0
        res.meta['n_classes'] = (self.xdmix.n_classes) if use_classes else 0

        # Warning: Old code start
        # Compute the extinction vector
        # color_ext_vec = cols.meta['reddening_law'].astype(dtype)
        # Compute all parameters (except the weights) for the 1st deconvolution
        #
        # V = self.xdmix.covariances_.astype(dtype)[np.newaxis, ...]
        # if not use_projection:
        #     mu = self.xdmix.means_.astype(dtype)[np.newaxis, :, :]
        #     PV = PVP = V.astype(dtype)
        # else:
        #     projections = cols['projections'].astype(dtype)
        #     P = projections[:, np.newaxis, :, :]
        #     PV = np.matmul(P, V)
        #     PVP = np.matmul(PV, np.moveaxis(P, -2, -1))
        #     mu = np.einsum('...ij,...j->...i', P,
        #                    self.xdmix.means_.astype(dtype)[np.newaxis, :, :])
        #     color_ext_vec = np.einsum('...ij,...j->...i',
        #                               projections,
        #                               color_ext_vec)[:, np.newaxis, :]
        # T = cols['col_covs'].astype(dtype)[:, np.newaxis, :, :] + PVP
        # Tc = np.linalg.cholesky(T)
        # d = cols['cols'].astype(dtype)[:, np.newaxis, :] - mu
        # T_k = cho_solve(Tc, color_ext_vec)
        # T_d = cho_solve(Tc, d)
        # Tlogdet = np.sum(np.log(np.diagonal(Tc, axis1=-2, axis2=-1)), axis=-1)
        # sigma_k2 = 1.0 / np.sum(T_k * T_k, axis=-1)
        # means_A = (sigma_k2 * np.sum(T_k * T_d, axis=-1)).astype(dtype)
        # variances_A = sigma_k2.astype(dtype)
        # C_k = np.sum(T_d * T_d, axis=-1) - means_A**2 / sigma_k2
        # log_weights0_ = - Tlogdet - \
        #     (cat.meta['n_bands'] - 1) * np.log(2.0*np.pi) / 2.0 - \
        #     C_k / 2.0 + np.log(2.0 * np.pi * sigma_k2) / 2.0
        # if full:
        #     T_V = cho_matrix_solve(Tc, PV)
        #     V_TT_k = np.einsum('...ji,...j->...i', T_V, T_k)
        #     covariances = (-V_TT_k * variances_A[:, :, np.newaxis]
        #         ).astype(dtype)
        #     variances_c = (V - np.einsum('...ij,...ik->...jk', T_V, T_V) -
        #         np.einsum('...i,...j->...ij', V_TT_k, covariances)).astype(dtype)
        #     means_c = (self.xdmix.means_[np.newaxis, :, :] +
        #         np.einsum('...ji,...j->...i', T_V, T_d) - \
        #             V_TT_k * means_A[:, :, np.newaxis]).astype(dtype)
        #
        # old code end

        nobjs = len(cols)
        k = self.xdmix.n_components
        ndim = cols.meta['n_bands'] - 1
        Amean = np.asfortranarray(np.zeros((nobjs, k)).T)
        Avar = np.asfortranarray(np.zeros((nobjs, k)).T)
        Aweight = np.asfortranarray(np.zeros((nobjs, k)).T)
        if full:
            Wmean = np.asfortranarray(np.zeros((nobjs, k, ndim)).T)
            Wcov = np.asfortranarray(np.zeros((nobjs, k, ndim)).T)
            Wvar = np.asfortranarray(np.zeros((nobjs, k, ndim, ndim)).T)
        else:
            Wmean = Wcov = Wvar = None
        predict_d(np.asfortranarray(cols['cols'].T, dtype=dtype),
                  np.asfortranarray(cols['col_covs'].T, dtype=dtype),
                  np.asfortranarray(self.xdmix.means_.T, dtype=dtype),
                  np.asfortranarray(self.xdmix.covariances_.T, dtype=dtype),
                  np.asfortranarray(cols.meta['reddening_law'].T, dtype=dtype),
                  Amean, Avar, Aweight,
                  np.asfortranarray(cols['projections'].T, dtype=dtype) if
                  cols.meta['use_projection'] else None,
                  Wmean, Wcov, Wvar)
        # We are now ready to compute the means, variances, and weights for
        # the GMM of the extinction. Each of these parameters has a shape
        # (n_objs, n_bands - 1). We can go on with the means and variances,
        # but the weights require more care, because they are linked to the
        # xdmix weights, and these in turn depend on the particular extinction
        # of the star. We therefore initially only compute log_weights0_, the
        # log of the weights of the extinction GMM that does not include any
        # term related to the xdmix weight.
        res.add_column(table.Column(Amean.T, copy=False, name='means_A',
            description='Array of means of extinction components',
            unit='mag', format='%6.3f'))
        res.add_column(table.Column(Avar.T, copy=False, name='variances_A',
            description='Array of variances of extinction components',
            unit='mag^2', format='%7.5f'))
        # We now compute, if requested, the intrinsic color quantities.
        if full:
            res.add_column(table.Column(Wcov.T, copy=False, name='covariances',
                description='Array of covariances of extinction-magnitude',
                unit='mag^2', format='%7.5f'))
            res.add_column(table.Column(Wvar.T, copy=False, name='variances_c',
                description='Array of variances of intrinsic colors',
                unit='mag^2', format='%7.5f'))
            res.add_column(table.Column(Wmean.T, copy=False, name='means_c',
                description='Array of means of intrinsic colors',
                unit='mag', format='%6.3f'))
        # Fine, now need to perform two loops: one outer loop where we iterate
        # n_iters time, and an internal loop that computes the weights for
        # each extinction step. To proceed we define a new array,
        # log_ext_weights, that saves the log of the contribution of a
        # particular extinction step for each object; we initially assume that
        # all objects only have a contribution from the first extinction step.
        log_weights0_ = Aweight.T
        log_ext_weights = np.full(
            (len(cols), len(self.extinctions)), -np.inf, dtype=dtype)
        log_ext_weights[:, 0] = 0.0
        for _ in range(n_iters):
            # If the classes are available, use them
            if use_classes:
                # The logsumexp's stuff has shape (n, e, k, c), where n=# objs,
                # e=# calibration extinctions, k=# clusters, and c=# classes
                # I am summing over all extinctions to use the correct
                # combination of values.
                tmp = log_weights0_[:, :, np.newaxis] + logsumexp(
                    self.log_weights_.astype(dtype)[np.newaxis, :, :, np.newaxis] +
                    self.log_classes_.astype(dtype)[np.newaxis, :, :, :] +
                    cols['log_class_probs'].astype(dtype)[:, np.newaxis, np.newaxis, :] +
                    log_ext_weights[:, :, np.newaxis, np.newaxis], axis=1)
                res['log_weights'] = logsumexp(tmp, axis=2)
                res['log_class_probs'] = logsumexp(tmp, axis=1)
                res['log_evidence'] = logsumexp(res['log_weights'], axis=-1)
                res['log_weights'] -= res['log_evidence'][:, np.newaxis]
                res['log_class_probs'] -= res['log_evidence'][:, np.newaxis]
            else:
                res['log_weights'] = log_weights0_ + logsumexp(
                    self.log_weights_.astype(dtype)[np.newaxis, :, :] +
                    log_ext_weights[:, :, np.newaxis], axis=1)
                res['log_evidence'] = logsumexp(res['log_weights'], axis=-1)
                res['log_weights'] -= res['log_evidence'][:, np.newaxis]
            # Now we need to update the weights for the extinction steps according
            # to each object's average extinction
            for e, extinction in enumerate(self.extinctions):
                log_ext_weights[:, e] = res.score_samples(
                    np.repeat(extinction, len(cols)), np.zeros(len(cols)))
            log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
        res.update_()
        res['mean_A'].description = 'Mean of the posterior extinction'
        res['mean_A'].unit = 'mag'
        res['mean_A'].format = '%6.3f'
        res['variance_A'].description = 'Variance of the posterior extinction'
        res['variance_A'].unit = 'mag^2'
        res['variance_A'].format = '%7.3f'
        res['log_weight'].description = 'Final log of the weight'
        res['log_weights'].description = 'log of the weight of each component'
        res['log_evidence'].description = 'log of the evidence'
        res['log_weight'].format = res['log_weights'].format = \
            res['log_evidence'].format = '%5g'
        if use_classes:
            res['log_class_probs'].description = \
                'log of the probability to belong to each class'
            res['log_class_probs'].format = '%g'

        # In case we have a calibrated object, perform the bias correction and
        # the XNicest estimates
        if self.calibration:
            n_objs = len(cols)
            log_ext_weights = np.empty((n_objs, len(self.calibration[0])),
                                       dtype=dtype)
            # Perform a first pass with no bias corrected
            for e, extinction in enumerate(self.calibration[0]):
                log_ext_weights[:, e] = res.score_samples(
                    np.repeat(extinction, n_objs), np.zeros(n_objs))
            log_ext_weights -= logsumexp(log_ext_weights,
                                         axis=-1)[..., np.newaxis]
            ext_weights = np.exp(log_ext_weights)
            # Now do the bias correction, based on the measured extinctions
            bias = np.sum(ext_weights * self.calibration[1], axis=1)
            res['mean_A'] -= bias
            res['means_A'] -= bias[:, np.newaxis]
            # Recompute the weights
            # FIXME: is this really necessary?
            for e, extinction in enumerate(self.calibration[0]):
                log_ext_weights[:, e] = res.score_samples(
                    np.repeat(extinction, n_objs), np.zeros(n_objs))
            log_ext_weights -= logsumexp(log_ext_weights,
                                         axis=-1)[..., np.newaxis]
            ext_weights = np.exp(log_ext_weights)
            # Finally perform the XNicest evaluations
            res.add_column(table.Column(
                np.sum(ext_weights / self.calibration[2], axis=1),
                name='xnicest_weight',
                description='Object\'s weight to use with the XNicest alogorithm',
                format='%7.5f'))
            res.add_column(table.Column(
                np.sum(ext_weights * self.calibration[0]
                       / self.calibration[2], axis=1) /
                res['xnicest_weight'] - np.sum(
                    ext_weights * self.calibration[0], axis=1),
                name='xnicest_bias',
                description='Bias to subtract with the XNicest alogorithm',
                format='%7.5f'))
        # Now, in case of use of log_probs, we need to correct the weights so
        # to include the original color weights.
        if 'log_probs' in cols.colnames:
            res['log_weights'] += cols['log_probs'][:, np.newaxis]
            res['log_weight'] += cols['log_probs']

        return res
