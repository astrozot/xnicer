"""Extreme decomposition and XNICER code.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13"""

# Author: Marco Lombardi <marco.lombardi@gmail.com>

# See https://numpydoc.readthedocs.io/en/latest/format.html

from __future__ import print_function, division
import numpy as np
import warnings
import copy
from scipy.special import ndtri, logsumexp
from sklearn.base import BaseEstimator
from .xdeconv import XD_Mixture, FIX_MEAN, FIX_COVAR
from .utilities import log1mexp, cho_solve, cho_matrix_solve
from .catalogs import ExtinctionCatalogue


class XNicer(BaseEstimator):
    """Class XNicer, used to perform the XNicer extinction estimate.

    This class allows one to estimate the extinction from two color catalogues,
    a control field and a science field. It uses the extreme deconvolution
    provided by the XD_Mixture class.

    Parameters
    ----------
    xdmix : XD_Mixture
        An XD_Mixture instance used to perform all necessary extreme deconvolutions.

    extinctions : array-like, shape (n_extinctions,)
        A 1D vector of extinctions used to perform a selection correction.

    log_weights_ : tuple of array-like, shape (n_extinctions, xdmix.sum_components))
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
        self.xdmix = xdmix
        if extinctions is None:
            extinctions = [0.0]
        self.extinctions = np.array(extinctions)
        self.log_weights_ = None
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
        for n, extinction in enumerate(self.extinctions):
            # If update is set, skip the first iteration: this makes the fitting
            # procedure much faster
            if update and n == 0:
                continue
            # Add the extinction, as requested
            cat_A = cat.extinguish(extinction, apply_completeness=True,
                                   update_errors=False)
            cat_A['mags'] -= extinction * cat.meta['reddening_law']
            cols_A = cat_A.get_colors(use_projection=True)
            if n == 0:
                self.xdmix.fit(cols_A['cols'].data, cols_A['col_covs'], 
                               cols_A['projections'], 
                               log_weight=cols_A['log_probs'], 
                               log_class_prob=cols_A['log_class_probs'])
            else:
                self.xdmix.fit(cols_A['cols'], cols_A['col_covs'], 
                               cols_A['projections'],
                               log_weight=cols_A['log_probs'], 
                               log_class_prob=cols_A['log_class_probs'],
                               fixpars=FIX_MEAN | FIX_COVAR)
            if self.log_weights_ is None:
                # We could set this earlier in the __init__, but it does not
                # work in case the numbero of components for self.xdmix is an
                # array (this is possible if we request a BIC criterion)
                self.log_weights_ = np.zeros(
                    (len(self.extinctions), self.xdmix.sum_components))
            with np.errstate(divide='ignore'):
                self.log_weights_[n] = np.log(self.xdmix.weights_)


    def calibrate(self, cat, extinctions=None, **kw):
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

        extinctions : list of array of floats, default to self.extinctions
            The list of extinctions to use for the calibration. All
            extinctions should be non-negative. The first extinction must be 0.
            Can be different from self.extinction.

        kw : dictionary
            Additional keywords are directly passed to `predict`.
        """
        self.calibration = None
        if extinctions is None:
            extinctions = self.extinctions
        biases = []
        ivars = []
        if 'log_probs' not in cat.colnames:
            warnings.warn('For best results add log probabilities to cat')
        for extinction in extinctions:
            cat_t = cat.extinguish(extinction)
            ext_t = self.predict(cat_t, **kw)
            objweight = np.exp(ext_t['log_weight']) / ext_t['variance_A']
            ivar = np.sum(objweight)
            mean = np.sum(ext_t['mean_A'] * objweight) / ivar
            biases.append(mean - extinction)
            ivars.append(ivar)
        self.calibration = (np.array(extinctions), np.array(biases),
                            np.array(ivars) / ivars[0])
            
    def predict(self, cat, use_projection=True, full=False, n_iters=3):
        """Compute the extinction for each object of a PhotometryCatalogue

        Parameters
        ----------
        cat : PhotometryCatalogue
            A PhotometryCatalogue with the science field data.

        use_projection : bool, default=True
            Parameter passed directly to `PhotometricCatalogue`.

        full : bool, default=False
            If True, the function returns complete extinction catalogue,
            including estimates of the intrinsic colors.

        n_iters : int, default=3
            Number of iterations to use during the fitting procedure. If
            n_iters=1, then no adjustment for the extinction selection
            effect is made.
        """
        # Compute the colors
        cols = cat.get_colors(use_projection=use_projection)
        
        # Check if we need to use classes
        if 'log_class_probs' in cols.colnames and \
            isinstance(self.xdmix.n_components, tuple):
            use_classes = True
            # Distribute equally the class probabilities among the class members
            full_class_prob = np.empty((len(cols), self.xdmix.sum_components))
            cum_c = 0
            for e, c in enumerate(self.xdmix.n_components):
                full_class_prob[:, cum_c:cum_c +
                                c] = (cols['log_class_probs'][:, e] - np.log(c))[:, np.newaxis]
                cum_c += c
        else:
            use_classes = False

        # Allocate the result: note that we use n_objs=len(cols), in case the
        # original catalogue has the log_probs columns. This would trigger
        # the generation of a different number of colors. Note also that, in
        # this case, we need to keep track of the original objects and of the
        # associated probabilities.
        res = ExtinctionCatalogue()
        res.meta['n_components'] = self.xdmix.sum_components,
        res['idx'] = cols['idx']
        res.meta['n_colors'] = (cat.meta['n_bands']-1) if full else 0

        # Compute the extinction vector
        color_ext_vec = cols.meta['reddening_law']

        # Compute all parameters (except the weights) for the 1st deconvolution
        V = self.xdmix.covariances_[np.newaxis, ...]
        if not use_projection:
            mu = self.xdmix.means_[np.newaxis, :, :]
            PV = PVP = V
        else:
            P = cols['projections'][:, np.newaxis, :, :]
            PV = np.einsum('...ij,...jk->...ik', P, V)
            PVP = np.einsum('...ik, ...lk ->...il', PV, P)
            mu = np.einsum('...ij,...j->...i', P,
                           self.xdmix.means_[np.newaxis,:,:])
            color_ext_vec = np.einsum('...ij,...j->...i',
                                      cols['projections'],
                                      color_ext_vec)[:, np.newaxis, :]
        T = cols['col_covs'][:, np.newaxis, :, :] + PVP
        Tc = np.linalg.cholesky(T)
        d = cols['cols'][:, np.newaxis, :] - mu
        T_k = cho_solve(Tc, color_ext_vec)
        T_d = cho_solve(Tc, d)
        Tlogdet = np.sum(np.log(np.diagonal(Tc, axis1=-2, axis2=-1)), axis=-1)
        sigma_k2 = 1.0 / np.sum(T_k * T_k, axis=-1)
        # We are now ready to compute the means, variances, and weights for the
        # GMM of the extinction. Each of these parameters has a shape
        # (n_objs, n_bands - 1). We can go on with the means and variances, but
        # the weights require more care, because they are linked to the xdmix
        # weights, and these in turn depend on the particular extinction of the
        # star. We therefore initially only compute log_weights0_, the log of
        # the weights of the extinction GMM that does not include any term related
        # to the xdmix weight.
        res['means_A'] = sigma_k2 * np.sum(T_k * T_d, axis=-1)
        # variances_ = np.sum(T_d * T_d, axis=-1) - self.ext*self.ext / sigma_k2
        res['variances_A'] = sigma_k2
        C_k = np.sum(T_d * T_d, axis=-1) - res['means_A']**2 / sigma_k2
        log_weights0_ = - Tlogdet - \
            (cat.meta['n_bands'] - 1) * np.log(2.0*np.pi) / 2.0 - \
            C_k / 2.0 + np.log(2.0 * np.pi * sigma_k2) / 2.0
        # We now compute, if requested, the intrinsic color quantities.
        if full:
            T_V = cho_matrix_solve(Tc, PV)
            V_TT_k = np.einsum('...ji,...j->...i', T_V, T_k)
            res['covariances'] = -V_TT_k * res['variances_A'][:, :, np.newaxis]
            res['variances_c'] = V - np.einsum('...ij,...ik->...jk', T_V, T_V) - \
                np.einsum('...i,...j->...ij', V_TT_k, res['covariances'])
            res['means_c'] = self.xdmix.means_[np.newaxis, :, :] + \
                np.einsum('...ji,...j->...i', T_V, T_d) - \
                V_TT_k * res['means_A'][:,:, np.newaxis]
        # Fine, now need to perform two loops: one outer loop where we iterate
        # n_iters time, and an internal loop that computes the weights for each
        # extinction step. To proceed we define a new array, log_ext_weights,
        # that saves the log of the contribution of a particular extinction step
        # for each object; we initially assume that all objects only have a
        # contribution from the first extinction step.
        log_ext_weights = np.full(
            (len(cols), len(self.extinctions)), -np.inf)
        log_ext_weights[:, 0] = 0.0
        for _ in range(n_iters):
            res['log_weights'] = log_weights0_ + \
                logsumexp(self.log_weights_[
                            np.newaxis, :, :] + log_ext_weights[:, :, np.newaxis], axis=1)
            # If the classes are available, use them
            if use_classes:
                res['log_weights'] += full_class_prob
            res['log_evidence'] = logsumexp(res['log_weights'], axis=-1)
            res['log_weights'] -= res['log_evidence'][..., np.newaxis]
            # Now we need to update the weights for the extinction steps according
            # to each object's average extinction
            for e, extinction in enumerate(self.extinctions):
                log_ext_weights[:, e] = res.score_samples(
                    np.repeat(extinction, len(cols)), np.zeros(len(cols)))
            log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
        res.update_()

        # In case we have a calibrated object, perform the bias correction and
        # the XNicest estimates
        if self.calibration:
            n_objs = len(cols)
            log_ext_weights = np.empty((n_objs, len(self.calibration[0])))
            # Perform a first pass with no bias corrected
            for e, extinction in enumerate(self.calibration[0]):
                log_ext_weights[:, e] = res.score_samples(
                        np.repeat(extinction, n_objs), np.zeros(n_objs))
            log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
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
            log_ext_weights -= logsumexp(log_ext_weights, axis=-1)[..., np.newaxis]
            ext_weights = np.exp(log_ext_weights)
            # Finally perform the XNicest evaluations
            res['xnicest_weight'] = np.sum(
                ext_weights / self.calibration[2], axis=1)
            res['xnicest_bias'] = np.sum(ext_weights * self.calibration[0]
                / self.calibration[2], axis=1) / res['xnicest_weight'] - np.sum(
                    ext_weights * self.calibration[0], axis=1)
        # Now, in case of use of log_probs, we need to correct the weights so
        # to include the original color weights.
        if 'log_probs' in cols.colnames:
            res['log_weights'] += cols['log_probs'][:, np.newaxis]
            res['log_weight'] += cols['log_probs']

        return res
