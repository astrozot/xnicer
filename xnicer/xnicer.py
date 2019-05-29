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
from .utilities import log1mexp, cho_solve
from .catalogs import ExtinctionCatalogue


class XNicer(BaseEstimator):
    """Class XNicer, used to perform the XNicer extinction estimate.

    This class allows one to estimate the extinction from two color catalogues,
    a control field and a science field. It uses the extreme deconvolution
    provided by the XDCV class.

    Parameters
    ----------
    xdcv : XDCV
        An XDCV instance used to perform all necessary extreme deconvolutions.

    extinctions : array-like, shape (n_extinctions,)
        A 1D vector of extinctions used to perform a selection correction.

    extinction_vec : array-like, shape (n_bands,)
        The extinction vector, that is A_band / A_ref, for each band.

    log_weights_ : tuple of array-like, shape (n_extinctions, xdcv[class].n_components))
        The log of the weights of the extreme decomposition, for each class of
        objects, at each extintion value.
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
            # TODO: check how to do this in case of several classes!
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
