"""Extreme deconvolution code.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13
"""

# pylint: disable=no-name-in-module
from .em_step import FIX_NONE, FIX_AMP, FIX_CLASS, FIX_MEAN, FIX_COVAR, \
    FIX_ALL, em_step_s, em_step_d, log_likelihoods_s, log_likelihoods_d
from .xdeconv import xdeconv, scores, splitnmerge_rank
from .xd_mixture import XDGaussianMixture
