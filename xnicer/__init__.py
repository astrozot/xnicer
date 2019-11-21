"""The XNICER package.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13"""

from .xnicer import XNicer
from .xdeconv import XDGaussianMixture, FIX_NONE, FIX_AMP, FIX_MEAN, FIX_COVAR, FIX_ALL
from .catalogs import PhotometricCatalogue, ColorCatalogue, ExtinctionCatalogue
from .kde import KDE
from .mapping import guess_wcs, make_maps
