"""The XNICER package.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13"""

from .xnicer import XNicer
from .catalogs import PhotometricCatalogue, ColorCatalogue, ExtinctionCatalogue
from .kde import KDE
from .mapping import guess_wcs, make_maps

