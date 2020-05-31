import numpy as np
from astropy.io import ascii
from astropy import table
from astropy.coordinates import SkyCoord
from xnicer import XNicer, XDGaussianMixture, guess_wcs, make_maps
from xnicer.catalogs import AstrometricCatalogue, PhotometricCatalogue, ColorCatalogue


def test_xnicer():
    path = '/Users/mlombard/TeX/art64/Vision/'
    cat_c = ascii.read(path + 'control.dat', readme=path + 'ReadMe')

    reddening_law = [2.50, 1.55, 1.00]
    mags = ["Jmag", "Hmag", "Ksmag"]
    mag_errs = ["e_Jmag", "e_Hmag", "e_Ksmag"]
    phot_c = PhotometricCatalogue.from_table(
        cat_c, mags, mag_errs, reddening_law=reddening_law,
        class_names=["star", "galaxy"], class_prob_names=["ClassSex"],
        log_class_probs=False)
    phot_c.add_log_probs()

    xd = XDGaussianMixture(n_components=5, n_classes=2)
    xnicer = XNicer(xd, [0.0])
    xnicer.fit(phot_c)
