# Typical test usage for the XNICER pipeline.
# Created by Marco Lombardi on 13 May 2019.
# N.B. This example is based on the VISION data and is specific to this case;
# just change the code accordongly.

import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from xnicer import XNicer, XD_Mixture, guess_wcs, make_maps
from xnicer.catalogs import PhotometricCatalogue

from astropy.io import votable
vo = votable.parse('/Users/mlombard/Downloads/vizier_votable.vot')
table = vo.get_first_table()
xc = PhotometricCatalogue(table)
field = table.fields[4]
votable.ucd.parse_ucd(field.ucd)


# Load the control (*_c) and the science (*_s) data. The following lines are
# based on the Vision dataset in Orion, and have hard-coded here my local
# setup. Change them accordingly to your data.
path = '/Users/mlombard/TeX/art64/Vision/'
cat_c = ascii.read(path + 'control.dat', readme=path + 'ReadMe')
cat_s = ascii.read(path + 'science.dat', readme=path + 'ReadMe')

# Reddening law in J, H, and Ks bands
reddening_law = [2.50, 1.55, 1.00]

# Create a photometric catalog for the control field data. For this catalogue,
mags = ["Jmag", "Hmag", "Ksmag"]
mag_errs = ["e_Jmag", "e_Hmag", "e_Ksmag"]
phot_c = PhotometricCatalogue(cat_c, mags=mags, mag_errs=mag_errs, 
                              reddening_law=reddening_law, 
                              class_names=["star", "galaxy"], class_probs=["ClassSex"],
                              log_class_probs=False)

# The following line add "probabilities" to each detection. Initially, the
# probabilities are all 1; however, as we simulate the effects of extinction,
# the probabilities of each object will change. This is useful to calibrate
# the XNicest algorithm (not required for XNicer).
phot_c.add_log_probs()

# Create a photometric and an astrometric catalogue for the science field.
phot_s = PhotometricCatalogue(cat_s, mags=mags, mag_errs=mag_errs, 
                              reddening_law=reddening_law, 
                              class_names=["star", "galaxy"], class_probs=["ClassSex"],
                              log_class_probs=False)
coord_s = SkyCoord(cat_s['RAdeg'], cat_s['DEdeg'], unit='deg', frame='icrs')

# Setup the extreme deconvolution. We fix to 5 the number of components: this
# gives a good balance between the speed of the algorithm and the overall
# final accuracy. The last line can be slow, depending on the computer used.
xd = XD_Mixture(n_components=5)
xnicer = XNicer(xd, np.linspace(0.0, 6.0, 5))
xnicer.fit(phot_c)

# The calibration below is used for the XNicest algorithm: it computes the
# XNicest weights and bias corrections. That line can also be slow.
xnicer.calibrate(phot_c, np.linspace(-1.0, 6.0, 29))

# We now use the trained XNicer object to predict the science field
# extinctions.
ext_s = xnicer.predict(phot_s, n_iters=1, full=True)

# >>> TEST: reddening law
def logprob(k1):
    phot_c.reddening_law[1] = k1
    xnicer.fit(phot_c, update=True)
    xnicer.calibrate(phot_c, np.linspace(-1.0, 6.0, 29))
    ext_s = xnicer.predict(phot_s, n_iters=3)
    log_evidence = np.sum(ext_s.log_evidence)
    print(k1, log_evidence)
    print(xnicer.xdcv.weights_)
    return log_evidence
xs = np.linspace(1.0, 2.0, 11)
ys = np.zeros_like(xs)
for n in range(len(xs)):
    ys[n] = logprob(xs[n])

from matplotlib import pyplot as plt
plt.plot(xs, ys*xs); plt.show()
# <<< TEST

# As a test, we can also compute the extinctions in the control field and make
# sure they are around zero. Note that each object is weighted by its inverse
# variance: this is important to reduce both the noise and the bias. Note also
# that we are here using all objects, even if they do not have complete
# magnitude measurements (for example, one band is missing). Using only
# objects with complete measurements would improve the standard deviation.
ext_c = xnicer.predict(phot_c)
weight_c = 1.0 / ext_c.variance_A
bias_c = np.sum(ext_c.mean_A * weight_c) / np.sum(weight_c)
stdv_c = np.sqrt(np.sum(ext_c.mean_A ** 2 * weight_c ** 2) / \
    np.sum(weight_c)**2) * np.sqrt(len(weight_c))
print(f"Bias = {bias_c}")
print(f"Stdv = {stdv_c}")

# Now we have all extinctions. We need therefore to build an extinction map.
# For this we need to define first a KDE smoother.
wcs = guess_wcs(coord_s.galactic, target_density=5.0)
smoother = KDE(tuple(reversed(wcs.pixel_shape)), max_power=2)
hdu = make_maps(coord_s.galactic, ext_s, wcs, smoother)
maps = hdu.data

# We can finally display the results
from matplotlib import pyplot as plt
plt.subplot(projection=wcs)
plt.imshow(maps[5], origin='lower')
plt.grid(color='white', ls='solid')
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')
plt.colorbar(orientation='horizontal')
plt.show()

# Optionally, we can save the final results in a FITS file
from astropy.io import fits
hdul = fits.HDUList([hdu])
hdul.writeto('XNicest.fits')
