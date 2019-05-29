# Typical test usage for the XNICER pipeline.
# Created by Marco Lombardi on 13 May 2019.
# N.B. This example is based on the VISION data and is specific to this case;
# just change the code accordongly.

# The following lines make sure that NumPy does not perform any
# multithreading. This is very important for the extreme deconvolution
# library, which uses Open MP and would be slowed down significantly without
# these lines.
import os
os.environ['CC'] = 'gcc-8'
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import math
import numpy as np
del os.environ["MKL_NUM_THREADS"]
del os.environ["MKL_THREADING_LAYER"]
del os.environ["NUMEXPR_NUM_THREADS"]
del os.environ["OMP_NUM_THREADS"]

# Other imports can use multithreading if necessary. In particular, xwcs,
# which internally uses the extreme deconvolution, MUST be imported with
# multithreading enabled. Note that importing xwcs at the top would not work,
# since xwcs uses NumPy internally.
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from xnicer import *
from xnicer.catalogs import *
from xnicer.xdcv import *
from xnicer.xnicer import *

# Load the control (*_c) and the science (*_s) data. The following lines are
# based on the Vision dataset in Orion, and have hard-coded here my local
# setup. Change them accordingly to your data.
path = '/Users/mlombard/TeX/art64/Vision/'
cat_c = ascii.read(path + 'control.dat', readme=path + 'ReadMe')
cat_s = ascii.read(path + 'science.dat', readme=path + 'ReadMe')

# Create a photometric catalog for the control field data. For this catalogue,
phot_c = PhotometricCatalogue(cat_c, mags=['Jmag', 'Hmag', 'Ksmag'],
                              mag_errs=['e_Jmag', 'e_Hmag', 'e_Ksmag'])

# The following line add "probabilities" to each detection. Initially, the
# probabilities are all 1; however, as we simulate the effects of extinction,
# the probabilities of each object will change. This is useful to calibrate
# the XNicest algorithm (not required for XNicer).
phot_c.add_log_probs()

# Create a photometric and an astrometric catalogue for the science field.
phot_s = PhotometricCatalogue(cat_s, mags=['Jmag', 'Hmag', 'Ksmag'],
                              mag_errs=['e_Jmag', 'e_Hmag', 'e_Ksmag'])
coord_s = SkyCoord(cat_s['RAdeg'], cat_s['DEdeg'], unit='deg', frame='icrs')

# Extinction vector in the J, H, and K band
ext_vec = np.array([2.50, 1.55, 1.0])

# Setup the extreme deconvolution. We fix to 5 the number of components: this
# gives a good balance between the speed of the algorithm and the overall
# final accuracy. The last line can be slow, depending on the computer used.
xdcv = XDCV(n_components=5)
xnicer = XNicer(xdcv, np.linspace(0.0, 6.0, 5), ext_vec)
xnicer.fit(phot_c)

# The calibration below is used for the XNicest algorithm: it computes the
# XNicest weights and bias corrections.
xnicer.calibrate(phot_c, np.linspace(-1.0, 6.0, 29))

# We now use the trained XNicer object to predict the science field
# extinctions.
ext_s = xnicer.predict(phot_s, n_iters=3)

# As a test, we can also compute the extinctions in the control field and make
# sure they are around zero. Note that each object is weighted by its inverse
# variance: this is important to reduce both the noise and the bias. Note also
# that we are here using all objects, even if they do not have complete
# magnitude measurements (for example, one band is missing). Using only
# objects with complete measurements would improve the standard deviation.
ext_c = xnicer.predict(phot_c, n_iters=3)
weight_c = 1.0 / ext_c.variance_
bias_c = np.sum(ext_c.mean_ * weight_c) / np.sum(weight_c)
stdv_c = np.sqrt(np.sum(ext_c.mean_ ** 2 * weight_c ** 2) / \
    np.sum(weight_c)**2) * np.sqrt(len(weight_c))
print(f"Bias = {bias_c}")
print(f"Stdv = {stdv_c}")

# Debugging code
#import pickle
#f = open("test_xnicer.pkl", "bw")
#pickle.dump(coord_s, f)
#pickle.dump(ext_s, f)
#f.close()

#import pickle
#import kde
#f = open("test_xnicer.pkl", "rb")
#coord_s = pickle.load(f)
#ext_s = pickle.load(f)
#f.close()

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