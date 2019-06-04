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
xnicer = XNicer(xdcv, [0.0], ext_vec)
xnicer.fit(phot_c)

# The calibration below is used for the XNicest algorithm: it computes the
# XNicest weights and bias corrections.
xnicer.calibrate(phot_c, np.linspace(-1.0, 6.0, 29))

# We now use the trained XNicer object to predict the science field
# extinctions.
ext_s = xnicer.predict(phot_s, n_iters=1, full=True)

# As a test, we can also compute the extinctions in the control field and make
# sure they are around zero. Note that each object is weighted by its inverse
# variance: this is important to reduce both the noise and the bias. Note also
# that we are here using all objects, even if they do not have complete
# magnitude measurements (for example, one band is missing). Using only
# objects with complete measurements would improve the standard deviation.
phot_c.mag_errs[1, 0] = 100.0
phot_c.mag_errs[2, 1] = 100.0
phot_c.mag_errs[3, 2] = 100.0
ext_c = xnicer.predict(phot_c, n_iters=1, full=True)
weight_c = 1.0 / ext_c.variance_A
bias_c = np.sum(ext_c.mean_A * weight_c) / np.sum(weight_c)
stdv_c = np.sqrt(np.sum(ext_c.mean_A ** 2 * weight_c ** 2) / \
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

phot_c.mag_errs[1, 0] = 100.0
phot_c.mag_errs[2, 1] = 100.0
phot_c.mag_errs[3, 2] = 100.0
c = phot_c.get_colors(use_projection=False)
c2 = phot_c.get_colors(use_projection=True)
k = xnicer.extinction_vec[:-1] - xnicer.extinction_vec[1:]
n_objs = len(c)
n_components = xnicer.xdcv.n_components
P = np.zeros((n_objs, n_components, 3, 3))
E = np.linalg.inv(c.col_covs)
V = np.linalg.inv(xnicer.xdcv.covariances_)
Ek = np.dot(k, E)
P[:, :, 0, 0] = np.dot(Ek, k)[:, np.newaxis]
P[:, :, 0, 1:] = Ek[:, np.newaxis, :]
P[:, :, 1:, 0] = Ek[:, np.newaxis, :]
P[:, :, 1:, 1:] = E[:, np.newaxis, :, :] + V[np.newaxis, :, :, :]
m = np.zeros((n_objs, n_components, 3))
m[:, :, 0] = np.einsum('...i,...i->...', Ek, c.cols)[:, np.newaxis]
m[:, :, 1:] = np.einsum('...ij,...j->...i', E, c.cols)[:, np.newaxis, :] + \
    np.einsum('...ij,...j->...i', V, xnicer.xdcv.means_)[np.newaxis,:,:]
obj = 1
np.linalg.solve(P, m)[obj]
ext_c.means_A[obj], ext_c.means_c[obj]
np.linalg.inv(P)[obj, :, 0, 0]
ext_c.variances_A[obj]
np.linalg.inv(P)[obj, :, 0, 1:]
ext_c.covariances[obj]
np.linalg.inv(P)[obj, 3, 1:, 1:]
ext_c.variances_c[obj, 3]

V = xnicer.xdcv.covariances_[np.newaxis, ...]
cols = c
cols2 = c2
color_ext_vec = k
T = cols.col_covs[:, np.newaxis, :, :] + V
Tc = np.linalg.cholesky(T)
d = cols.cols[:, np.newaxis, :] - \
    xnicer.xdcv.means_[np.newaxis, :, :]
T_k = cho_solve(Tc, color_ext_vec)

V2 = np.einsum('...ij,...jk,...lk->...il',
            cols2.projections[:, np.newaxis, :, :],
            xnicer.xdcv.covariances_[np.newaxis, :, :, :],
            cols2.projections[:, np.newaxis,:,:])
T2 = cols2.col_covs[:, np.newaxis, :, :] + V2
Tc2 = np.linalg.cholesky(T2)
d2 = cols2.cols[:, np.newaxis, :] - \
    np.einsum('...ij,...j->...i', cols2.projections[:, np.newaxis, :, :],
            xnicer.xdcv.means_[np.newaxis, :, :])
T_k2 = cho_solve(Tc2, np.einsum('...ij,...j->...i',
                            cols2.projections,
                            color_ext_vec)[:, np.newaxis, :])

T_d = cho_solve(Tc, d)
T_d2 = cho_solve(Tc2, d2)
Tlogdet = np.sum(np.log(np.diagonal(Tc, axis1=-2, axis2=-1)), axis=-1)
Tlogdet2 = np.sum(np.log(np.diagonal(Tc2, axis1=-2, axis2=-1)), axis=-1)
sigma_k2 = 1.0 / np.sum(T_k * T_k, axis=-1)
sigma_k22 = 1.0 / np.sum(T_k2 * T_k2, axis=-1)

means_A = sigma_k2 * np.sum(T_k * T_d, axis=-1)
variances_A = sigma_k2
means_A2 = sigma_k22 * np.sum(T_k2 * T_d2, axis=-1)
variances_A2 = sigma_k22

T_V = cho_matrix_solve(Tc, V)
V_W_k = np.einsum('...ji,...j->...i', T_V, T_k)
covariances = -V_W_k * variances_A[:,:, np.newaxis]
V3 = np.einsum('...ij,...jk->...ik',
               cols2.projections[:, np.newaxis, :, :],
               xnicer.xdcv.covariances_[np.newaxis, :, :, :])
T_V2 = cho_matrix_solve(Tc2, V3)
V_W_k2 = np.einsum('...ji,...j->...i', T_V2, T_k2)
covariances2 = -V_W_k2 * variances_A2[:,:, np.newaxis]
covariances[obj, :, :]
covariances2[obj, :, :]

variances_c = V - np.einsum('...ij,...ik->...jk', T_V, T_V) - \
    np.einsum('...i,...j->...ij', V_W_k, covariances)
means_c = xnicer.xdcv.means_[np.newaxis,:,:] + \
    np.einsum('...ji,...j->...i', T_V, T_d) - V_W_k * means_A[:,:, np.newaxis]
variances_c2 = V - np.einsum('...ij,...ik->...jk', T_V2, T_V2) - \
    np.einsum('...i,...j->...ij', V_W_k2, covariances2)
means_c2 = xnicer.xdcv.means_[np.newaxis, :, :] + \
    np.einsum('...ji,...j->...i', T_V2, T_d2) - V_W_k2 * means_A2[:, :, np.newaxis]

np.linalg.inv(P)[obj, :, 0, 0]
variances_A[obj,:]
variances_A2[obj, :]
ext_c.variances_A[obj]

np.linalg.inv(P)[obj, :, 1:, 0]
covariances[obj,:,:]
covariances2[obj,:,:]
ext_c.covariances[obj]

np.linalg.inv(P)[obj, 3, 1:, 1:]
variances_c[obj, 3, :, :]
variances_c2[obj, 3, :, :]
ext_c.variances_c[obj, 3]

np.linalg.solve(P, m)[obj, :, 0]
means_A[obj,:]
means_A2[obj, :]
ext_c.means_A[obj]

np.linalg.solve(P, m)[obj, :, 1:]
means_c[obj,:,:]
means_c2[obj,:,:]
ext_c.means_c[obj]

comp = 2
obj = 5
col = cols.cols[3]
# col = xnicer.xdcv.means_[comp]
A = 0.3
E = np.linalg.inv(cols.col_covs[obj])
V = np.linalg.inv(xnicer.xdcv.covariances_[comp])
d1 = cols.cols[obj] - col - A * k
d2 = col - xnicer.xdcv.means_[comp]
res1 = np.exp(-0.5 * np.dot(np.dot(E, d1), d1)) * np.sqrt(np.linalg.det(E / (2 * np.pi))) * \
    np.exp(-0.5 * np.dot(np.dot(V, d2), d2)) * np.sqrt(np.linalg.det(V / (2 * np.pi))) * \
    xnicer.xdcv.weights_[comp]

M = P[obj, comp]
mu = np.linalg.solve(M, m[obj, comp])
x = np.array([A, col[0], col[1]]) - mu
res2 = np.exp(-0.5 * np.dot(np.dot(M, x), x)) * np.sqrt(np.linalg.det(M / (2 * np.pi))) * \
    np.exp(ext_c.log_weights_[obj, comp] + ext_c.log_evidence[obj])

res1, res2, res1 / res2

c2 = phot_c.get_colors(use_projection=True)
k = xnicer.extinction_vec[1:] - xnicer.extinction_vec[:-1]
n_objs = len(c)
n_components = xnicer.xdcv.n_components





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
