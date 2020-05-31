"""Mapping module, used to create registered extinction maps.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13
"""

import datetime
import numpy as np
import astropy.wcs
from astropy.io import fits

def guess_wcs(coords, projection='TAN', border=10, target_density=5.0,
              pixel_size=None):
    """Perform an initial guess of a WCS given a catalogue.

    Parameters
    ----------
    coords : astropy.coordinates.SkyCoord
        A SkyCoord object, with a list of objects in a given astronomical
        coordinate system. The coordinate system of the coordinates is kept
        in the WCS.

    projection : str, optional, default to 'TAN'
        The mnemonic for the projection to be used in the WCS for both
        coordinates.

    border : int, optional, default to 10
        The number of pixels in a frame around the objects to include in the
        final mapping.

    target_density : float, default to 5.0
        The approximate value of the final mean density requested. Depending
        on this value the final image will have different scales and sizes.

    pixel_size : float, default to None
        If not None, the exact value of the pixel size in arcmin; if None,
        the pixel size will be computed using the target_density.

    Returns
    -------
    wcs : astropy.wcs.WCS
        The proposed WCS structure to use with the provided catalogue.

    (naxis1, naxis2) : (int, int)
        The proposed size of the final image along the two axes.

    """
    goodscales = np.array([15*60, 10*60, 8*60, 5*60, 3*60, 2*60, 1.5*60, 60,
                           40, 30, 20, 15, 10, 8, 5, 3, 2, 1.5, 1,
                           0.75, 0.5, 1/3, 0.25, 10/60, 8/60, 5/60, 3/60,
                           2/60, 1.5/60, 1/60])
    if coords.name in ('fk4', 'fk5', 'icrs', 'cirs', 'gcrs', 'hcrs'):
        ctypes = ('RA--', 'DEC-')
    elif coords.name in ('galactic', 'galacticlsr'):
        ctypes = ('GLON', 'GLAT')
    elif coords.name == 'supergalactic':
        ctypes = ('SLON', 'SLAT')
    elif coords.name == 'geocentrictrueecliptic':
        ctypes = ('ELON', 'ELAT')
    elif coords.name == 'heliocentrictrueecliptic':
        ctypes = ('HLON', 'HLAT')
    names = list(coords.frame.representation_component_names.keys())
    lon_min = np.min(getattr(coords, names[0]).deg)
    lon_max = np.max(getattr(coords, names[0]).deg)
    lat_min = np.min(getattr(coords, names[1]).deg)
    lat_max = np.max(getattr(coords, names[1]).deg)
    if lon_min > lon_max:
        lon_min -= 360
    crval1 = (lon_min + lon_max) * 0.5
    if crval1 < 0:
        crval1 += 360
    crval2 = (lat_min + lat_max) * 0.5
    c = np.cos(crval2 * np.pi / 180.0)
    aspect = (lon_max - lon_min) / (c * (lat_max - lat_min))
    area = (lon_max - lon_min) * (lat_max - lat_min) * c
    density = len(coords) / area
    scale = np.sqrt(target_density / density) * 60
    if pixel_size is None:
        best_scale = goodscales[np.argmin(np.abs(scale - goodscales))]
    else:
        best_scale = float(pixel_size)
    naxis1 = np.ceil(
        (np.floor(np.sqrt(area * aspect) / best_scale * 66) + 20) / 10) * 10
    naxis2 = np.ceil(
        (np.floor(np.sqrt(area / aspect) / best_scale * 66) + 20) / 10) * 10

    # Build the preliminary WCS
    w = astropy.wcs.WCS(naxis=2)
    w.wcs.crpix = [naxis1 / 2, naxis2 / 2]
    w.wcs.ctype = [f"{ctypes[0]}-{projection}", f"{ctypes[1]}-{projection}"]
    w.wcs.crval = [np.round(crval1 * 1e6) / 1e6, np.round(crval2 * 1e6) / 1e6]
    w.wcs.cdelt = [-best_scale / 60, best_scale / 60]
    if coords.equinox:
        w.wcs.equinox = np.round(coords.equinox.decimalyear)

    # Now we improve the WCS by making sure that all points will stay entirely
    # within our image
    xy = w.all_world2pix(
        getattr(coords, names[0]).deg, getattr(coords, names[1]).deg, 0)
    min_x = np.min(xy[0])
    max_x = np.max(xy[0])
    min_y = np.min(xy[1])
    max_y = np.max(xy[1])
    w.wcs.crpix[0] -= min_x
    w.wcs.crpix[1] -= min_y
    w.wcs.crpix[0] = np.round(w.wcs.crpix[0]) + border
    w.wcs.crpix[1] = np.round(w.wcs.crpix[1]) + border
    naxis1 = int(np.ceil(max_x - min_x) + 2 * border)
    naxis2 = int(np.ceil(max_y - min_y) + 2 * border)
    w.pixel_shape = (naxis1, naxis2)
    return w


def make_maps(coords, ext, wcs, kde, n_iters=3, tolerance=3.0,
              use_xnicest=True):
    """Map making algorithm.

    Parameters
    ----------
    coords : SkyCoord
        The coordinates of all objects. Not all objects might have measured
        extinctions, so the size of this catalogue is generally larger than
        the one of `ext`.

    ext : ExtinctionCatalogue
        An extinction catalogue for the valid objects. Note that we use the
        `selection` attribute of the catalogue to select only valid objects in
        the coords.

    wcs : WCS
        A Worlk Coordinate System structure, returned for example by
        `guess_wcs`.

    kde : KDE
        A KDE object, the kernel density estimator that will be used to build
        smooth maps.

    n_iters : int, default to 3
        The number of iterations performed. At the end of each iteration,
        unusual objects are clipped. Clipping does not occur if n_iters = 1.

    tolerance : float, default to 3.0
        The maximum tolerance allowed in the extinction of each object.
        Objects that have an extinction that differs the local extinction by
        more than `tolerance` times the local standard deviation will be
        excluded from the analysis. This requires several passes or
        iterations, as dictated by the `n_iters` argument.

    use_xnicest : bool, default to True
        If False, the XNicest algorithm will not be used, not even if the
        `ext` catalogue has XNicest weights computed.

    Returns
    -------
    fits.HDU
        A full HDU fits structure, which can be used directly to save a FITS
        file. Alternatively, one can use the `data` attribute of the result to
        obtain directly the various maps created. The `data` attribute is a 3D
        array: each "plane" contains different results:

        data[0]
            The XNicer extinction map, in units of [mag].

        data[1]
            The inverse variance on the XNicer extinction map [mag^-2]. One
            can obtains the error of the extinction map by computing
            data[1]**(-0.5)

        data[2]
            The sum of weights of the extinction map [mag^-2 pix^-1]. This is
            mostly provided for sanity checks: pixels with a small value in
            data[2] indicate places where few objects are observed, or places
            where objects have large extinction measurement errors.

        data[3]
            The local density of objects, in units of [objects pix^-1].

        data[4]
            The XNicest extinction map [mag]. Only returned if the XNicest
            algorithm has been used.

        data[5]
            The inverse variance on the XNicest extinction map [mag^-2]. Only
            returned if the XNicest algorithm has been used.

        data[6]
            The sum of weights of the XNIcest extinction map [mag^-2 pix^-1].
            Only returned if the XNicest algorithm has been used.

    """
    names = list(coords.frame.representation_component_names.keys())
    if 'idx' in ext.colnames:
        xy = wcs.all_world2pix(
            getattr(coords, names[0]).deg[ext['idx']],
            getattr(coords, names[1]).deg[ext['idx']], 0)
    else:
        xy = wcs.all_world2pix(
            getattr(coords, names[0]).deg,
            getattr(coords, names[1]).deg, 0)
    xy = np.stack((xy[1], xy[0]), axis=-1)
    mask = kde.mask_inside(xy)
    n_objs = xy.shape[0]
    # Check if we need to use the XNicest algorithm
    xnicest = bool(use_xnicest) and ('xnicest_bias' in ext.colnames)
    # Weights & power arrays: 0-cmap, 1-cvar, 2-cwgt, 3-dmap
    # Moreover, if xnicest: 4-amap, 5-avar, 6-awgt
    if xnicest:
        weights = np.empty((7, n_objs))
        power = np.zeros(7)
    else:
        weights = np.empty((4, n_objs))
        power = np.zeros(4)
    ext_ivar = 1.0 / ext['variance_A']
    weights[0, :] = ext['mean_A'] * ext_ivar
    weights[1, :] = ext['mean_A'] * ext['mean_A'] * ext_ivar
    weights[2, :] = ext_ivar
    weights[3, :] = 1
    if xnicest:
        weights[4, :] = (ext['mean_A'] - ext['xnicest_bias']) * ext_ivar * \
            ext['xnicest_weight']
        # FIXME: The following line, used to compute the XNicest error,
        # ignores the error on the xnicest_weight. That is, the final error
        # will be computed by assuming only the error on the extinction. The
        # issue is not so easy to solve (one would need to take into account
        # both the variance on the XNicest weight and the covariance with the
        # extinction). For a further release...
        weights[5, :] = ext_ivar * ext['xnicest_weight']
        power[5] = 1
        weights[6, :] = ext_ivar * ext['xnicest_weight']
    # Shifted (unframed) coordinates, to the nearest int
    x_ = np.rint(xy[:, 1] + kde.kernel_size).astype(np.int)
    y_ = np.rint(xy[:, 0] + kde.kernel_size).astype(np.int)
    for iteration in range(n_iters):
        if iteration < n_iters - 1:
            # Faster steps for intermediate iterations
            res = kde.kde(xy, weights[0:3],
                          power=power[0:3], mask=mask, nocut=True)
        else:
            # Last iteration is complete; also, change the weights and power
            # for the map #1, so that it measures the expected variance,
            # instead of the observed one.
            weights[1] = weights[2]
            power[1] = 1
            res = kde.kde(xy, weights,
                          power=power, mask=mask, nocut=True)
        res[0] = np.divide(res[0], res[2], out=np.zeros_like(res[0]),
                           where=(res[2] != 0))
        if iteration < n_iters - 1:
            # Intermediate iterations: cvar is the observed scatter and has to
            # be computed as a sampled variance
            res[1] = np.divide(res[1], res[2], out=np.zeros_like(res[1]),
                               where=(res[2] != 0))
            res[1] -= res[0]**2
            # Intermediate iterations: update the mask
            x_mask = x_[mask]
            y_mask = y_[mask]
            clip = (res[0, y_mask, x_mask] - ext['mean_A'][mask])** 2 \
                > tolerance**2 * res[1, y_mask, x_mask]
            mask[np.where(mask)[0][clip]] = False
        else:
            # Last iteration: cvar is the ensemble variance
            res[1] = np.divide(res[2]*res[2], res[1], out=np.zeros_like(res[1]),
                               where=(res[1] != 0))
            if xnicest:
                # Last iteration: compute the other intermediate maps
                res[4] = np.divide(res[4], res[6], out=np.zeros_like(res[4]),
                                   where=(res[4] != 0))
                res[5] = np.divide(res[6]*res[6], res[5], out=np.zeros_like(res[5]),
                                   where=(res[5] != 0))
    # Cut the data
    res = res[:, kde.kernel_size:-kde.kernel_size,
              kde.kernel_size:-kde.kernel_size]
    # Prepare the header
    hdu = fits.PrimaryHDU(res, header=wcs.to_header())
    hdu.header['PLANE1'] = ('extinction', '[mag]')
    hdu.header['PLANE2'] = ('inverse variance', '[mag^-2]')
    hdu.header['PLANE3'] = ('weight', '[mag^-2 pix^-1]')
    hdu.header['PLANE4'] = ('density', '[pix^-1], objects per pixel')
    if xnicest:
        hdu.header['PLANE5'] = ('extinction', '[mag], XNicest method')
        hdu.header['PLANE6'] = ('inverse variance', '[mag^-2], XNicest method')
        hdu.header['PLANE7'] = ('weight', '[mag^2 pix^-1], XNicest method')
    hdu.header['CREATOR'] = 'XNicer v0.2.0'
    hdu.header['DATE'] = datetime.datetime.now().isoformat()
    hdu.header['AUTHOR'] = 'Marco Lombardi'
    hdu.add_checksum()
    # All done: return the final maps, truncated
    return hdu
