"""Catalogue handling code.

:Author: Marco Lombardi
:Version: 0.1.0 of 2019/05/13"""

# Author: Marco Lombardi <marco.lombardi@gmail.com>

import collections
import warnings
import numpy as np
import copy
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp
from astropy.io import votable
from astropy.coordinates import SkyCoord
from .utilities import log1mexp


# Table 3 from Rieke & Lebovsky (1985), ApJ 288, 618: ratio A_lambda / A_V.
rieke_lebovsky_ucd = {
    'em.opt.U': 1.531,
    'em.opt.B': 1.324,
    'em.opt.V': 1.000,
    'em.opt.R': 0.748,
    'em.opt.I': 0.482,
    'em.IR.J': 0.282,
    'em.IR.H': 0.175,
    'em.IR.K': 0.112,
    'em.IR.3-4um': 0.058,
    'em.IR.4-8um': 0.023
}
rieke_lebovsky_names = {
    'U': 1.531,
    'B': 1.324,
    'V': 1.000,
    'R': 0.748,
    'I': 0.482,
    'J': 0.282,
    'H': 0.175,
    'K': 0.112,
    'KS': 0.112,
    'L': 0.058,
    'M': 0.023
}


def _find_reddening_vector(name):
    """Tries to find the reddening for a given band name."""
    name = name.upper()
    if name in rieke_lebovsky_names:
        return rieke_lebovsky_names[name]
    elif name[-4:] == '_MAG' and name[:-4] in rieke_lebovsky_names:
        return rieke_lebovsky_names[name[:-4]]
    elif name[-3:] == 'MAG' and name[:-3] in rieke_lebovsky_names:
        return rieke_lebovsky_names[name[:-3]]
    else:
        return False
    

def AstrometricCatalogue(table, frame=None, **kwargs):
    """Extract coordinates from a VOTable into a SkyCoord object

    Parameters
    ----------
    table : VOTable
        A VOTable that include astrometric coordinates.
        
    frame : Coordinate Frame
        Should be a known coordinate frame, such as 'fk4', 'fk5', 
        'galactic', 'icrs'... If unspecified, the frame is inferred
        from the kind of coordinate. In future we should parse the
        frame from the VO table directly, using `vo.iter_coosys`.

    **kwargs : dictionary
        Additional keyword arguments passed to SkyCoord.

    Returns
    -------
    SkyCoord
        A SkyCoord object with the coordinates of the input table
    """
    frames = {
        'eq': {'ra': 0, 'dec': 1},
        'galactic': {'lon': 0, 'lat': 1},
        'gal': {'lon': 0, 'lat': 1},
    }
    result = {}
    for field in table.fields:
        ucd = votable.ucd.parse_ucd(field.ucd)
        force = len(ucd) > 1 and (ucd[1][1] == 'meta.main')
        words = ucd[0][1].split('.')
        if words[0] == 'pos':
            ucd_frame = words[1]
            ucd_coord = words[2]
            if ucd_frame in frames and ucd_coord in frames[ucd_frame]:
                i = frames[ucd_frame][ucd_coord]
                if ucd_frame not in result:
                    result[ucd_frame] = [None, None]
                if force or result[ucd_frame][i] == None:
                    result[ucd_frame][i] = field                        
    if 'eq' in result:
        if frame is None:
            frame = 'icrs'
        result = result['eq']
    elif 'galactic' in result:
        frame = 'galactic'
        result = result['galactic']
    elif 'gal' in result:
        frame = 'galactic'
        result = result['gal']
    else:
        raise KeyError('No known coordinate system found in the table')
    return SkyCoord(table.array[result[0].ID],
                    table.array[result[1].ID], 
                    frame=frame, unit=result[0].unit,
                    equinox=result[0].ref, **kwargs)


class PhotometricCatalogue(object):
    """Initialize a new photometric catalogue.

    The initialization can be carried using either a Table or arrays of
    magnitudes and associated errors.

    Parameters
    ----------
    cat : Table or VOTable, optional
        If specified, must be a table containing the magnitude and
        associated errors.

    mags : array_like, shape (n_objs, n_bands) or list of strings
        If cat is not specified, an array with the measured magnitudes of all
        objects; otherwise, a list of strings indicating which columns of cat
        should be interpreted as magnitudes.

    mag_errs : array_like, shape (n_objs, n_bands) or list of strings
        If cat is not specified, an array with the measured magnitude errors
        of all objects; otherwise, a list of strings indicating which columns
        of cat should be interpreted as magnitude errors.

    probs : array_like, shape (n_objs, n_bands) or None
        The (log) probability to observe a given magnitude for a given 
        object. Used to simulate the effects of extinction through 
        `extinguish`. If None, the probabilities are ignored (that is, all
        objects are assumed to be observable with 100% probability).
        
    log_probs : bool, default=False
        If True, `probs` are really log probabilities; otherwise, are
        just probabilities.
        
    class_names : tuple or list of strings or None
        The name of the various discrete classes of objects the catalogue
        describes. If None, all objects are taken to be part of the same
        class.

    class_probs : array_like, shape (n_objs, n_classes) or None
        If the catalogue contains classified objects, an array that stores,
        for each object, the (log) probability that each objecs belongs to a
        given class. The probabilities should be normalized to unity (that
        is, the sum for a given object of all probabilities must be 1). 
        Alternatively, one can provide an array of shape (n_objs, 
        n_classes-1): in this case, it is assumed that these probabilities
        are associated to the first n_classes-1 classes; the probability for
        the last class is inferred from the normalization condition.
        
    log_class_probs : bool, default=False
        If True, `class_probs` are really log probabilities.

    reddening_law : array_like, shape (n_bands,) or None
        The reddening law associated with the bands in the catalogue. If not
        provided and the catalogue is a VO table, the reddening law is 
        inferred from the ucd or names of the bands using the Rieke & Lebovsky
        (1985) data.

    use_projection : bool, default to True
        What to do in case of missing band measurements. If true, the
        catalogue will be built using different combinations of magnitudes and
        the associated projection matrix will be returned when computing
        colors.

    max_err : float, default to 1.0
        Maximum admissible error above which a datum is discarted.

    min_bands : int, default to 2
        The minimum number of bands with valid measuremens to include the
        object in the final catalogue.

    null_mag: float, default to 15.0
        The value used to mark a null magnitude.

    null_err: float, default to 100.0
        The value used to mark a null magnitude error.

    Attributes
    ----------
    n_objs : int
        The final number of objects in the catalogue, after all filtering.

    n_bands : int
        The maximum number of bands used; note that some objects might have
        less bands available.

    selection : array_like, shape (n_objs,)
        Indices in the original catalogue of objects that have been used.

    mags : array_like, shape (n_objs, n_bands)
        Array with the extracted magnitudes.

    mag_errs : array_like, shape (n_objs, n_bands, n_bands)
        Array with the extracted magnitude errors.
    
    log_probs : array_like, shape (n_objs, n_bands) or None
        The log probability to observe a given magnitude for a given 
        object. Used to simulate the effects of extinction through 
        `extinguish`. If None, the probabilities are ignored.
        
    class_names : tuple or list of strings or None
        The name of the various discrete classes of objects the catalogue
        describes. If None, all objects are taken to be part of the same
        class.

    log_class_probs : array_like, shape (n_objs, n_classes) or None
        If the catalogue contains classified objects, an array that stores,
        for each object, the logarithms of the probability that each object
        belongs to a given class. The probabilities, if provided, should be
        normalized to unity along axis 1 (the second axis).
        
    nc_pars: array_like, shape (n_bands, 3) or None
        Array that reports, for each band, the best-fit number-count
        parameters, written as (exponential slope, 50% completeness limit,
        completeness width).

    null_mag: float
        The value used to mark a null magnitude.

    null_err: float
        The value used to mark a null magnitude error.
    """

    def __init__(self, cat=None, mags=None, mag_errs=None, probs=None,
                 log_probs=False, class_names=None, class_probs=None, 
                 log_class_probs=False, reddening_law=None,
                 max_err=1.0, min_bands=2, null_mag=15.0, null_err=100.0):
        # First of all the easy stuff
        self.max_err = max_err
        self.min_bands = min_bands
        self.nc_pars = None
        self.null_mag = null_mag
        self.null_err = null_err
        self.reddening_law = reddening_law.copy() if reddening_law else []
        if class_names is not None:
            self.class_names = tuple(class_names)

        # Check if the input is a VOTable: in case performs the required conversions
        if cat is not None:
            if isinstance(cat, votable.tree.Table):
                if mags is None:
                    bands = collections.OrderedDict()
                    for field in cat.fields:
                        ucd = votable.ucd.parse_ucd(field.ucd)
                        if ucd[0][1] == 'phot.mag':
                            # it's a magnitude, get its name
                            mag = ucd[1][1]
                            if mag in bands:
                                if bands[mag][0] is None:
                                    bands[mag][0] = field.ID
                            else:
                                bands[mag] = [field.ID, None]
                        if ucd[0][1] == 'stat.error' and ucd[1][1] == 'phot.mag':
                            # it's a magnitude error, get its name
                            mag = ucd[2][1]
                            if mag in bands:
                                if bands[mag][1] is None:
                                    bands[mag][1] = field.ID
                            else:
                                bands[mag] = [None, field.ID]
                    # Now write the mags and mag_errs arrays        
                    mags = []
                    mag_errs = []
                    for ucd, (mag, mag_err) in bands.items():
                        if mag is not None and mag_err is not None:
                            mags.append(mag)
                            mag_errs.append(mag_err)
                            if reddening_law is None:
                                if ucd in rieke_lebovsky_ucd:
                                    self.reddening_law.append(rieke_lebovsky_ucd[ucd])
                                else:
                                    reddening = _find_reddening_vector(mag)
                                    if reddening:
                                        self.reddening_law.append(reddening)
                                    else:
                                        warnings.warn(f"Cannot automatically find the reddening law for {mag}")
                                        self.reddening_law = None
                                        reddening_law = True
                # Regardless of what we have done above, extract the arrray of the table
                cat = cat.array
            # Check if we need to extract the reddening law
            if reddening_law is None:
                for mag in mags:
                    reddening = _find_reddening_vector(mag)
                    if reddening:
                        self.reddening_law.append(reddening)
                    else:
                        warnings.warn(f"Cannot automatically find the reddening law for {mag}")
                        self.reddening_law = None
                        break
            if self.reddening_law:
                self.reddening_law = np.array(self.reddening_law)


        # Now deal with the empty constructor case
        if mags is None:
            self.n_objs = self.n_bands = 0
            self.n_bands = 0
            self.selection = self.mags = self.mag_errs = self.log_probs = None
            return

        # OK, we got some input, let us use it
        if cat is not None:
            n_objs = len(cat)
            n_bands = len(mags)
            if n_bands != len(mag_errs):
                raise ValueError(
                    "Magnitudes and errors must have the same number of bands")
            self.mag_names = mags
            self.err_names = mag_errs
            if probs is not None:
                if n_bands != len(probs):
                    raise ValueError(
                        "Magnitudes and log-probabilites must have the same number of bands")
                prob_names = probs
                probs = np.empty((n_objs, n_bands))
            mags = np.empty((n_objs, n_bands))
            mag_errs = np.empty((n_objs, n_bands))
            for n in range(n_bands):
                mag_col = cat[self.mag_names[n]]
                err_col = cat[self.err_names[n]]
                mags[:, n] = mag_col
                mag_errs[:, n] = err_col
                if isinstance(mag_col, np.ma.MaskedArray):
                    w = np.where((~np.isfinite(mag_col)) | (~np.isfinite(err_col)) |
                                 mag_col.mask | err_col.mask)
                else:
                    w = np.where((~np.isfinite(mag_col)) |
                                 (~np.isfinite(err_col)))
                mags[w, n] = null_mag
                mag_errs[w, n] = null_err
                if probs is not None:
                    probs[:, n] = cat[prob_names[n]]
                    probs[w, n] = -np.inf if log_probs else 0.0 
            if class_probs is not None:
                if len(self.class_names) != len(class_probs) and \
                        len(self.class_names) != len(class_probs) + 1:
                    raise ValueError(
                        "Class names and log of class probabilities must have the same number of objects")
                class_prob_names = class_probs
                class_probs = np.empty((n_objs, len(self.class_names)))
                for n in range(len(class_prob_names)):
                    class_probs[:, n] = cat[class_prob_names[n]]
                if len(class_probs) != len(self.class_names):
                    if log_class_probs:
                        with np.errstate(divide='ignore'):
                            class_probs[:, -1] = np.log(1.0 - \
                                np.sum(np.exp(class_probs[:, :-1]), axis=1))
                    else:
                        class_probs[:, -1] = 1.0 - \
                            np.sum(class_probs[:, :-1], axis=1)
        else:
            if mags.ndim != 2 or mag_errs.ndir != 2:
                raise ValueError(
                    "Magnitudes and errors must be two-dimensional arrays")
            n_objs, n_bands = mags.shape
            if mag_errs.shape[0] != n_objs:
                raise ValueError(
                    "Magnitudes and errors must have the same number of objects")
            if mag_errs.shape[1] != n_bands:
                raise ValueError(
                    "Magnitudes and errors must have the same number of bands")
            if probs is not None:
                if probs.ndim != 2:
                    raise ValueError(
                        "Log-probabities must be a two-dimensional array")
                if n_objs != probs.shape[0]:
                    raise ValueError(
                        "Log-probabilites must have the right number of objects")
                if n_bands != probs.shape[1]:
                    raise ValueError(
                        "Log-probabilites must have the right number of bands")
            if class_probs is not None:
                if class_probs.ndim != 2:
                    raise ValueError(
                        "Log-probabities of classes must be a two-dimensional array")
                if n_objs != class_probs.shape[0]:
                    raise ValueError(
                        "Log-probabilites of classes must have the right number of objects")
                if len(class_names) == class_probs.shape[1] + 1:
                    class_probs = np.hstack((class_probs, np.zeros((n_objs, 1))))
                    if log_class_probs:
                        class_probs[:, -1] = np.log(1.0 - \
                            np.sum(np.exp(class_probs[:, :-1]), axis=1))
                    else:
                        class_probs[:, -1] = 1.0 - \
                            np.sum(class_probs[:, :-1], axis=1)
                if len(class_names) != class_probs.shape[1]:
                    raise ValueError(
                        "Log-probabilites of classes must have the same number of classes")
            if n_objs < n_bands:
                raise ValueError(
                    "Expecting a n_objs x n_bands array for mags and errs")
            if reddening_law is None:
                warnings.warn("Reddening law not specified")
                self.reddening_law = None
            else:
                if len(self.reddening_law) != n_bands:
                    raise ValueError("The length of the reddening_law vector does not match the number of bands")
                self.reddening_law = np.array(self.reddening_law)
        w = np.where(np.sum(mag_errs < max_err, axis=1) >= min_bands)[0]
        mags = mags[w, :]
        mag_errs = mag_errs[w, :]
        # Saves the results
        self.selection = w
        self.n_objs = len(w)
        self.n_bands = n_bands
        self.mags = mags
        self.mag_errs = mag_errs
        if probs is not None:
            if log_probs:
                self.log_probs = probs
            else:
                with np.errstate(divide='ignore'):
                    self.log_probs = np.log(probs)
        else:
            self.log_probs = None
        if class_probs is not None:
            if log_class_probs:
                self.log_class_probs = class_probs
            else:
                with np.errstate(divide='ignore'):
                    self.log_class_probs = np.log(class_probs)
        else:
            self.log_class_probs = None

    def __len__(self):
        return self.n_objs

    def __getitem__(self, sliced):
        res = copy.deepcopy(self)
        res.selection = res.selection[sliced]
        res.mags = res.mags[sliced]
        res.mag_errs = res.mag_errs[sliced]
        if res.log_probs is not None:
            res.log_probs = res.log_probs[sliced]
        if res.log_class_probs is not None:
            res.log_class_probs = res.log_class_probs[sliced]
        try:
            res.n_objs = len(res.selection)
        except TypeError:
            res.n_objs = 1
        return res

    def add_log_probs(self):
        """Add (log) probabilities to the photometric catalogue.

        Magnitude measurements with magnitude errors larger than max_err are
        marked with 0 probability.
        """
        if self.log_probs is None:
            self.log_probs = np.zeros((self.n_objs, self.n_bands))
        self.log_probs[np.where(self.mag_errs > self.max_err)] = -np.inf

    def remove_log_probs(self):
        """Removes log probabilities from the photometric catalogue.
        
        This method also applies decimation to the data: that is, it "cleans"
        magnitudes bands depending on the value of the log probabilities.
        """
        if self.log_probs is None:
            return self
        removed = self.log_probs < np.log(
            np.random.uniform(size=(self.n_objs, self.n_bands)))
        self.mags[removed] = self.null_mag
        self.mag_errs[removed] = self.null_err
        self.log_probs = None

    def get_colors(self, use_projection=True, band=None, map_mags=lambda _: _,
                   extinctions=None, tolerance=1e-5):
        """Compute the colors associate to the current catalogue.

        This operation is performed by subtracting two consecutive bands. For this reason,
        it is advisable to sort the band from the bluest to the reddest.

        Arguments
        ---------
        use_projection : bool, default to True
            If True, the procedure sorts bands so that missed bands are
            excluded from the color computation. A projection matrix will be
            returned.

        band : int or None, default to None
            If not None, include in the output a column with a magnitude. This
            is useful in a number of cases. A negative integer is interpreted
            as in the index operator [].

        mag_mags : function, default to identity
            A function used to map the magnitude, used only for the band
            selected (and thus only if band != None). Must accept an array of
            shape (n_objs,) and return an array of shape (n_objs,).

        extinctions : array-like, shape (n_objs,), or None, default to None
            If not None and if band is not None, it is an array of values that
            will be *subtracted* to the magnitudes of the band before applying
            map_mags. Used to correcte for estinguished magnitudes in the
            xnicer code.

        tolerance : float, default to 1e-5
            The minimum probability allowed: combinations of colors that have
            a smaller probability will be deleted from the final catalogue.
            Only used if the catalogue has log_probs associated to it.

        Return value
        ------------
        A ColorCatalogue. Note that, in case the catalogue has log_probs sets
        (and non-vanishing), the result will have a different number of
        objects with respect to the original catalogue, because it will
        contain all possible color combinations.
        """
        # If we have to work with probabilities, we will make a new catalogue
        # containing all possible combinations of magnitudes. We will
        # associate to each combination a probability that this is realized,
        # and we will call this function again on the all combinations.
        if self.log_probs is not None:
            cat = copy.deepcopy(self)
            lp_min = np.log(tolerance)
            lp_max = np.log(1 - tolerance)
            log_probs = np.zeros(cat.n_objs)
            for b in range(cat.n_bands):
                # Bands with too small probabilities are directly set to the
                # null value
                w = np.where(cat.log_probs[:, b] < lp_min)
                cat.mags[w, b] = cat.null_mag
                cat.mag_errs[w, b] = cat.null_err
                # Other bands with intermediate probabilities (and valid band
                # measurements) are duplicated; bands with high probabilities
                # are not duplicated because we assume that the band will
                # always be observed.
                w = np.where((cat.log_probs[:, b] > lp_min) & (
                    cat.log_probs[:, b] < lp_max))[0]
                cat.selection = np.concatenate(
                    (cat.selection, cat.selection[w]))
                cat.mags = np.concatenate((cat.mags, cat.mags[w]))
                cat.mags[cat.n_objs:, b] = cat.null_mag
                cat.mag_errs = np.concatenate((cat.mag_errs, cat.mag_errs[w]))
                cat.mag_errs[cat.n_objs:, b] = cat.null_err
                cat.log_probs = np.concatenate(
                    (cat.log_probs, cat.log_probs[w]))
                log_probs = np.concatenate(
                    (log_probs, log_probs[w] + log1mexp(cat.log_probs[w, b])))
                log_probs[w] = log_probs[w] + cat.log_probs[w, b]
                cat.n_objs += len(w)
            cat.log_probs = None
            # We now filter all objects with too small probabilities or not
            # enough valid vands
            w = np.where((log_probs > lp_min) & (np.sum(cat.mag_errs < cat.max_err,
                                                        axis=1) >= cat.min_bands))[0]
            cat = cat[w]

            res = cat.get_colors(use_projection=use_projection, band=band, map_mags=map_mags,
                                 extinctions=extinctions)
            res.log_probs = log_probs[w]
            return res

        # Computes the colors
        n_objs = self.n_objs
        if band is None:
            n_cols = self.n_bands - 1
        else:
            if band < 0:
                band = self.n_bands + band
            n_cols = self.n_bands
        cols = np.zeros((n_objs, n_cols))
        col_covs = np.zeros((n_objs, n_cols, n_cols))
        for c in range(self.n_bands - 1):
            cols[:, c] = self.mags[:, c] - self.mags[:, c+1]
            col_covs[:, c, c] = self.mag_errs[:, c]**2 + \
                self.mag_errs[:, c+1]**2
            if c > 0:
                col_covs[:, c, c-1] = col_covs[:, c-1, c] = - \
                    self.mag_errs[:, c]**2
        if band is not None:
            mags = self.mags[:, band]
            mag_errs = self.mag_errs[:, band]
            if extinctions is not None:
                mags -= extinctions
            cols[:, n_cols - 1] = map_mags(mags)
            diff_map = (map_mags(mags + mag_errs) -
                        map_mags(mags - mag_errs)) / \
                       (2*mag_errs)
            col_covs[:, n_cols - 1, n_cols -
                     1] = (diff_map*mag_errs)**2
            if band < self.n_bands - 1:
                col_covs[:, n_cols-1, band] = col_covs[:, band, n_cols-1] = \
                    diff_map*mag_errs**2
            if band > 0:
                col_covs[:, n_cols-1, band-1] = col_covs[:, band-1, n_cols-1] = \
                    -diff_map*mag_errs**2

        # Use projections
        if use_projection:
            # Projection matrix
            projections = np.zeros((n_objs, n_cols, n_cols))
            # Check which magnitudes are usable
            mask = self.mag_errs < self.max_err
            csum = np.cumsum(mask, axis=1) - 1
            line = csum.flat
            # Create the other two indices for the projections matrix
            obj, col = np.mgrid[0:n_objs, 0:self.n_bands]
            obj = obj.flat
            col = col.flat
            # Set the projections matrix
            w = np.where((line >= 0) & (col < self.n_bands - 1))
            projections[obj[w], line[w], col[w]] = 1
            # Remove the last row of each object if the last band is not
            # available for that object
            last_row = csum[:, -1]
            w = np.where((last_row >= 0) & (last_row < self.n_bands - 1))
            projections[w, last_row[w], :] = 0
            # Add the band row if we need to
            if band is not None:
                w = np.where((last_row >= 0) & (mask[:, band]))
                projections[w, last_row[w], -1] = 1
            # Now the projections matrix is ready! Use it to compute the cols
            # and col_covs arrays
            cols = np.einsum('...ij,...j->...i', projections, cols)
            col_covs = np.einsum('...ij,...jk,...lk->...il',
                                 projections, col_covs, projections)
            # Add errors for degenerate matrices
            eps = 1e-10
            col_covs += np.identity(n_cols) * eps
        else:
            projections = None

        return ColorCatalogue(cols, col_covs, selection=self.selection, projections=projections,
                              log_probs=None, log_class_probs=self.log_class_probs)

    def fit_number_counts(self, start_completeness=20.0, start_width=0.3, method='Nelder-Mead',
                          indices=None):
        """Perform a fit with a simple model for the number counts.

        The assumed model is an exponential distribution, truncated at high
        magnitudes with an erfc function:

        ..math: p(m) \\propto \\exp(b m) \\erfc((m - c) / \\sqrt{2 s^2})

        where b is the number count slope, c the completeness limit, and s
        its width. Note that this procedure must be used to correctly
        simulate extinction in a control field.

        The results of the best-fit are saved in self.nc_pars and also
        returned.
        
        Arguments
        ---------
        start_completeness : float, default = 20.0
            The initial guess for the 50% completeness magnitude.

        start_width : float, default = 0.3
            The initial guess for the width of the completeness truncation.

        method : string, default = 'Nelder-Mead'
            The optimization method to use (see `minimize`).

        indices : list of indices, slice, or None
            If provided, an index specification indicating the objects to use.

        Return value
        ------------
        nc_pars : array like, shape (n_bands, 3)
            Array with the tripled (exponential slope, 50% completeness
            limit, completeness width) for each band, also saved in
            self.nc_pars.
        """
        def number_count_lnlikelihood_(x, mags):
            # Magnitude errors ignored here, not sure if important...
            c = x[0]  # c -> m_c
            s = x[1]  # s -> \sigma_c
            # Solve analytically for the best beta, this reduces one parameter
            d = c - np.mean(mags)
            b = 2.0 / (d + np.sqrt(d * d + 4 * s * s))

            # Model: p(m) = exp(n)*exp(b*m+b*b*e*e/2) * 0.5*erfc((m-c) / sqrt(2*s*s))
            # Normalization: K = exp(n)/b * exp(b*c + (e*e + s*s) * b*b / 2)

            # Just use normalized probability, this reduces one further parameter:
            # ln L = sum(log p(m)) - N*log K
            lnlike = np.log(b) - b * d - b * b * 0.5 * s * s + \
                np.mean(log_ndtr((c - mags) / np.abs(s)))
            return -lnlike

        nc_pars = []
        if indices is None:
            indices = slice(None)
        for band in range(self.n_bands):
            mags = self.mags[indices, band]
            mags = mags[self.mag_errs[indices, band] < self.max_err]
            p0 = np.array([start_completeness, start_width])
            m = minimize(number_count_lnlikelihood_, p0,
                         args=(mags,), method=method)
            m_c = m.x[0]
            s_c = m.x[1]
            d = m_c - np.mean(mags)
            beta = 2.0 / (d + np.sqrt(d * d + 4 * s_c * s_c))
            nc_pars.append([beta, m_c, np.abs(s_c)])
        self.nc_pars = np.array(nc_pars)
        return self.nc_pars

    def log_completeness(self, band, magnitudes):
        """Compute the log of the completeness function in a given band.

        Parameters
        ----------
        band : int
            The band to use for the calculation; negative values are accepted.

        magnitudes : float or array-like, shape (n_mags,)
            A single magnitude or an array of magnitudes, indicating where the
            calculation of the completeness is to be performed.

        Returns
        -------
        float or array-like, shape (n_mags,)
            The log of the completeness function, computed for the requested
            band, at the magnitude values indicated.
        """
        _, m_c, s_c = self.nc_pars[band]
        return log_ndtr((m_c - magnitudes) / s_c)

    def extinguish(self, extinction, apply_completeness=True, update_errors=False):
        """Simulate the effect of extinction and return an updated catalogue.

        Arguments
        ---------
        extinction : float or array-like, shape (n_bands,)
            The extinction to apply for each band, in magnitudes. Must be always
            non-negative. If it is a float, multiply it by self.reddening_law to
            obtain the extinction in each band.

        apply_completeness : bool, default to True
            If True, the completeness function is taken into account, and
            random objects that are unlikely to be observable given the added
            extinction are dropped. Requires one to have called the
            `fit_number_counts` method before. If the catalogue has
            probabilities associated, the function updates the probabilities
            and does not perform decimation.

        update_errors : bool, default to False
            If set, errors are also modified to reflect the fact that objects
            are now fainter. Not implemented yet.

        Returns
        -------
        PhotometricCatalogue
            The updated PhotometricCatalogue (self is left untouched).
        """
        cat = copy.deepcopy(self)
        if isinstance(extinction, float):
            extinction = extinction * self.reddening_law
        if apply_completeness and cat.nc_pars is None:
            # Fit the number counts if this has not been done earlier on
            cat.fit_number_counts()
        for band in range(cat.n_bands):
            mask = np.where(cat.mag_errs[:, band] < cat.max_err)[0]
            cat.mags[mask, band] += extinction[band]
            if update_errors:
                # It would be a good idea to update the errors too! Find a reliable way to do it
                raise NotImplementedError
            if apply_completeness and extinction[band] > 0:
                log_completeness_ratio = (cat.log_completeness(band, cat.mags[mask, band]) -
                                          cat.log_completeness(band, cat.mags[mask, band] - extinction[band]))
                if cat.log_probs is not None:
                    cat.log_probs[mask, band] += log_completeness_ratio
                else:
                    removed = log_completeness_ratio < np.log(
                        np.random.uniform(size=len(mask)))
                    cat.mags[mask[removed], band] = cat.null_mag
                    cat.mag_errs[mask[removed], band] = cat.null_err
        # Now remove objects with errors too large
        if cat.log_probs is None and (apply_completeness or update_errors):
            w = np.where(np.sum(cat.mag_errs < cat.max_err,
                                axis=1) >= cat.min_bands)[0]
            cat = cat[w]
        return cat


class ColorCatalogue(object):
    """Initialize a new color catalogue.

    The initialization can be carried using an arrays of colors and associated
    covariance matrices.

    Parameters
    ----------
    cols : array_like, shape (n_objs, n_cols)
        An array with the colors of all objects.

    col_covs : array_like, shape (n_objs, n_cols, n_cols)
        An array with the color covariance matrices.

    selection : array_like, shape (n_objs,) or None
        Indices in the original catalogue of objects that have been used. If
        None, all objects (in their order) are taken to have been used.

    projections : array_like, shape (n_objs, n_cols, n_cols) or None
        If set, the array of projection matrices used to compute the colors.

    log_probs : array_like, shape (n_objs,) or None
        An array with the (log) probabilities to detect a given color
        combination (i.e., object). If set to None, all objects are assumed to
        be observable with 100% probability.

    class_names : tuple or list of strings or None
        The name of the various discrete classes of objects the catalogue
        describes. If None, all objects are taken to be part of the same
        class.

    log_class_probs : array_like, shape (n_objs, n_classes) or None
        If the catalogue contains classified objects, an array that stores,
        for each object, the logarithms of the probability that each object
        belongs to a given class. The probabilities, if provided, should be
        normalized to unity, so that logaddexp(log_class_probs, axis=1) == 0.


    Attributes
    ----------
    n_objs : int
        The final number of objects in the catalogue, after all filtering.

    n_bands : int
        The maximum number of bands used; note tshat some objects might have
        less bands available.

    cols : array_like, shape (n_objs, n_cols)
        An array with the colors of all objects.

    col_covs : array_like, shape (n_objs, n_cols, n_cols)
        An array with the color covariance matrices.

    selection : array_like, shape (n_objs,)
        Indices in the original catalogue of objects that have been used.

    projections : array_like, shape (n_objs, n_cols, n_cols) or None
        If set, the array of projection matrices used to compute the colors.

    log_probs : array_like, shape (n_objs,) or None
        An array with the (log) probabilities to detect a given color
        combination (i.e., object). If set to None, all objects are assumed to
        be observable with 100% probability.

    class_names : tuple or list of strings or None
        The name of the various discrete classes of objects the catalogue
        describes. If None, all objects are taken to be part of the same
        class.

    log_class_probs : array_like, shape (n_objs, n_classes) or None
        If the catalogue contains classified objects, an array that stores,
        for each object, the logarithms of the probability that each object
        belongs to a given class. The probabilities, if provided, should be
        normalized to unity, so that logaddexp(log_class_probs, axis=1) == 0.
    """

    def __init__(self, cols, col_covs, selection=None, projections=None,
                 log_probs=None, class_names=None, log_class_probs=None):
        if cols.ndim != 2:
            raise ValueError('Colors must be a two-dimensional array')
        n_objs, n_cols = cols.shape
        if col_covs.ndim != 3:
            raise ValueError(
                'Color covariances must be a three-dimensional array')
        if col_covs.shape[0] != n_objs:
            raise ValueError(
                "Color and color covariances must have the same number of objects")
        if col_covs.shape[1] != n_cols or col_covs.shape[2] != n_cols:
            raise ValueError(
                "Wrong shape for the color covariance matrix")
        if selection is not None:
            if selection.ndim != 1:
                raise ValueError(
                    'Selection must be a one-dimensional array')
            if selection.shape[0] != n_objs:
                raise ValueError(
                    "Color and selection must have the same number of objects")
        if projections is not None:
            if projections.ndim != 3:
                raise ValueError(
                    'Projections must be a three-dimensional array')
            if projections.shape[0] != n_objs:
                raise ValueError(
                    "Color and projections must have the same number of objects")
            if projections.shape[1] != n_cols or projections.shape[2] != n_cols:
                raise ValueError(
                    "Wrong shape for the projection matrix")
        if log_probs is not None:
            if log_probs.ndim != 1:
                raise ValueError(
                    'Log probabilities must be a one-dimensional array')
            if log_probs.shape[0] != n_objs:
                raise ValueError(
                    "Color and log-probabilities must have the same number of objects")
        if log_class_probs is not None:
            if log_class_probs.ndim != 2:
                raise ValueError(
                    'Log probabilities for classes must be a two-dimensional array')
            if log_class_probs.shape[0] != n_objs or \
                log_class_probs.shape[1] != len(class_names):
                raise ValueError(
                    'Wrong size of log-probabilities for classes')
        elif class_names is not None:
            raise ValueError('If class names are provided, log probabilities '
                'for classes must be provided too')
        # Saves the results
        self.n_objs = n_objs
        self.n_cols = n_cols
        self.cols = cols
        self.col_covs = col_covs
        if selection is not None:
            self.selection = selection
        else:
            self.selection = np.arange(n_objs)
        self.projections = projections
        self.log_probs = log_probs
        self.class_names = class_names
        self.log_class_probs = log_class_probs

    def __len__(self):
        return self.n_objs

    def __getitem__(self, sliced):
        res = copy.deepcopy(self)
        res.cols = res.cols[sliced]
        res.col_covs = res.col_covs[sliced]
        res.selection = res.selection[sliced]
        if res.projections is not None:
            res.projections = res.projections[sliced]
        if res.log_probs is not None:
            res.log_probs = res.log_probs[sliced]
        if res.log_class_probs is not None:
            res.log_class_probs = res.log_class_probs[sliced]
        try:
            res.n_objs = len(res.selection)
        except TypeError:
            res.n_objs = 1
        return res


class ExtinctionCatalogue(object):
    """The result of XNicer extinction measurements.
    
    Parameters
    ----------
    n_objs : int
        The number of objects present in the catalogue

    n_components : int
        The number of Gaussian components for each extinction measurement.

    n_colors : int, default=0
        If >0, the number of colors used for the estimate of the intrinsic
        colors of each star.
    
    selection : array-like, shape (n_objs,) or None
        If not None, the original selection of objects in the original
        catalogue.
    
    Attributes
    ----------
    n_objs : int
        The number of objects present in the catalogue

    n_components : int
        The number of Gaussian components for each extinction measurement.

    selection : array-like, shape (n_objs,) or None
        If not None, the original selection of objects in the original
        catalogue.

    log_weights : array-like, shape (n_objs, n_components)
        The log of the weight of each component for each object.

    means_A : array-like, shape (n_objs, n_components)
        The centers of the Gaussian profiles for each extinction component 
        of each object.

    means_c : array-like, shape (n_objs, n_components, n_colors)
        The centers of the Gaussian profiles for each intrinsic color 
        component of each object.
        
    variances_A : array-like, shape (n_objs, n_components)
        The variances of the Gaussian profiles for each extinction component
        of each object.

    variances_c : array-like, shape (n_objs, n_components, n_colors, n_colors)
        The (co)variances of the Gaussian profiles for each intrinsic color 
        component of each object.

    covariances : array-like, shape (n_objs, n_components, n_colors)
        The covariances of the Gaussian profiles between extinction and
        intrinsic colors for each component of each object.

    log_evidence : array-like, shape (n_objs,)
        The log evidence for each extinction measurement of each object. A
        small evidence might indicate that the object is a spurious detection.

    log_weight : array-like, shape (n_objs,)
        The log weight of each object. It is the log of the sum of
        exp(log_weights). It is normally unity, unless the associated object
        in the original color catalogue has probability different than unity
        (that is, a log_probs smaller than zero).

    mean_A : array-like, shape (n_objs,)
        The center of the single Gaussian profile that best approximates the
        extinction measurement. This is the single value to use for the
        extinction measurement (if one only needs a single value...)

    mean_c : array-like, shape (n_objs, n_colors)
        The center of the single Gaussian profile that best approximates the
        intrinsic color. This is the single value to use for the
        intrinsic color (if one only needs a single value...)

    variance_A : array-like, shape (n_objs,)
        The variance associated to mean_A, that is, an estimate of the square
        of the error of the extinction measurement for each object.

    variance_c : array-like, shape (n_objs, n_colors, n_colors)
        The (co)variance associated to mean_c, for each object.

    covariance : array-like, shape (n_objs, n_colors)
        The covariance between mean_A and mean_c, for each object.

    xnicest_bias : array-like, shape (n_objs,)
        The estimated bias associated to each object in the XNicest algorithm.
    
    xnicest_weight : array-like, shape (n_objs,)
        The estimated weight associated to each object that needs to be used
        in the XNicest algorithm.
    """

    def __init__(self, n_objs, n_components, n_colors=0, selection=None):
        self.n_objs = n_objs
        self.n_components = n_components
        self.n_colors = n_colors
        self.selection = selection
        self.log_weights = np.zeros((n_objs, n_components))
        self.means_A = np.zeros((n_objs, n_components))
        self.variances_A = np.zeros((n_objs, n_components))
        self.log_evidence = np.zeros(n_objs)
        self.log_weight = np.zeros(n_objs)
        self.mean_A = np.zeros(n_objs)
        self.variance_A = np.zeros(n_objs)
        if self.n_colors > 0:
            self.means_c = np.zeros((n_objs, n_components, n_colors))
            self.covariances = np.zeros((n_objs, n_components, n_colors))
            self.variances_c = np.zeros((n_objs, n_components, n_colors, n_colors))
            self.mean_c = np.zeros((n_objs, n_colors))
            self.covariance = np.zeros((n_objs, n_colors))
            self.variance_c = np.zeros((n_objs, n_colors, n_colors))
        self.xnicest_bias = None
        self.xnicest_weight = None

    def __len__(self):
        return self.n_objs

    def __getitem__(self, sliced):
        res = copy.deepcopy(self)
        if res.selection is not None:
            res.selection = res.selection[sliced]
        res.log_weights = res.log_weights[sliced]
        res.means_A = res.means_A[sliced]
        res.variances_A = res.variances_A[sliced]
        res.log_evidence = res.log_evidence[sliced]
        res.log_weight = res.log_weight_[sliced]
        res.mean_A = res.mean_A[sliced]
        res.variance_A = res.variance_A[sliced]
        if self.n_colors > 0:
            res.means_c = res.means_c[sliced]
            res.covariances = res.covariances[sliced]
            res.variances_c = res.variances_c[sliced]
            res.mean_c = res.mean_c[sliced]
            res.covariance = res.covariance[sliced]
            res.variance_c = res.variance_c[sliced]
        if res.xnicest_bias is not None:
            res.xnicest_bias = res.xnicest_bias[sliced]
        if res.xnicest_weight is not None:
            res.xnicest_weight = res.xnicest_weight[sliced]
        try:
            res.n_objs = len(res.mean_)
        except TypeError:
            res.n_objs = 1
        return res

    def update_(self):
        """Compute the mean_, variance_, and log_weight_ attributes.

        The mean_ and variance_ attributes are just a single-component
        equivalent of the multi-component extinction.  They are computed using
        a technique identical to the one implemented by merge_components. """
        ws = np.exp(self.log_weights)
        norm = np.sum(ws, axis=-1)
        self.log_weight = np.log(norm)
        self.mean_A = np.sum(ws * self.means_A, axis=-1) / norm
        diff_A = self.means_A - self.mean_A[..., np.newaxis]
        self.variance_A = np.sum(
            ws*(self.variances_A + diff_A**2), axis=-1) / norm
        if self.n_colors > 0:
            self.mean_c = np.sum(ws[..., np.newaxis] *
                                self.means_c, axis=-2) / norm[..., np.newaxis]
            diff_c = self.means_c - self.mean_c[..., np.newaxis, :]
            self.variance_c = np.sum(
                ws[..., np.newaxis, np.newaxis] *
                (self.variances_c +
                diff_c[..., np.newaxis, :]*diff_c[..., np.newaxis]),
                axis=-3) / norm[..., np.newaxis, np.newaxis]
            self.covariance = np.sum(
                ws[..., np.newaxis] *
                (self.covariances + diff_A[..., np.newaxis] * diff_c),
                axis=-2) / norm[..., np.newaxis]
    
    def merge_components(self, merged_components=None):
        """Merge a number of components together.

        Arguments
        ---------
        merged_components : array-like, shape (_,)
            The array containing the components that need to be merged together.
            If not specified, all components are merged.
        """
        # TODO: Add the other parts
        if self.n_components == 1:
            return self
        if merged_components is None:
            merged_components = np.arange(self.n_components)
        # Make sure merged_components[0] is the smallest value
        if merged_components[0] >= np.min(merged_components[1:]):
            merged_components = np.sort(merged_components)
        # Extract the relevant values
        ws = np.exp(self.log_weights[..., merged_components])
        bs = self.means_A[..., merged_components]
        Vs = self.variances_A[..., merged_components]
        # Compute the parameters of the merged gaussians
        w = np.sum(ws, axis=-1)
        b = np.sum(ws * bs, axis=-1) / w
        V = np.sum(ws * (Vs + (b[..., np.newaxis] - bs)**2), axis=-1) / w
        # Create the new parameter arrays
        ws = np.delete(self.log_weights, merged_components[1:], axis=-1)
        bs = np.delete(self.means_A, merged_components[1:], axis=-1)
        Vs = np.delete(self.variances_A, merged_components[1:], axis=-1)
        n_components = self.n_components - len(merged_components) + 1
        ws[..., merged_components[0]] = np.log(w)
        bs[..., merged_components[0]] = b
        Vs[..., merged_components[0]] = V
        self.log_weights_ = ws
        self.means_ = bs
        self.variances_ = Vs
        self.n_components = n_components
        return self

    def score_samples_components(self, X, Xerr):
        """
        Compute the log of the probability distribution for each component.

        Given a specific extinction value and associated error, this method
        computes the score (that is, the log of the pdf) for that extinction
        for all components and all objects.

        Parameters
        ----------
        X: array_like, shape (n_samples,)
            Input data.

        Xerr: array_like, shape (n_samples,)
            Variance on input data.

        Returns
        -------
        logprob : array_like, shape (n_samples, n_components)
            Log probabilities of each data point, for each component.
        """
        T = Xerr[:, np.newaxis] + self.variances_A
        delta = X[:, np.newaxis] - self.means_A
        return -delta*delta/(2.0*T) - np.log(2.0*np.pi*T) / 2

    def score_samples(self, X, Xerr):
        """
        Compute the log of the probability distribution for all objects.

        Given a specific extinction value and associated error, this method
        computes the score (that is, the log of the pdf) for that extinction
        for all objects.

        Parameters
        ----------
        X: array_like, shape (n_samples,)
            Input data.

        Xerr: array_like, shape (n_samples,)
            Variance on input data.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in A.
        """
        log = self.score_samples_components(X, Xerr)
        return logsumexp(self.log_weights + log, axis=-1)
