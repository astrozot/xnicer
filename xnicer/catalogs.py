"""Catalogue handling code.

:Author: Marco Lombardi
:Version: 0.2.0 of 2020/05/30
"""

# Author: Marco Lombardi <marco.lombardi@gmail.com>

import collections
import warnings
import copy
import logging
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr # pylint: disable=no-name-in-module
from astropy import table
from astropy.io import votable
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from .utilities import log1mexp
from .kde import KDE

LOGGER = logging.getLogger(__name__)

# Table 3 from Rieke & Lebovsky (1985), ApJ 288, 618: ratio A_lambda / A_V.
RIEKE_LEBOVSKY_UCD = {
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
RIEKE_LEBOVSKY_NAMES = {
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
    """Try to find the reddening for a given band name."""
    name = name.upper()
    if name in RIEKE_LEBOVSKY_NAMES:
        return RIEKE_LEBOVSKY_NAMES[name]
    elif name[-4:] == '_MAG' and name[:-4] in RIEKE_LEBOVSKY_NAMES:
        return RIEKE_LEBOVSKY_NAMES[name[:-4]]
    elif name[-3:] == 'MAG' and name[:-3] in RIEKE_LEBOVSKY_NAMES:
        return RIEKE_LEBOVSKY_NAMES[name[:-3]]
    else:
        return False


class AstrometricCatalogue(SkyCoord):
    """An astrometric catalogue, child of SkyCoord.

    Defined so that one can use some class methods to build the catalogue.
    """

    @classmethod
    def from_votable(cls, input_table, frame=None, **kwargs):
        """Extract coordinates from a VOTable into a SkyCoord object.

        Parameters
        ----------
        input_table : VOTable
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
        for field in input_table.fields:
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
                    if force or result[ucd_frame][i] is None:
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
        return cls(input_table.array[result[0].ID],
                   input_table.array[result[1].ID],
                   frame=frame, unit=result[0].unit,
                   equinox=result[0].ref, **kwargs)

    @classmethod
    def from_table(cls, input_table, colnames, unit='deg', frame='icrs', **kw):
        """Build a new AstrometricCatalogue from an astropy Table.

        Parameters
        ----------
        input_table : astropy Table
            The table from which we can extract the coordinates.

        colnames : sequence or list of strings
            The names of the columns with the object coordinates.

        unit : str, default='deg'
            The units of the coodinates.

        frame : str, default='icrs'
            The frame of the coordinates.

        Returns
        -------
        AstrometricCatalogue
            The newly built astrometric catalogue (a child of SkyCoord).

        """
        pars = []
        for colname in colnames:
            pars.append(input_table[colname])
        return cls(*pars, unit=unit, frame=frame, **kw)

class PhotometricCatalogue(table.Table):
    """A photometric catalogue (child of astropy Table).

    The catalogues will contain at least the following

    Columns
    -------
    mags :

    Attributes
    ----------
    n_objs : int
        The final number of objects in the catalogue, after all filtering.

    n_bands : int
        The maximum number of bands used; note that some objects might have
        less bands available.

    mags : array_like, shape (n_objs, n_bands)
        Array with the extracted magnitudes.

    mag_names : array_like, shape (n_bands,) or None
        The names of the bands present in the catalogue, if available. This
        variable is only set if the user provide the parameter `data`.

    mag_errs : array_like, shape (n_objs, n_bands, n_bands)
        Array with the extracted magnitude errors.

    err_names : array_like, shape (n_bands,) or None
        The names of the error in each band present in the catalogue, if
        available. This variable is only set if the user provide the
        parameter `data`.

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

    reddening_law : array_like, shape (n_bands,) or None
        The reddening law associated with the bands in the catalogue.

    nc_pars: array_like, shape (n_bands, 3) or None
        Array that reports, for each band,< the best-fit number-count
        parameters, written as (exponential slope, 50% completeness limit,
        completeness width).

    max_err : float
        Maximum admissible error above which a datum is discarted.

    min_bands : int
        The minimum number of bands with valid measuremens to include the
        object in the final catalogue.

    null_mag: float
        The value used to mark a null magnitude.

    null_err: float
        The value used to mark a null magnitude error.

    """

    @classmethod
    def from_votable(cls, data, dtype=np.float32, reddening_law=None, **kw):
        """Build a PhotometricCatalogue using a VO Table.

        Parameters
        ----------
        data : VOTable
            The VOTable containing the original data. The magnitudes and their
            associated errors are automatically extracted from the table.

        dtype : data-type, default=np.float32
            The data-type associated to the various columns of the photometric
            catalogue.

        reddening_law : array-like, optional
            The reddening law associated with the bands contained in the table.
            If not provided, the reddening law is inferred from the band names
            and taken to follow the Rieke & Lebovsky (1985) one.

        kw : dict
            Optional keywords, passed to `from_table`. Especially useful for
            classification codes (hard to parse with UCD's).

        Returns
        -------
        The newly built PhotometricCatalogue.

        """
        bands = collections.OrderedDict()
        for field in data.fields:
            ucd = votable.ucd.parse_ucd(field.ucd)
            if ucd[0][1] == 'phot.mag':
                # it's a magnitude, get its name
                mag = ucd[1][1]
                if mag in bands:
                    if bands[mag][0] is None:
                        bands[mag][0] = field.ID
                else:
                    bands[mag] = [field.ID, None]
            if ucd[0][1] == 'stat.error' and \
                ucd[1][1] == 'phot.mag':
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
        if reddening_law is None:
            fit_reddening_law = True
            reddening_law = []
        else:
            fit_reddening_law = False
        for ucd, (mag, mag_err) in bands.items():
            if mag is not None and mag_err is not None:
                mags.append(mag)
                mag_errs.append(mag_err)
                if fit_reddening_law:
                    if ucd in RIEKE_LEBOVSKY_UCD:
                        reddening_law.append(RIEKE_LEBOVSKY_UCD[ucd])
                    else:
                        reddening = _find_reddening_vector(mag)
                        if reddening:
                            reddening_law.append(reddening)
                        else:
                            warnings.warn(f"Cannot automatically find the reddening law for {mag}")
                            reddening_law = False
        if reddening_law is False:
            reddening_law = None
        return cls.from_table(data.array, mags, mag_errs, dtype=dtype,
                              reddening_law=reddening_law, **kw)

    @classmethod
    def from_table(cls, data, mag_names, err_names, *args, prob_names=None,
                   class_prob_names=None, **kw):
        """Build a catalogue from an astropy Table.

        Parameters
        ----------
        data : Table or table-like object
            A table-like object containing the magnitudes and the associated
            errors.

        mag_names : tuple or list of strings
            A list of strings indicating which columns of data should be
            interpreted as magnitudes.

        err_names : tuple or list of strings
            A list of strings indicating which columns of data should be
            interpreted as magnitude errors.

        prob_names : tuple or list of strings, or None
            A list of strings indicating which columns of data should be
            interpreted as detection probabilities. If None, the probabilities
            are ignored (that is, all objects are assumed to be observable with
            100% probability).

        class_prob_names : tuple or list of strings, or None
            A list of strings indicating which columns of data are to be taken
            the probability that each object belongs to a given class. If set,
            one needs also to set class_names (see `from_photometry`). If None,
            all objects are taken to be par of the same unique class.

        All further parameters are passed to `from_photometry`.

        """
        # Kind of data allocator: we used to change allocator depending on
        # data.masked; however, this does not work, since a table might not
        # be masked, and still have columns represented as masked arrays.
        allocator = np.ma.empty

        n_objs = len(data)
        n_bands = len(mag_names)
        if n_bands != len(err_names):
            raise ValueError(
                "Magnitudes and errors must have the same number of bands")
        mags = allocator((n_objs, n_bands))
        mag_errs = allocator((n_objs, n_bands))
        if prob_names is not None:
            if n_bands != len(prob_names):
                raise ValueError(
                    "Magnitudes and log-probabilites must have the same number of bands")
            probs = allocator((n_objs, n_bands))
        else:
            probs = None

        for n in range(n_bands):
            mags[:, n] = data[mag_names[n]]
            mag_errs[:, n] = data[err_names[n]]
            if prob_names is not None:
                probs[:, n] = data[prob_names[n]]
        if class_prob_names is not None:
            class_probs = allocator((n_objs, len(class_prob_names)))
            for n, name in enumerate(class_prob_names):
                class_probs[:, n] = data[name]
        else:
            class_probs = None
        return cls.from_photometry(mags, mag_errs,
                                   mag_names=mag_names, err_names=err_names,
                                   probs=probs, class_probs=class_probs,
                                   *args, **kw)

    @classmethod
    def from_photometry(cls, mags, mag_errs, mag_names=None, err_names=None,
                        probs=None, dtype=np.float32,
                        log_probs=False, class_names=None, class_probs=None,
                        log_class_probs=False, reddening_law=None,
                        max_err=1.0, min_bands=2, null_mag=15.0, null_err=99.999):
        """Initialize a new photometric catalogue from raw data (arrays).

        Parameters
        ----------
        mags : array_like, shape (n_objs, n_bands)
            An array with the measured magnitudes of all objects.

        mag_errs : array_like, shape (n_objs, n_bands)
            An array with the measured magnitude errors of all objects.

        mag_names : list of strings, default=None
            The names associated to each magnitude band.

        err_names : list of strings, default=None
            The names associated to each error in magnitude.

        probs : array_like, shape (n_objs, n_bands)
            The (log) probability to observe a given magnitude for a given
            object. Used to simulate the effects of extinction through
            `extinguish`. If None, the probabilities are ignored (that is, all
            objects are assumed to be observable with 100% probability).

        dtype : data-type, default=np.float32
            The data-type associated to the various columns of the photometric
            catalogue.

        log_probs : bool, default=False
            If True, `probs` are really log probabilities; otherwise, are just
            probabilities.

        class_names : tuple or list of strings or None
            The name of the various discrete classes of objects the catalogue
            describes. If None, all objects are taken to be part of the same
            class.

        class_probs : array_like, shape (n_objs, n_classes) or None
            If the catalogue contains classified objects, an array that stores,
            for each object, the (log) probability that each objecs belongs to
            a given class. The probabilities should be normalized to unity
            (that is, the sum for a given object of all probabilities must be
            1). Alternatively, one can provide an array of shape (n_objs,
            n_classes-1): in this case, it is assumed that these probabilities
            are associated to the first n_classes-1 classes; the probability
            for the last class is inferred from the normalization condition.

        log_class_probs : bool, default=False
            If True, `class_probs` are really log probabilities.

        reddening_law : array_like, shape (n_bands,) or None
            The reddening law associated with the bands in the catalogue.

        max_err : float, default to 1.0
            Maximum admissible error above which a datum is discarted.

        min_bands : int, default to 2
            The minimum number of bands with valid measuremens to include the
            object in the final catalogue.

        null_mag: float, default to 15.0
            The value used to mark a null magnitude.

        null_err: float, default to 99.999
            The value used to mark a null magnitude error.

        """
        cat = cls()

        # Initial checks
        if mags.ndim != 2 or mag_errs.ndim != 2:
            raise ValueError(
                "Magnitudes and errors must be two-dimensional arrays")
        n_objs, n_bands = mags.shape
        cat.meta['n_bands'] = n_bands
        if n_objs < n_bands:
            raise ValueError(
                "Expecting a n_objs x n_bands array for mags and errs")
        if mag_errs.shape[0] != n_objs:
            raise ValueError(
                "Magnitudes and errors must have the same number of objects")
        if mag_errs.shape[1] != n_bands:
            raise ValueError(
                "Magnitudes and errors must have the same number of bands")
        if reddening_law is None:
            warnings.warn("Reddening law not specified")
            cat.meta['reddening_law'] = None
        else:
            if len(reddening_law) != n_bands:
                raise ValueError(
                    "The length of the reddening_law vector does not match the number of bands")
            cat.meta['reddening_law'] = np.array(reddening_law)

        # First of all the easy stuff
        cat.meta['max_err'] = max_err
        cat.meta['min_bands'] = min_bands
        cat.meta['nc_pars'] = None
        cat.meta['noise_pars'] = None
        cat.meta['null_mag'] = null_mag
        cat.meta['null_err'] = null_err
        cat.meta['reddening_law'] = reddening_law.copy() \
            if reddening_law else []
        if cat.meta['reddening_law']:
            cat.meta['reddening_law'] = np.array(cat.meta['reddening_law'])
        if mag_names is not None:
            cat.meta['mag_names'] = tuple(mag_names)
        else:
            cat.meta['mag_names'] = tuple(f'mag{i+1}' for i in range(n_bands))
        if err_names is not None:
            cat.meta['err_names'] = tuple(err_names)
        else:
            cat.meta['err_names'] = tuple(f'err_mag{i+1}'
                                          for i in range(n_bands))
        if class_names is not None:
            cat.meta['class_names'] = tuple(class_names)
        else:
            cat.meta['class_names'] = None

        # Create the main columns
        mask = np.zeros((n_objs, n_bands), dtype=np.bool)
        if isinstance(mags, np.ma.MaskedArray):
            mask |= mags.mask
            mags = mags.filled(null_mag)
        cat.add_column(table.Column(
            mags.astype(dtype), copy=False,
            name='mags', description='Array of magnitudes',
            unit='mag', format='%6.3f'))
        if isinstance(mag_errs, np.ma.MaskedArray):
            mask |= mag_errs.mask
            mag_errs = mag_errs.filled(null_err)
        cat.add_column(table.Column(
            mag_errs.astype(dtype), copy=False,
            name='mag_errs', description='Array of errors in magnitudes',
            unit='mag', format='%5.3f'))
        mask |= cat['mag_errs'] >= max_err
        cat['mags'][mask] = null_mag
        cat['mag_errs'][mask] = null_err

        # log_prob column
        if probs is not None:
            if probs.ndim != 2:
                raise ValueError(
                    "(log-)probabities must be a two-dimensional array")
            if n_objs != probs.shape[0]:
                raise ValueError(
                    "(log-)probabilites must have the right number of objects")
            if n_bands != probs.shape[1]:
                raise ValueError(
                    "(log-)probabilites must have the right number of bands")
            if not log_probs:
                with np.errstate(divide='ignore'):
                    probs = np.log(probs)
                    probs[np.ma.getmaskarray(probs)] = -np.inf
            if isinstance(probs, np.ma.MaskedArray):
                probs = probs.filled(-np.inf)
            cat.add_column(table.Column(
                probs.astype(dtype), copy=False, name='log_probs',
                description='Log-probabilities of detection on each band',
                format='%+5.3f'))
            cat['log_probs'][mask] = -np.inf

        # class_probs column
        if class_probs is not None:
            if class_probs.ndim != 2:
                raise ValueError(
                    "(log-)probabities of classes must be a two-dimensional array")
            if n_objs != class_probs.shape[0]:
                raise ValueError(
                    "(log-)probabilites of classes must have the right number of objects")
            if class_names is None:
                raise ValueError("class names must be specified if class probabilities are used")
            if len(class_names) == class_probs.shape[1] + 1:
                class_probs = np.hstack((class_probs, np.zeros((n_objs, 1))))
                if log_class_probs:
                    class_probs[:, -1] = np.log(1.0 -
                                                np.sum(np.exp(class_probs[:, :-1]), axis=1))
                else:
                    class_probs[:, -1] = 1.0 - \
                        np.sum(class_probs[:, :-1], axis=1)
            if len(class_names) != class_probs.shape[1]:
                raise ValueError(
                    "(log-)probabilites of classes must have the same number of classes")
            if not log_class_probs:
                with np.errstate(divide='ignore'):
                    class_probs = np.log(class_probs)
                    class_probs[np.ma.getmaskarray(class_probs)] = -np.inf
            cat.add_column(table.Column(
                class_probs.astype(dtype), copy=False, name='log_class_probs',
                description='log-probabilities of class belonging to',
                format='%+5.3f'))

        # Remove objects with too few bands
        idx = np.where(np.sum(cat['mag_errs'] < max_err, axis=1)
                       >= min_bands)[0]
        cat = cat[idx]
        cat.add_column(table.Column(
            idx, name='idx',
            description='index with respect to the original catalogue',
            format='%d'
        ), index=0)
        return cat

    def __getitem__(self, item):
        """Get item method."""
        if isinstance(item, str):
            if item in self.colnames:
                return self.columns[item]
            else:
                return None
        else:
            return super().__getitem__(item)

    def __add__(self, cat):
        """Concatenate two PhotometricCatalogue's."""
        if not isinstance(cat, self.__class__):
            raise ValueError("Expecting a PhotometricCatalogue here")
        if self.meta['n_bands'] != cat.meta['n_bands']:
            raise ValueError("The two catalogues have a different number of bands")
        # Only use classes if they are present and identical in both catalogues
        if 'class_names' not in self.meta or \
            'class_names' not in cat.meta:
            use_classes = False
        else:
            use_classes = np.all(np.array(self.meta['class_names'])
                                 == np.array(cat.meta['class_names']))
        # TODO: Fix possible differences in null_mag and null_err
        res = table.vstack((self, cat))
        if not use_classes and 'class_log_probs' in res.colnames:
            res.remove_column('class_log_probs')
        if res.masked:
            # Table is masked: we must have missed some columns...
            if 'log_probs' in res.colnames and np.any(res['log_probs'].mask):
                w = np.any(res['log_probs'].mask, axis=1)
                res['log_probs'][w] = 0.0
            res = res.filled()
        for k in ('mag_names', 'err_names', 'class_names', 'reddening_law'):
            if k in self.meta and k in cat.meta:
                res.meta[k] = copy.copy(self.meta[k])
        if not use_classes:
            if 'class_names' in res.meta:
                del res.meta['class_names']
            if 'log_class_probs' in res.colnames:
                res.remove_column('log_class_probs')
        res.meta['nc_pars'] = None
        res.meta['noise_pars'] = None
        return res

    def add_log_probs(self):
        """Add (log) probabilities to the photometric catalogue.

        Magnitude measurements with magnitude errors larger than max_err are
        marked with 0 probability.
        """
        if 'log_probs' not in self.colnames:
            data = np.zeros((len(self), self.meta['n_bands']),
                            dtype=self['mags'].dtype)
            self.add_column(table.Column(
                data, copy=False, name='log_probs',
                description='log-probabilities for each object in each band',
                format='%+5.3f'))
        self['log_probs'][np.where(self['mag_errs'] > self.meta['max_err'])] = -np.inf

    def remove_log_probs(self):
        """Remove log probabilities from the photometric catalogue.

        This method also applies decimation to the data: that is, it "cleans"
        magnitudes bands depending on the value of the log probabilities.
        """
        if 'log_probs' not in self.colnames:
            return
        removed = np.where(self['log_probs'] < np.log(
            np.random.uniform(size=(len(self), self.meta['n_bands']))))[0]
        self.remove_rows(removed)
        self.remove_column('log_probs')

    def get_colors(self, *pars, **kw):
        """Return the ColorCatalogue associated to this PhotometricCatalogue.

        This function just call `ColorCatalogue.from_photometric_catalogue`
        passing all parameters to it.
        """
        return ColorCatalogue.from_photometric_catalogue(self, *pars, **kw)

    def fit_number_counts(self, start_completeness=20.0, start_width=0.3,
                          method='Nelder-Mead', indices=None):
        r"""Perform a fit with a simple model for the number counts.

        The assumed model is an exponential distribution, truncated at high
        magnitudes with an erfc function:

        ..math: p(m) \propto \exp(b m) \erfc((m - c) / \sqrt{2 s^2})

        where b is the number count slope, c the completeness limit, and s
        its width. Note that this procedure must be used to correctly
        simulate extinction in a control field.

        The results of the best-fit are saved in self.meta['nc_pars'] and
        also returned.

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
            self.meta['nc_pars'].

        """
        def number_count_lnlikelihood_(x, mags):  # pylint: disable=invalid-name
            """Service function to compute the likelihood for number counts."""
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
        for band in range(self.meta['n_bands']):
            mags = self['mags'][indices, band]
            mags = mags[self['mag_errs'][indices, band] < self.meta['max_err']]
            p0 = np.array([start_completeness, start_width])
            m = minimize(number_count_lnlikelihood_, p0,
                         args=(mags,), method=method)
            m_c = m.x[0]
            s_c = m.x[1]
            d = m_c - np.mean(mags)
            beta = 2.0 / (d + np.sqrt(d * d + 4 * s_c * s_c))
            LOGGER.info('Number-count fit for band %d: (%g, %g, %g)',
                        band, beta, m_c, np.abs(s_c))
            nc_pars.append([beta, m_c, np.abs(s_c)])
        self.meta['nc_pars'] = np.array(nc_pars, dtype=self['mags'].dtype)
        return self.meta['nc_pars']

    def plot_number_counts(self, band=0, bins=None, range=None, histtype='bar',
                           histcolor=None, histlabel=None, log=True, **kw):
        """Make a plot of the fitted number counts.

        Parameters
        ----------
        band : int, optional
            The band to consider, by default 0.

        bins : int or sequence or str, optional
            The bins parameter in `plt.hist`.

        range : tuple or None, optional
            The range to show for the histogram, by default None.

        histtype : str, optional
            The type of histogram to plot (see `plt.hist`), by default 'bar'.

        histcolor : str, optional
            The color for the histogram.

        histlabel : [type], optional
            The label to use for the histogram, by default None.

        log : bool, optional
            If True (default), the histogram will be done in log-space.

        Extra keyword parameters are passed to `errorbar`.

        """
        if self.meta['nc_pars'] is None:
            # We need the fit of the number counts
            self.fit_number_counts()
        errs = self['mag_errs'][:, band]
        w = np.where((errs < self.meta['null_err']))[0]
        mags = self['mags'][w, band]
        h = plt.hist(mags, bins=bins, range=range, histtype=histtype,
                     color=histcolor, label=histlabel, log=log)
        mags = (h[1][:-1] + h[1][1:]) * 0.5
        nc_pars = self.meta['nc_pars'][band]
        probs = np.exp(mags * nc_pars[0]) * ndtr((nc_pars[1] - mags) / nc_pars[2])
        probs *= np.sum(h[0]) / np.sum(probs)
        plt.errorbar(mags, probs, yerr=np.sqrt(probs), fmt='.', **kw)
        plt.xlabel(self.meta['mag_names'][band] if 'mag_names' in self.meta
                   else 'magnitude')
        plt.ylabel('number counts')


    def fit_phot_uncertainties(self, start_alpha=1.0, start_beta=1.0,
                               start_gamma=1.0, n_times=2, nc_cut=3.0,
                               method='Nelder-Mead', indices=None):
        r"""Perform a fit of the photometric uncertainties.

        The model assumes that the noise e in luminosity for each source can
        be written as

        ..math: e = \sqrt{\alpha l t + \beta t + \gamma}

        where l is the source luminosity and t is the exposure time (expressed
        in a suitable unit). This expression can correctly model the noise due
        to photon counting (shot noise, \alpha term), the dark current
        (termal noise, \beta term), the sky contribution (background noise,
        also included in the \beta term), and the readout noise (\gamma
        term). The model further assumes that all the greek-letter
        coefficients are constants for each source, and that only the time t
        changes (as a result, for example, of different coverage).

        The conversion from magnitudes to luminosities is done using the
        simple expression

        ..math: m = m0 - 2.5 \log_{10} l

        The constant m0 is chosen so that \alpha is approximately unity.

        The fit proceeds using a limited set of possible exposure times t (by
        default only two values), and then fits the other parameters.

        The results of the best-fit (i.e., the three greek coefficients for
        each band) together with the choice of m0 are saved in
        self.meta['noise_pars'] and also returned.

        Arguments
        ---------
        start_alpha : float, default = 1.0
            The initial guess for the alpha parameter.

        start_beta : float, default = 1.0
            The initial guess for the beta parameter.

        start_gamma : float, default = 1.0
            The initial guess for the gamma parameter.

        n_times : int, default = 2
            The number of different time values to use in the fit. The initial
            guesses are taken to be np.arange(1, n_times+1)

        nc_cut : float, array, or None, default = 3.0
            Cut to be carried out around the 50% completeness in units of the
            completeness width. Use nc_cut=None to avoid any cut. If a sequence
            is passed, it is used taken to be the nc_cut to use for each band.

        method : string, default = 'Nelder-Mead'
            The optimization method to use (see `minimize`).

        indices : list of indices, slice, or None
            If provided, an index specification indicating the objects to use.

        Return value
        ------------
        noise_pars : array like, shape (n_bands, 4)
            Array with the results of the fit (m0, \alpha, \beta, \gamma)
            for each band, also saved in self.meta['noise_pars'].

        """
        def phot_uncert_chisquare_(x, m0, mags, errs):  # pylint: disable=invalid-name
            """Chi-square function for the photometry uncertainty fit."""
            a, b, c = x[0:3]
            ts = np.hstack(([1.0], x[3:]))
            l = 10**(-0.4*(mags - m0))
            e_l = np.sqrt(a*a*ts[:, np.newaxis]*l[np.newaxis, :] +
                          b*b*ts[:, np.newaxis] + c*c)
            e_m = 2.5 / (l*np.log(10)) * e_l
            delta = np.min((errs[np.newaxis, :] - e_m)**2, axis=0)
            return np.sum(delta)

        noise_pars = []
        if indices is None:
            indices = slice(None)
        if nc_cut is not None:
            # We want to perform a cut in magnitude for faint objects
            if self.meta['nc_pars'] is None:
                # We need the fit of the number counts
                self.fit_number_counts(indices=indices)
            mag_lims = self.meta['nc_pars'][:, 1] - \
                self.meta['nc_pars'][:, 2]*np.array(nc_cut)
        else:
            mag_lims = np.repeat(np.inf, self.meta['n_bands'])
        LOGGER.debug('Magnitude limits: %g' + (', %g'*(len(mag_lims) - 1)),
                     *mag_lims)
        for band in range(self.meta['n_bands']):
            mags = self['mags'][indices, band]
            errs = self['mag_errs'][indices, band]
            w = np.where((errs < self.meta['null_err']) &
                         (mags < mag_lims[band]))[0]
            mags = mags[w]
            errs = errs[w]
            m0 = np.median(mags - 5 * np.log10(errs))
            p0 = np.array([start_alpha, start_beta, start_gamma])
            p0 = np.hstack((p0, np.arange(n_times-1) + 2))
            m = minimize(phot_uncert_chisquare_, p0,
                         args=(m0, mags, errs), method=method)
            LOGGER.info('Noise fit for band %d: (%g, %g, %g, %g)',
                        band, m0, m.x[0]**2, m.x[1]**2, m.x[2]**2)
            noise_pars.append(np.hstack(([m0], m.x[:3]**2)))
        self.meta['noise_pars'] = np.array(noise_pars,
                                           dtype=self['mags'].dtype)
        return self.meta['noise_pars']

    def plot_phot_uncertainties(self, band=0, stars_per_bin=1, kde=True,
                                norm=None, times=None, **kw):
        """Plot the fit to the photometric uncertainties.

        Parameters
        ----------
        band : int, optional
            The band to consider, by default 0

        stars_per_bin : int, optional
            The number of stars to plot in each 2D bin, by default 1

        kde : bool or dict, default True
            If False, the plot is done using a simple `hist2d`; otherwise,
            a KDE smoothing is performed and, if it is a dictionary, it is
            passed as keyword arguments to the `KDE` constructor.

        range : array-like, shapee (2, 2), optional
            The range in magnigude and magnitude errors, by default the full
            range

        norm : norm, optional
            A norm to use for the display, by default PowerNorm(0.5)

        times : array-like, shape (n,), optional
            A list of time values for the plot. If omitted, it is taken to be
            [1].

        All further keyword parameters are passed to `pcolormesh` (if kde is
        enabled) or to `hist2d`.

        """
        if norm is None:
            norm = PowerNorm(0.5)
        if times is None:
            times = [1]
        if self.meta['noise_pars'] is None:
            # We need the fit of the number counts
            self.fit_phot_uncertainties()
        errs = self['mag_errs'][:, band]
        w = np.where(errs < self.meta['null_err'])[0]
        errs = errs[w]
        mags = self['mags'][w, band]
        ranges = kw['range'] if ('range' in kw) else \
            [[np.min(mags), np.max(mags)],
             [np.min(errs), np.max(errs)]]
        sx, sy = plt.gcf().get_size_inches()
        nobjs = len(mags)
        biny = int(np.round(np.sqrt(nobjs*sy/(sx*stars_per_bin))))
        binx = int(np.round(nobjs/(biny*stars_per_bin)))
        if kde is not False:
            if kde is True:
                kde = KDE((binx, biny))
            else:
                kde = KDE((binx, biny), **kde)
            density = kde.kde(np.stack(((mags - ranges[0][0]) /
                                        (ranges[0][1] - ranges[0][0]) * binx,
                                        (errs - ranges[1][0]) /
                                        (ranges[1][1] - ranges[1][0]) * biny),
                                       axis=-1))[0]
            xi, yi = np.mgrid[ranges[0][0]:ranges[0][1]:binx*1j,
                              ranges[1][0]:ranges[1][1]:biny*1j]
            plt.pcolormesh(xi, yi, density, antialiased=True, norm=norm, **kw)
        else:
            plt.hist2d(mags, errs, bins=(binx, biny), norm=norm, **kw)
        pars = self.meta['noise_pars'][band]
        ms = np.linspace(ranges[0][0], ranges[0][1], 500)
        lums = 10**(-0.4*(ms - pars[0]))
        for t in times:
            e_lums = np.sqrt(pars[1]*t*lums + pars[2]*t + pars[3])
            errs = e_lums * (2.5 / np.log(10)) / lums
            plt.plot(ms, errs, 'w')
        plt.xlim(*ranges[0])
        plt.ylim(*ranges[1])
        plt.xlabel(self.meta['mag_names'][band] if 'mag_names' in self.meta
                   else 'magnitude')
        plt.ylabel(self.meta['err_names'][band] if 'err_names' in self.meta
                   else 'error')

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
        _, m_c, s_c = self.meta['nc_pars'][band]
        return log_ndtr((m_c - magnitudes) / s_c)

    def extinguish(self, extinction, apply_completeness=True,
                   update_errors=True):
        """Simulate the effect of extinction and return an updated catalogue.

        Arguments
        ---------
        extinction : float or array-like, shape (n_bands,)
            The extinction to apply for each band, in magnitudes. Must be
            always non-negative. If it is a float, multiply it by
            self.meta['reddening_law'] to obtain the extinction in each band.

        apply_completeness : bool, default to True
            If True, the completeness function is taken into account, and
            random objects that are unlikely to be observable given the added
            extinction are dropped. Requires one to have called the
            `fit_number_counts` method before. If the catalogue has
            probabilities associated, the function updates the probabilities
            and does not perform decimation.

        update_errors : bool, default to False
            If set, errors are also modified to reflect the fact that objects
            are now fainter. Requires one to have called the
            `fit_phot_uncertainties` method before.

        Returns
        -------
        PhotometricCatalogue
            The updated PhotometricCatalogue (self is left untouched).

        """
        cat = self.copy()
        if isinstance(extinction, (float, int)):
            extinction = extinction * self.meta['reddening_law']
        if apply_completeness and cat.meta['nc_pars'] is None:
            # Fit the number counts if this has not been done earlier on
            cat.fit_number_counts()
        if update_errors and cat.meta['noise_pars'] is None:
            # Fit the photometric noise if this has not been done earlier on
            cat.fit_phot_uncertainties()
        for band in range(cat.meta['n_bands']):
            mask = np.where(cat['mag_errs'][:, band] < cat.meta['max_err'])[0]
            if update_errors and extinction[band] > 0:
                pars = cat.meta['noise_pars'][band]
                mags = cat['mags'][mask, band]
                errs = cat['mag_errs'][mask, band]
                # Compute the luminosities
                lums = 10**(-0.4*(mags - pars[0]))
                # and the luminosity errors
                e_lums = errs * lums * (np.log(10) / 2.5)
                # Find out the associated exposure time
                ts = (e_lums**2 - pars[3]) / (pars[1]*lums + pars[2])
                # Update the luminosities for extinction
                lums *= 10**(-0.4*extinction[band])
                # Find the new errors
                e_lums = np.sqrt(pars[1]*ts*lums + pars[2]*ts + pars[3])
                errs = e_lums * (2.5 / np.log(10)) / lums
                cat['mag_errs'][mask, band] = errs
            cat['mags'][mask, band] += extinction[band]
            if apply_completeness and extinction[band] > 0:
                log_completeness_ratio = (cat.log_completeness(band, cat['mags'][mask, band]) -
                                          cat.log_completeness(band, cat['mags'][mask, band] -
                                                               extinction[band]))
                if 'log_probs' in cat.colnames:
                    cat['log_probs'][mask, band] += log_completeness_ratio
                else:
                    removed = log_completeness_ratio < np.log(
                        np.random.uniform(size=len(mask)))
                    cat['mags'][mask[removed], band] = cat.meta['null_mag']
                    cat['mag_errs'][mask[removed], band] = cat.meta['null_err']
        # Now remove objects with errors too large
        if 'log_probs' in cat.colnames and \
            (apply_completeness or update_errors):
            w = np.where(np.sum(cat['mag_errs'] < cat.meta['max_err'],
                                axis=1) >= cat.meta['min_bands'])[0]
            cat = cat[w]
        return cat


class ColorCatalogue(table.Table):
    """Initialize a new color catalogue.

    The initialization can be carried using an arrays of colors and associated
    covariance matrices.

    Parameters
    ----------
    cols : array_like, shape (n_objs, n_cols)
        An array with the colors of all objects.

    col_covs : array_like, shape (n_objs, n_cols, n_cols)
        An array with the color covariance matrices.

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

    @classmethod
    def from_photometric_catalogue(cls, cat, use_projection=True,
                                   dtype=None,
                                   band=None, map_mags=lambda _: _,
                                   extinctions=None, tolerance=1e-5):
        """Compute the colors associate to the current catalogue.

        This operation is performed by subtracting two consecutive bands. For
        this reason, it is advisable to sort the band from the bluest to the
        reddest.

        Arguments
        ---------
        cat : PhotometricCatalogue
            The catalogue to convert into a ColorCatalogue.

        use_projection : bool, default to True
            If True, the procedure sorts bands so that missed bands are
            excluded from the color computation. A projection matrix will be
            returned.

        dtype : data-tipe or None, default None
            The datatype of the color catalogue. If None, the datatype is
            inferred from the original photometric catalogue.

        band : int or None, default to None
            If not None, include in the output a column with a magnitude.
            This is useful in a number of cases. A negative integer is
            interpreted as in the index operator [].

        mag_mags : function, default to identity
            A function used to map the magnitude, used only for the band
            selected (and thus only if band != None). Must accept an array of
            shape (n_objs,) and return an array of shape (n_objs,).

        extinctions : array-like, shape (n_objs,), or None, default to None
            If not None and if `band` is not None, it is an array of values
            that will be *subtracted* to the magnitudes of the band before
            applying map_mags. Used to correct for estinguished magnitudes in
            the `xnicer` code.

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
        if dtype is None:
            dtype = cat['mags'].dtype
        # If we have to work with probabilities, we will make a new catalogue
        # containing all possible combinations of magnitudes. We will
        # associate to each combination a probability that this is realized,
        # and we will call this function again on the all combinations.
        if 'log_probs' in cat.colnames is not None:
            cat2 = cat.copy()
            lp_min = np.log(tolerance)
            lp_max = np.log(1 - tolerance)
            n_objs = len(cat2)
            log_probs = np.zeros(n_objs, dtype=dtype)
            for b in range(cat2.meta['n_bands']):
                # Bands with too small probabilities are directly set to the
                # null value
                w = np.where(cat2['log_probs'][:, b] < lp_min)
                cat2['mags'][w, b] = cat2.meta['null_mag']
                cat2['mag_errs'][w, b] = cat2.meta['null_err']
                # Other bands with intermediate probabilities (and valid band
                # measurements) are duplicated; bands with high probabilities
                # are not duplicated because we assume that the band will
                # always be observed.
                w = np.where((cat2['log_probs'][:, b] > lp_min) & (
                    cat2['log_probs'][:, b] < lp_max))[0]
                cat2 = cat2 + cat2[w]
                log_probs = np.concatenate(
                    (log_probs, log_probs[w] + log1mexp(cat2['log_probs'][w, b])))
                log_probs[w] = log_probs[w] + cat2['log_probs'][w, b]
                n_objs += len(w)
            cat2.remove_column('log_probs')
            # We now filter all objects with too small probabilities or not
            # enough valid vands
            w = np.where((log_probs > lp_min) &
                         (np.sum(cat2['mag_errs'] < cat2.meta['max_err'],
                                 axis=1) >= cat2.meta['min_bands']))[0]
            res = cat2[w].get_colors(use_projection=use_projection, band=band,
                                     map_mags=map_mags, extinctions=extinctions)
            res['log_probs'] = log_probs[w]
            res['log_probs'].description = 'log-probabilities for each object'
            return res

        # Computes the colors
        n_objs = len(cat)
        if band is None:
            n_cols = cat.meta['n_bands'] - 1
        else:
            if band < 0:
                band = cat.meta['n_bands'] + band
            n_cols = cat.meta['n_bands']
        cols = np.zeros((n_objs, n_cols), dtype=dtype)
        col_covs = np.zeros((n_objs, n_cols, n_cols), dtype=dtype)
        for c in range(cat.meta['n_bands'] - 1):
            cols[:, c] = cat['mags'][:, c] - cat['mags'][:, c+1]
            col_covs[:, c, c] = cat['mag_errs'][:, c]**2 + \
                cat['mag_errs'][:, c+1]**2
            if c > 0:
                col_covs[:, c, c-1] = col_covs[:, c-1, c] = - \
                    cat['mag_errs'][:, c]**2
        if band is not None:
            mags = cat['mags'][:, band]
            mag_errs = cat['mag_errs'][:, band]
            if extinctions is not None:
                mags -= extinctions
            cols[:, n_cols - 1] = map_mags(mags)
            diff_map = (map_mags(mags + mag_errs) -
                        map_mags(mags - mag_errs)) / \
                       (2*mag_errs)
            col_covs[:, n_cols - 1, n_cols -
                     1] = (diff_map*mag_errs)**2
            if band < cat.meta['n_bands'] - 1:
                col_covs[:, n_cols-1, band] = col_covs[:, band, n_cols-1] = \
                    diff_map*mag_errs**2
            if band > 0:
                col_covs[:, n_cols-1, band-1] = col_covs[:, band-1, n_cols-1] = \
                    -diff_map*mag_errs**2

        # Use projections
        if use_projection:
            # Projection matrix
            projections = np.zeros((n_objs, n_cols, n_cols), dtype=dtype)
            # Check which magnitudes are usable
            mask = cat['mag_errs'] < cat.meta['max_err']
            csum = np.cumsum(mask, axis=1) - 1
            line = csum.flat
            # Create the other two indices for the projections matrix
            obj, col = np.mgrid[0:n_objs, 0:cat.meta['n_bands']]
            obj = obj.flat
            col = col.flat
            # Set the projections matrix
            w = np.where((line >= 0) & (col < cat.meta['n_bands'] - 1))
            projections[obj[w], line[w], col[w]] = 1
            # Remove the last row of each object if the last band is not
            # available for that object
            last_row = csum[:, -1]
            w = np.where((last_row >= 0) & \
                (last_row < cat.meta['n_bands'] - 1))
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

        res = cls()
        # Copy all relevant metadata
        for key in ['n_bands', 'min_bands', 'mag_names', 'err_names',
                    'class_names']:
            if key in cat.meta:
                res.meta[key] = cat.meta[key]
        # The reddening law changes
        if 'reddening_law' in cat.meta:
            res.meta['reddening_law'] = cat.meta['reddening_law'][:-1] - \
                cat.meta['reddening_law'][1:]
        res.meta['use_projection'] = use_projection
        res.add_column(table.Column(
            cat['idx'], name='idx',
            description='index with respect to the original catalogue',
            format='%d'
        ))
        res.add_column(table.Column(
            cols, name='cols', description='Array of colors',
            unit='mag', format='%6.3f'))
        res.add_column(table.Column(
            col_covs, name='col_covs',
            description='Array of color covariances',
            unit='mag**2', format='%8.6f'))
        if projections is not None:
            res.add_column(table.Column(
                projections, name='projections',
                description='Projection matrices',
                format='%g'))
        if 'log_class_probs' in cat.colnames:
            res['log_class_probs'] = cat['log_class_probs'].copy()
            res.meta['class_names'] = cat.meta['class_names']
        return res

    def __getitem__(self, item):
        """Get item method."""
        if isinstance(item, str):
            if item in self.colnames:
                return self.columns[item]
            else:
                return None
        else:
            return super().__getitem__(item)


class ExtinctionCatalogue(table.Table):
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

    log_class_probs : array-like, shape (n_objs, n_classes)
        Only present if the catalogue has classes. The log-probability of
        each object to belong to a given class.

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

    def __getitem__(self, item):
        """Get item implementation."""
        if isinstance(item, str):
            if item in self.colnames:
                return self.columns[item]
            else:
                return None
        else:
            return super().__getitem__(item)

    def update_(self):
        """Compute the mean_A, variance_A, and log_weight columns.

        The mean_A and variance_A columns are just a single-component
        equivalent of the multi-component extinction.  They are computed using
        a technique identical to the one implemented by merge_components.
        """
        ws = np.exp(self['log_weights'])
        norm = np.sum(ws, axis=-1)
        dtype = self['variances_A'].dtype
        self['log_weight'] = np.log(norm).astype(dtype)
        self['mean_A'] = (np.sum(ws * self['means_A'], axis=-1) / norm).astype(dtype)
        diff_ext = self['means_A'] - self['mean_A'][..., np.newaxis]
        self['variance_A'] = (np.sum(
            ws*(self['variances_A'] + diff_ext**2), axis=-1) / norm).astype(dtype)
        if self.meta['n_colors'] > 0:
            self['mean_c'] = \
                (np.sum(ws[..., np.newaxis] *
                        self['means_c'], axis=-2) / norm[..., np.newaxis]).astype(dtype)
            diff_c = self['means_c'] - self['mean_c'][..., np.newaxis, :]
            self['variance_c'] = (np.sum(
                ws[..., np.newaxis, np.newaxis] *
                (self['variances_c'] +
                 diff_c[..., np.newaxis, :]*diff_c[..., np.newaxis]),
                axis=-3) / norm[..., np.newaxis, np.newaxis]).astype(dtype)
            self['covariance'] = (np.sum(
                ws[..., np.newaxis] *
                (self['covariances'] + diff_ext[..., np.newaxis] * diff_c),
                axis=-2) / norm[..., np.newaxis]).astype(dtype)

    def merge_components(self, merged_components=None):
        """Merge a number of components together.

        Arguments
        ---------
        merged_components : array-like, shape (_,)
            The array containing the components that need to be merged together.
            If not specified, all components are merged.

        """
        # pylint: disable=invalid-name
        if self.meta['n_components'] == 1:
            return self
        if merged_components is None:
            merged_components = np.arange(self.meta['n_components'])
        # Make sure merged_components[0] is the smallest value
        if merged_components[0] >= np.min(merged_components[1:]):
            merged_components = np.sort(merged_components)
        # Extract the relevant values
        ws = np.exp(self['log_weights'][..., merged_components])
        bs = self['means_A'][..., merged_components]
        Vs = self['variances_A'][..., merged_components]
        # Compute the parameters of the merged gaussians
        w = np.sum(ws, axis=-1)
        b = np.sum(ws * bs, axis=-1) / w
        V = np.sum(ws * (Vs + (b[..., np.newaxis] - bs)**2), axis=-1) / w
        # Create the new parameter arrays
        ws = np.delete(self['log_weights'], merged_components[1:], axis=-1)
        bs = np.delete(self['means_A'], merged_components[1:], axis=-1)
        Vs = np.delete(self['variances_A'], merged_components[1:], axis=-1)
        n_components = self.meta['n_components'] - len(merged_components) + 1
        ws[..., merged_components[0]] = np.log(w)
        bs[..., merged_components[0]] = b
        Vs[..., merged_components[0]] = V
        self['log_weights'] = ws
        self['means_A'] = bs
        self['variances_A'] = Vs
        self.meta['n_components'] = n_components
        return self

    def score_samples_components(self, X, Xerr):  # pylint: disable=invalid-name
        """Compute the log of the probability distribution for each component.

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
        T = Xerr[:, np.newaxis] + \
            self['variances_A']  # pylint: disable=invalid-name
        delta = X[:, np.newaxis] - self['means_A']
        return -delta*delta/(2.0*T) - np.log(2.0*np.pi*T) / 2

    def score_samples(self, X, Xerr):  # pylint: disable=invalid-name
        """Compute the log of the probability distribution for all objects.

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
        return logsumexp(self['log_weights'] + log, axis=-1)

    def score_samples_single(self, X, logs=None):  # pylint: disable=invalid-name
        """Compute the log of the probability distribution for all objects.

        Given a specific single extinction value for all objects, this method
        computes the score (that is, the log of the pdf) for that extinction
        for all objects.

        Parameters
        ----------
        X : real
            Input datum.

        Optinal Parameters
        ------------------
        logs : array_like, shape (n_samples, n_components), default None
            The precomputed logarithms of the self['variances_A']. If None,
            it is computed by the function and returned.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in A.

        """
        if logs is None:
            logs = np.log(2*np.pi*self['variances_A']) / 2
        delta = self['means_A'] - X
        return logsumexp(self['log_weights']
                         - delta*delta/(2.0*self['variances_A']) - logs,
                         axis=-1)
