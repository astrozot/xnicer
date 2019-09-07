#cython: language_level=3
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True
#cython: profile=False, linetrace=False, embedsignature=True

from libc.math cimport fabs, sin, floor, ceil, lrint, exp, log, sqrt, erfc, M_PI, M_SQRT2, NAN, isnormal
from cython.parallel cimport prange
cimport cython

import numpy as np
from astropy.io import fits
import os, glob
from scipy.ndimage import uniform_filter

cdef class KDE:
    """
    Class to perform a kernel density estimation.
    
    This class implements all the machinery to efficiently perform kernel
    density estimation. To speed up all computations, the kernel is
    evaluated when the class is constructed on a fine grid (as dictated by
    the `oversampling` parameter, see below). The precomputed kernel is
    then used as a "rubber stamp" centered at each object position.

    Attributes
    ----------
    bandwidth : float
        A quantity that controls the size of the kernel; it is just the
        sigma for a Gaussian smoothing.
        
    kernel : string
        The kernel used. Can be 'gaussian', 'tophat', 'linear', 'epanechinov',
        'exponential', 'cosine'.
        
    kernel_radius : int
        The radius, in pixel, used for the current kernel. It is automatically
        computed given the kernel and the bandwidth. Note that kernels with
        infinite support, such as 'gaussian' and 'exponential', will still be
        truncated to a finite kernel_radius for computation efficiency.

    kernel_size : int
        Computed as kernel_radius*2 + 1, is the total size of a kernel "rubber
        stamp", in each dimension.
        
    binned_kernel : boolean
        If True, the class uses a binned version of the kernel: that is, the
        kernel function is integrated within each pixel. This way pixels are 
        interpreted as "bins" of a histogram, and not as sampling positions.

    metric : float
        The exponent of the p-metric to use. Must be within zero and infinity,
        included. The default corresponds to the Eucledian metric.

    eff_area : float
        The effictive area of the kernel: that is, the equivalent area of a
        tophat kernel that would produce the same variance due to the data
        noise.

    naxis : tuple of ints
        The size of the requested map in the various dimensions, in pixels.
        This has to be fixed when the class is built, but can changed
        later on if necessary using the `set_naxis` method.

    framed_naxis : tuple of ints
        The size of the final map in the various dimensions, in pixels.
        This is larger than naxis, because it includes a "frame" used to speed
        up calculations.

    oversampling : int
        The oversampling factor. Must be an odd integer. High oversampling values
        make the code smaller only for the initialization, but can have an impact
        on the memory consumption.
        
    max_power : int
        The higher power allowed for the kernel when computing the smoothing. One
        generally is only interested to a simple kernel (that is, with power=1);
        however, occasionally one might need to compute a power-two kernel for
        error estimates.
    """
    cdef public:
        str kernel
        double metric, bandwidth, eff_area
        int kernel_radius, kernel_size
        int oversampling, ndim, max_power
        tuple naxis
        tuple framed_naxis
        bint binned_kernel
    cdef:
        float[:,:,:] kernel_table
        long[:] kernel_index

    def __cinit__(self, tuple naxis, double bandwidth=1.0, 
                  str kernel='gaussian', double metric=2.0,
                  bint binned_kernel=True, int oversampling=11, 
                  int max_power=1):
        """
        Cython onstructor for the KDE class.
        """
        self.kernel = kernel
        self.metric = metric
        self.binned_kernel = binned_kernel
        self.naxis = tuple(naxis)
        self.ndim = len(naxis)
        if bandwidth <= 0:
            raise ValueError("The bandwidth parameter must be positive")
        self.bandwidth = bandwidth
        if oversampling % 2 != 1:
            raise ValueError("The oversampling parameter must be an odd integer")
        self.oversampling = oversampling
        self.max_power = max_power

    def __init__(self, naxis, bandwidth=1.0, kernel='gaussian', metric=2.0,
                 binned_kernel=True, oversampling=11, max_power=1):
        """
        Constructor for the KDE class.

        Parameters
        ----------
        naxis : tuple of ints
            The size of the final map in the various dimensions, in pixels.
            This has to be fixed when the class is built, but can changed
            later on if necessary using the `set_naxis` method.

        bandwidth : float, default to 1.0
            A parameter that controls the size of the kernel; it is just the
            sigma for a Gaussian smoothing.
            
        kernel : string, default to 'gaussian'
            The kernel used. Can be 'gaussian', 'tophat', 'linear', 'epanechinov',
            'exponential', 'cosine'.
            
        metric : float, default to 2.0
            The exponent of the p-metric to use. Must be within zero and infinity,
            included. The default corresponds to the Eucledian metric.
            
        binned_kernel : boolean, default to True
            If True, the method uses a binned version of the kernel: that is, the
            kernel function is integrated within each pixel. This way pixels are 
            interpreted as "bins" of a histogram, and not as sampling positions.

        oversampling : int, default to 11
            The oversampling factor. Must be an odd integer. High oversampling values
            make the code smaller only for the initialization, but can have an impact
            on the memory consumption.
            
        max_power : int, default to 1
            The higher power allowed for the kernel when computing the smoothing. One
            generally is only interested to a simple kernel (that is, with power=1);
            however, occasionally one might need to compute a power-two kernel for
            error estimates.
        """
        if self.kernel == 'gaussian':
            self.kernel_radius = lrint(3.0*self.bandwidth + 0.5)
            kernel = lambda x2: np.exp(-x2 / (2*self.bandwidth*2))
        elif self.kernel == 'tophat':
            self.kernel_radius = np.int(np.ceil(self.bandwidth + 0.5))
            kernel = lambda x2: (x2 <= self.bandwidth**2).astype(np.float)
        elif self.kernel == 'linear':
            self.kernel_radius = np.int(np.ceil(self.bandwidth + 0.5))
            kernel = lambda x2: np.maximum(1.0 - np.sqrt(x2) / self.bandwidth, 0.0)
        elif self.kernel == 'epanechinov':
            self.kernel_radius = np.int(np.ceil(self.bandwidth + 0.5))
            kernel = lambda x2: np.maximum(1.0 - x2 / self.bandwidth**2, 0.0)
        elif self.kernel == 'exponential':
            self.kernel_radius = lrint(6.0*self.bandwidth + 0.5)
            kernel = lambda x2: np.exp(-np.sqrt(x2) / self.bandwidth)
        elif self.kernel == 'cosine':
            self.kernel_radius = np.int(np.ceil(self.bandwidth + 0.5))
            kernel = lambda x2: np.maximum(np.cos(np.sqrt(x2) / (2*self.bandwidth/np.pi)), 0.0)
        else:
            raise ValueError('Unknown kernel')
        # Size of the oversampled kernel radius, in each direction
        kernel_over_radius = self.kernel_radius * self.oversampling + self.oversampling // 2
        # The kernel size: twice the kernel radius + one pixel for the center
        self.kernel_size = self.kernel_radius * 2 + 1
        # Compute the framed_naxis for the framing
        self.framed_naxis = tuple(axis + self.kernel_size*2 for axis in self.naxis)
        # Make an oversampled grid
        coords = np.mgrid[(slice(-kernel_over_radius, kernel_over_radius+1),) *
                          self.ndim] / self.oversampling
        # Convert the coords 2D array to 1D using the metric
        if self.metric == 2.0:
            coords = np.sum(coords**2, axis=0)
        elif self.metric == 1.0:
            coords = np.sum(np.abs(coords), axis=0)**2
        elif self.metric == np.inf:
            coords = np.max(np.abs(coords), axis=0)**2
        elif self.metric == 0:
            coords = np.min(np.abs(coords), axis=0)**2
        elif self.metric > 0:
            coords = np.sum((np.abs(coords))**self.metric, axis=0)**(2.0/self.metric) 
        else:
            raise ValueError('The p-metric must have a non-negative exponent')
        # Compute the kernel table on the oversampled area
        kernel_table = kernel(coords)
        # Sum adjacent pixels to perform a sort of pixel integration, if requested
        if self.binned_kernel:
            kernel_table = uniform_filter(kernel_table, size=self.oversampling, mode='constant')
        # Now reorganize the kernel_table and normalize it
        kernel_table = kernel_table.reshape(*(self.kernel_size, self.oversampling) * self.ndim)
        # At the end of the next line, the kernel_table has the dimensions
        # oversampling^ndim x kernel_size^ndim
        kernel_table = kernel_table.transpose(tuple(range(1, 1+2*self.ndim, 2)) + 
                                              tuple(range(0, 2*self.ndim, 2)))
        last = tuple(range(self.ndim, self.ndim*2))
        kernel_table /= \
            np.sum(kernel_table, axis=last)[(Ellipsis,)+(None,)*self.ndim]
        # Check everything is OK
        assert np.allclose(np.sum(kernel_table, axis=last), 1), "Kernel table normalization error"
        self.eff_area = np.mean(1.0 / np.sum(kernel_table*kernel_table, axis=last))
        # Reformat so that it is a 2D array
        kernel_table = kernel_table.reshape(self.oversampling**self.ndim,
                                            self.kernel_size**self.ndim)
        kernel_table_3d = np.empty((self.max_power, self.oversampling**self.ndim, 
                                    self.kernel_size**self.ndim), dtype=np.float32)
        for p in range(self.max_power):
            kernel_table_3d[p, :, :] = (kernel_table**(p+1)).astype(np.float32)
        self.kernel_table = kernel_table_3d
        # Finally build the array index
        self.kernel_index = \
            np.ravel_multi_index(np.mgrid[(slice(self.kernel_size),)*self.ndim],
                                 self.framed_naxis).ravel()

    def __repr__(self):
        return (f"MapMaker({self.naxis)}, "
                f"bandwidth={self.bandwidth}, kernel='{self.kernel}', "
                f"metric={self.metric}, binned_kernel={self.binned_kernel}, "
                f"max_power={self.max_power}, oversampling={self.oversampling})")

    def set_naxis(self, naxis):
        if len(naxis) != self.ndim:
            raise ValueError("Cannot change the number of axes of a KDE class")
        # Update the naxis for the framing
        self.naxis = tuple(naxis)
        self.framed_naxis = tuple(axis + self.kernel_size*2 for axis in self.naxis)
        # Update the kernel index
        self.kernel_index = \
            np.ravel_multi_index(np.mgrid[(slice(self.kernel_size),)*self.ndim],
                                 self.framed_naxis).ravel()
        return self

    def mask_inside(self, *args):
        """
        Return a mask corresponding to all objects that are within the current map.
        
        Parameters
        ----------
        coords : array-like, shape (n_objs, ndim)
            The object coordinates, as provided to `kde`. Alternatively, the
            coordinates can be entered as several parameters, one for each 
            dimension.

        Return
        ------
        mask : array-like, shape (n_objs,)
            The mask of objects within the field: that is, a list of boolean
            values, where True means the object is entirely in the field (and
            also its "stamp" is in the field).
        """
        if len(args) == 0:
            raise ValueError('At least one array must be provided')
        elif len(args) == 1:
            coords = np.rint(args[0])
        else:
            coords = np.rint(np.stack(args, axis=-1))
        return (np.all(coords >= -self.kernel_radius - 1, axis=-1) & 
                np.all(coords < np.array(self.framed_naxis) - 3*self.kernel_radius - 2, axis=-1))


    cpdef kde_float(self, float[:,:] coords, float[:,:] weights=None, short int[:] power=None, 
                    short int[:] mask=None, float[:,:] output=None, bint nocut=False,
                    object callback=None):
        """
        Implementation of the KDE algorithm for arrays of floats.

        See `kde` for details.
        """
        cdef long n_objs = coords.shape[0], obj, dim, plane, i, j, k
        cdef long n_planes = 1, coord_i
        cdef long kernel_full_size = self.kernel_size**self.ndim
        cdef long[:] framed_naxis = np.array(self.framed_naxis, dtype=np.long)
        cdef float coord, delta, weight = 1.0
        cdef float[:,:] result
        
        if weights is not None:
            n_planes = weights.shape[0]
        elif power is not None:
            n_planes = power.shape[0]
        if power is None:
            power = np.zeros(n_planes, dtype=np.short)
        if output is not None:
            result = output
        else:
            result = np.zeros((n_planes, np.prod(self.framed_naxis)), dtype=np.float32)

        for obj in range(n_objs):
            if callback is not None and obj % 16384 == 0:
                callback(self, obj, n_objs)
            if mask is not None and mask[obj] == 0:
                continue
            i = j = 0
            for dim in range(self.ndim):
                coord = coords[obj, dim] + self.kernel_size
                coord_i = lrint(coord)
                delta = coord - coord_i
                if fabs(delta) == 0.5:
                    delta *= 0.999999
                if dim > 0:
                    i *= self.oversampling
                    j *= framed_naxis[dim]
                i += self.oversampling // 2 - lrint(delta * self.oversampling)
                j += coord_i - self.kernel_radius
            for plane in range(n_planes):
                if weights is not None:
                    weight = weights[plane, obj]
                for k in range(kernel_full_size):
                    result[plane, j + self.kernel_index[k]] += \
                        weight * self.kernel_table[power[plane], i, k]
        if callback is not None: 
            callback(self, n_objs, n_objs)
        if output is None:
            if nocut:
                return np.asarray(result).reshape((n_planes,) + self.framed_naxis)
            else:
                return np.asarray(result).reshape((n_planes,) + self.framed_naxis) \
                    [(slice(None),) + (slice(self.kernel_size, -self.kernel_size),)*self.ndim]
        else:
            return output


    cpdef kde_double(self, double[:,:] coords, double[:,:] weights=None, short int[:] power=None, 
                     short int[:] mask=None, double[:,:] output=None, bint nocut=False,
                     object callback=None):
        """
        Implementation of the KDE algorithm for arrays of doubles.

        See `kde` for details.
        """
        cdef long n_objs = coords.shape[0], obj, dim, plane, i, j, k
        cdef long n_planes = 1, coord_i
        cdef long kernel_full_size = self.kernel_size**self.ndim
        cdef long[:] framed_naxis = np.array(self.framed_naxis, dtype=np.long)
        cdef double coord, delta, weight = 1.0
        cdef double[:,:] result
        
        if weights is not None:
            n_planes = weights.shape[0]
        elif power is not None:
            n_planes = power.shape[0]
        if power is None:
            power = np.zeros(n_planes, dtype=np.short)
        if output is not None:
            result = output
        else:
            result = np.zeros((n_planes, np.prod(self.framed_naxis)), dtype=np.float64)

        for obj in range(n_objs):
            if callback is not None and obj % 16384 == 0:
                callback(self, obj, n_objs)
            if mask is not None and mask[obj] == 0:
                continue
            i = j = 0
            for dim in range(self.ndim):
                coord = coords[obj, dim] + self.kernel_size
                coord_i = lrint(coord)
                delta = coord - coord_i
                if fabs(delta) == 0.5:
                    delta *= 0.999999
                if dim > 0:
                    i *= self.oversampling
                    j *= framed_naxis[dim]
                i += self.oversampling // 2 - lrint(delta * self.oversampling)
                j += coord_i - self.kernel_radius
            for plane in range(n_planes):
                if weights is not None:
                    weight = weights[plane, obj]
                for k in range(kernel_full_size):
                    result[plane, j + self.kernel_index[k]] += \
                        weight * self.kernel_table[power[plane], i, k]
        if callback is not None: 
            callback(self, n_objs, n_objs)
        if output is None:
            if nocut:
                return np.asarray(result).reshape((n_planes,) + self.framed_naxis)
            else:
                return np.asarray(result).reshape((n_planes,) + self.framed_naxis) \
                    [(slice(None),) + (slice(self.kernel_size, -self.kernel_size),)*self.ndim]
        else:
            return output


    def kde(self, coords, weights=None, power=None, mask=None, output=None,
            nocut=False, callback=None):
        """Perform the KDE computation.
        
        This procedure computes a KDE map given a set of points and associated weights.
        
        coords : array-like, shape (n_objs, ndim)
            An array with the coordinate of each object along each axis.

        weights : array-like, shape (n_planes, n_objs), default to 1.0.
            A list of weights for each object. Different weights can be used
            for different planes.

        power : array-like, shape (n_planes,), default to 0
            A list of integer values interpreted as exponents for the kernel
            function. Note that the effective exponent is given by power+1.

        mask : array-like, shape (n_objs,), default no mask
            If provided, must be a list of boolean values that denotes the
            objects that can be used for the analysis.
            
        output : array-like, shape (n_planes, prod(self.framed_naxis)), default to None
            If not None, the output grid is not initialized but the provided
            array is used instead. The same, updated array is also returned as
            output.

        nocut : boolean, default to False
            If True, the final out produced is larger than the provided naxis
            by 2*kernel_radius. This is useful if the output parameter is used,
            to avoid allocating the same array several times. Note that the
            framed_naxis parameters of the class will report the *larger*
            field, that is the one that includes the frame.

        callback : function
            A function to call to update the current status. Must accept three
            arguments: (self, object number, total number of objects)
            
        Returns
        -------
        If output is None:
        
        array-like, shape (n_planes, naxis[0], ..., naxis[ndim-1])
            A set of images with the KDE associated to each weight.
            
        The naxis values can be self.framed_naxis (if nocut=True) or
        self.naxis (if nocut=False). If output is not None:
        
        array-like, shape (n_planes, prod(framed_naxis))
            A set of images with the KDE associated to each weight, compressed
            along the various coordinates. To recover the real output use
            
            result.reshape((n_planes,) + tuple(self.framed_naxis))
        """
        if coords.dtype == np.float64 or coords.dtype == np.float128:
            float_type = np.float64
        else:
            float_type = np.float32
        coords = coords.astype(float_type, copy=False)
        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            weights = weights.astype(float_type, copy=False)
            nplanes = weights.shape[0]
        else:
            nplanes = 1
        if power is not None:
            if hasattr(power, '__iter__'):
                power = np.array(power, dtype=np.short)
            else:
                power = np.full((nplanes,), power, dtype=np.short)
        if mask is not None:
            mask = mask.astype(np.short, copy=False)
        if output is not None:
            if output.ndim == 1:
                output = output.reshape(1, -1)
            output = output.astype(float_type, copy=False)
        if float_type == np.float32:
            return self.kde_float(coords, weights, power, mask, output, nocut,
                                  callback)
        else:
            return self.kde_double(coords, weights, power, mask, output, nocut,
                                   callback)
        

