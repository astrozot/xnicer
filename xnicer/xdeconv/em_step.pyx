# Compile with: python setup.py build_ext --use-cython --inplace
# cython: language_level=3, embedsignature=True
# cython: wraparound=False, boundscheck=False, initializedcheck=False, cdivision=True

# Profiling options: disabled
# _cython: linetrace=True, binding=True
# _distutils: define_macros=CYTHON_TRACE_NOGIL=1

import cython
from cython.parallel import prange, parallel

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport exp, log, isinf
import numpy as np
cimport numpy as np
from numpy cimport float64_t, float32_t, uint8_t
from numpy.math cimport INFINITY, PI
from scipy.linalg.cython_lapack cimport dpotrf, dtrtri
from scipy.linalg.cython_lapack cimport spotrf, strtri
from scipy.linalg.cython_blas cimport dtrmv, dtrmm, ddot, dgemv, dgemm, dsyrk, dsyr, dsymm
from scipy.linalg.cython_blas cimport strmv, strmm, sdot, sgemv, sgemm, ssyrk, ssyr, ssymm

# Single precision libc functions are not exported
cdef extern from "<math.h>" nogil:
    double expf(double x)
    double logf(double x)

ctypedef float64_t DOUBLE
ctypedef float32_t FLOAT
ctypedef uint8_t BOOL

cdef int _FIX_NONE = 0
cdef int _FIX_AMP = 1
cdef int _FIX_MEAN = 2
cdef int _FIX_COVAR = 4
cdef int _FIX_CLASS = 8
cdef int _FIX_AMPCLASS = _FIX_AMP + _FIX_CLASS
cdef int _FIX_ALL = _FIX_AMP + _FIX_MEAN + _FIX_COVAR + _FIX_CLASS

FIX_NONE = _FIX_NONE
FIX_AMP = _FIX_AMP
FIX_MEAN = _FIX_MEAN
FIX_COVAR = _FIX_COVAR
FIX_CLASS = _FIX_CLASS
FIX_AMPCLASS = _FIX_AMPCLASS
FIX_ALL = _FIX_ALL


cpdef omp_test1(n):
    cdef int sum = 0, i, imax
    imax = <int>n
    for i in prange(imax, nogil=True):
        sum += 1
    return sum


cpdef omp_test2(n):
    cdef int sum = 0, i, imax
    imax = <int>n
    with nogil, parallel():
        for i in prange(imax):
            sum += 1
    return sum


cdef double logsumexp_d(double *x, int n) nogil:
    """Computes the log of the sum of the exponentials."""
    if n == 0:
        return -INFINITY
    elif n == 1:
        return x[0]
    cdef int i
    cdef double result = 0.0
    cdef double xmax = x[0]
    for i in range(1, n):
        if x[i] > xmax:
            xmax = x[i]
    if isinf(xmax):
        return xmax
    for i in range(n):
        result += exp(x[i] - xmax)
    return xmax + log(result)


cdef float logsumexp_s(float *x, int n) nogil:
    """Computes the log of the sum of the exponentials."""
    if n == 0:
        return -INFINITY
    elif n == 1:
        return x[0]
    cdef int i
    cdef float result = 0.0
    cdef float xmax = x[0]
    for i in range(1, n):
        if x[i] > xmax:
            xmax = x[i]
    if isinf(xmax):
        return xmax
    for i in range(n):
        result += expf(x[i] - xmax)
    return xmax + logf(result)


cpdef double log_likelihoods_d(double[:,:] deltas, double[:,:,:] covars, double[::1] results=None):
    """
    Compute the log-likelihood for a set of data.

    Parameters
    ----------
    deltas : array-like, shape(n, r)
        The differences between the means of the normal distributions and
        the datapoints.

    covars : array-like, shape(n, r, r)
        The covariances of the normal distributions.

    Output Parameters
    -----------------
    results : array-like, shape(n)
        The computed log-likelihoods for each multivariate normal
        distribution. If set to None, no value is returned.

    """
    cdef int n, i, j, info
    cdef int nobjs=deltas.shape[0]
    cdef int r=covars.shape[2]
    cdef int inc=1
    cdef double* C
    cdef double* d
    cdef double* res=NULL
    cdef double result=0, norm=-r * log(2.0*PI) / 2.0

    C = <double *>malloc(r*r*sizeof(DOUBLE))
    d = <double *>malloc(r*sizeof(DOUBLE))
    if results is None:
        res = <double *>malloc(nobjs*sizeof(DOUBLE))
        results = <double[:nobjs]> res
    for n in range(nobjs):
        # Copy of deltas and covars
        for i in range(r):
            d[i] = deltas[n, i]
            for j in range(i+1):
                C[i*r+j] = C[j*r+i] = covars[n, i, j]
        # Chowlesky decomposition of C
        dpotrf("U", &r, C, &r, &info)
        # Inverse of the upper triangular C
        dtrtri("U", "N", &r, C, &r, &info)
        # Computes x := C^T*delta
        dtrmv("U", "T", "N", &r, C, &r, d, &inc)
        # Computes the log of the determinant of Tnew
        result = norm
        for i in range(r):
            result += log(C[i*r + i])
        # Computes log(det C) - x^T*x / 2
        results[n] = result - 0.5*ddot(&r, d, &inc, d, &inc)
    free(C)
    free(d)
    if res != NULL:
        free(res)


cpdef float log_likelihoods_s(float[:,:] deltas, float[:,:,:] covars, float[::1] results=None):
    """
    Compute the log-likelihood for a set of data.

    Parameters
    ----------
    deltas : array-like, shape(n, r)
        The differences between the means of the normal distributions and
        the datapoints.

    covars : array-like, shape(n, r, r)
        The covariances of the normal distributions.

    Output Parameters
    -----------------
    results : array-like, shape(n)
        The computed log-likelihoods for each multivariate normal
        distribution. If set to None, no value is returned.

    """
    cdef int n, i, j, info
    cdef int nobjs=deltas.shape[0]
    cdef int r=covars.shape[2]
    cdef int inc=1
    cdef float* C
    cdef float* d
    cdef float* res=NULL
    cdef float result=0, norm=-r * logf(2.0*PI) / 2.0

    C = <float *>malloc(r*r*sizeof(FLOAT))
    d = <float *>malloc(r*sizeof(FLOAT))
    if results is None:
        res = <float *>malloc(nobjs*sizeof(FLOAT))
        results = <float[:nobjs]> res
    for n in range(nobjs):
        # Copy of deltas and covars
        for i in range(r):
            d[i] = deltas[n, i]
            for j in range(i+1):
                C[i*r+j] = C[j*r+i] = covars[n, i, j]
        # Chowlesky decomposition of C
        spotrf("U", &r, C, &r, &info)
        # Inverse of the upper triangular C
        strtri("U", "N", &r, C, &r, &info)
        # Computes x := C^T*delta
        strmv("U", "T", "N", &r, C, &r, d, &inc)
        # Computes the log of the determinant of Tnew
        result = norm
        for i in range(r):
            result += logf(C[i*r + i])
        # Computes log(det C) - x^T*x / 2
        results[n] = result - 0.5*sdot(&r, d, &inc, d, &inc)
    free(C)
    free(d)
    if res != NULL:
        free(res)

cdef int e_single_step_noproj_d(double[::1] w, double[::1,:] S,
        double[::1] m, double[::1,:] V,
        double* x, double* T, double* VRt,
        double* q, double* b, double* B) nogil:
    # Perform a single E-step for the extreme deconvolution.
    #
    # This function is identical to e_single_step, with the difference that
    # the projection is taken to be an identity matrix.
    #
    # Input data
    # w: (r,), single observation
    # S: (r, r), covariance of the observational data w
    # m: (r,), single center of multivariate Gaussian
    # V: (r, r), single covariance matrix of the multivariate Gaussian
    #
    # Temporary data
    # These are taken to be simple double* pointer with enough allocated momery.
    # The sizes are
    # x: (r,), difference = w - m
    # T: (r, r), combined projected covariance = V + S
    # VRt: (r, r), product sqrt(T)*V
    # As a result, incx = 1, ldT = r, and ldRV = r.
    #
    # Output data, also double * with allocated memory. These are essentially
    # given by Eq. (25) of the original paper, with the important difference
    # that b is here in reality b_ij - m_j. This ensures better numerical
    # stability.
    # q: (1)
    # b: (r,)
    # B: (r, r)
    cdef int r=w.shape[0]
    cdef int ldV=V.strides[1] // sizeof(DOUBLE)
    cdef int ldVRt=r
    cdef int ldT=r
    cdef int ldB=r
    cdef int incm=m.strides[0] // sizeof(DOUBLE)
    cdef int incx=1
    cdef int incb=1
    cdef int info, inc=1, n1, n2
    cdef double zero=0.0, one=1.0, _one=-1.0, a
    # Computes T <- V + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = V[n2,n1] + S[n2,n1]
    # Computes x <- w - m
    for n1 in range(r):
        x[n1] = w[n1] - m[n1]
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    dpotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    dtrtri("U", "N", &r, T, &ldT, &info)
    # Computes x <- Tnew^T*x
    dtrmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes VRt <- V*Tnew
    for n1 in range(r):
        for n2 in range(r):
            VRt[n1*r+n2] = V[n2,n1]
    dtrmm("R", "U", "N", "N", &r, &r, &one, T, &ldT, VRt, &ldVRt)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += log(T[n1*r+n1])
    # Computes q <- log(det Tnew) - x^T*x / 2)
    q[0] = a - 0.5*ddot(&r, x, &incx, x, &incx)
    # Computes b <- VRt*x (the +m term has been dropped)
    for n1 in range(r):
        # b[n1] = m[n1]
        b[n1] = 0.0
    dgemv("N", &r, &r, &one, VRt, &ldVRt, x, &incx, &one, b, &incb)
    # Computes B <- V - RV^T*RV
    for n1 in range(r):
        for n2 in range(r):
            B[n1*r+n2] = V[n2,n1]
    dsyrk("U", "N", &r, &r, &_one, VRt, &ldVRt, &one, B, &ldB)
    for n1 in range(r):
        for n2 in range(n1):
            B[n2*r+n1] = B[n1*r+n2]
    return 0


cdef int e_single_step_noproj_s(float[::1] w, float[::1,:] S,
        float[::1] m, float[::1,:] V,
        float* x, float* T, float* VRt,
        float* q, float* b, float* B) nogil:
    # Perform a single E-step for the extreme deconvolution.
    #
    # This function is identical to e_single_step, with the difference that
    # the projection is taken to be an identity matrix.
    #
    # Input data
    # w: (r,), single observation
    # S: (r, r), covariance of the observational data w
    # m: (r,), single center of multivariate Gaussian
    # V: (r, r), single covariance matrix of the multivariate Gaussian
    #
    # Temporary data
    # These are taken to be simple float* pointer with enough allocated momery.
    # The sizes are
    # x: (r,), difference = w - m
    # T: (r, r), combined projected covariance = V + S
    # VRt: (r, r), product sqrt(T)*V
    # As a result, incx = 1, ldT = r, and ldRV = r.
    #
    # Output data, also float * with allocated memory. These are essentially
    # given by Eq. (25) of the original paper, with the important difference
    # that b is here in reality b_ij - m_j. This ensures better numerical
    # stability.
    # q: (1)
    # b: (r,)
    # B: (r, r)
    cdef int r=w.shape[0]
    cdef int ldV=V.strides[1] // sizeof(FLOAT)
    cdef int ldVRt=r
    cdef int ldT=r
    cdef int ldB=r
    cdef int incm=m.strides[0] // sizeof(FLOAT)
    cdef int incx=1
    cdef int incb=1
    cdef int info, inc=1, n1, n2
    cdef float zero=0.0, one=1.0, _one=-1.0, a
    # Computes T <- V + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = V[n2,n1] + S[n2,n1]
    # Computes x <- w - m
    for n1 in range(r):
        x[n1] = w[n1] - m[n1]
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    spotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    strtri("U", "N", &r, T, &ldT, &info)
    # Computes x <- Tnew^T*x
    strmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes VRt <- V*Tnew
    for n1 in range(r):
        for n2 in range(r):
            VRt[n1*r+n2] = V[n2,n1]
    strmm("R", "U", "N", "N", &r, &r, &one, T, &ldT, VRt, &ldVRt)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += logf(T[n1*r+n1])
    # Computes q <- log(det Tnew) - x^T*x / 2)
    q[0] = a - 0.5*sdot(&r, x, &incx, &x[0], &incx)
    # Computes b <- VRt*x (the +m term has been dropped)
    for n1 in range(r):
        # b[n1] = m[n1]
        b[n1] = 0.0
    sgemv("N", &r, &r, &one, VRt, &ldVRt, x, &incx, &one, b, &incb)
    # Computes B <- V - RV^T*RV
    for n1 in range(r):
        for n2 in range(r):
            B[n1*r+n2] = V[n2,n1]
    ssyrk("U", "N", &r, &r, &_one, VRt, &ldVRt, &one, B, &ldB)
    for n1 in range(r):
        for n2 in range(n1):
            B[n2*r+n1] = B[n1*r+n2]
    return 0


cdef int e_single_step_d(double[::1] w, double[::1,:] Rt, double[::1,:] S,
                         double[::1] m, double[::1,:] V,
                         double* x, double* T, double* VRt,
                         double* q, double* b, double* B) nogil:
    # Perform a single E-step for the extreme deconvolution.
    #
    # Input data
    # w: (r,), single observation
    # Rt: (d, r), single projection matrix
    # S: (r, r), covariance of the observational data w
    # m: (d,), single center of multivariate Gaussian
    # V: (d, d), single covariance matrix of the multivariate Gaussian
    #
    # Temporary data
    # These are taken to be simple double* pointer with enough allocated momery.
    # The sizes are
    # x: (r,), difference = w - R*m
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    # As a result, incx = 1, ldT = r, and ldVRt = d.
    #
    # Output data, also double * with allocated memory. These are essentially
    # given by Eq. (25) of the original paper, with the important difference
    # that b is here in reality b_ij - m_j. This ensures better numerical
    # stability.
    # q: (1)
    # b: (d,)
    # B: (d, d)
    cdef int r=w.shape[0], d=Rt.shape[0]
    cdef int ldV=V.strides[1] // sizeof(DOUBLE)
    cdef int ldRt=Rt.strides[1] // sizeof(DOUBLE)
    cdef int ldVRt=d
    cdef int ldT=r
    cdef int ldB=d
    cdef int incm=m.strides[0] // sizeof(DOUBLE)
    cdef int incx=1
    cdef int incb=1
    cdef int info, inc=1, n1, n2
    cdef double zero=0.0, one=1.0, _one=-1.0, a
    # Computes VRt <- VRt
    dsymm("L", "U", &d, &r, &one, &V[0,0], &ldV, &Rt[0,0], &ldRt, &zero, VRt, &ldVRt)
    # Computes T <- R*V*Rt + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = S[n2,n1]
    dgemm("T", "N", &r, &r, &d, &one, &Rt[0,0], &ldRt, VRt, &ldVRt, &one, T, &ldT)
    # Computes x <- w - R*m
    for n1 in range(r):
        x[n1] = w[n1]
    dgemv("T", &d, &r, &_one, &Rt[0,0], &ldRt, &m[0], &incm, &one, x, &incx)
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    dpotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    dtrtri("U", "N", &r, T, &ldT, &info)
    # Computes x <- Tnew^T*x
    dtrmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += log(T[n1*r+n1])
    # Computes q <- log(det Tnew) - x^T*x / 2
    q[0] = a - 0.5*ddot(&r, x, &incx, x, &incx)
    # Computes VRt <- VRt*Tnew = V*Rt*Tnew
    dtrmm("R", "U", "N", "N", &d, &r, &one, T, &ldT, VRt, &ldVRt)
    # Computes b <- VRt*x (the +m term has been dropped)
    for n1 in range(d):
        # b[n1] = m[n1]
        b[n1] = 0.0
    dgemv("N", &d, &r, &one, VRt, &ldVRt, x, &incx, &one, b, &incb)
    # Computes B <- V - (VRt)*(VRt)^T
    for n1 in range(d):
        for n2 in range(d):
            B[n1*d+n2] = V[n2,n1]
    dsyrk("U", "N", &d, &r, &_one, VRt, &ldVRt, &one, B, &ldB)
    for n1 in range(d):
        for n2 in range(n1):
            B[n2*d+n1] = B[n1*d+n2]
    return 0


cdef int e_single_step_s(float[::1] w, float[::1,:] Rt, float[::1,:] S,
                         float[::1] m, float[::1,:] V,
                         float* x, float* T, float* VRt,
                         float* q, float* b, float* B) nogil:
    # Perform a single E-step for the extreme deconvolution.
    #
    # Input data
    # w: (r,), single observation
    # Rt: (d, r), single projection matrix
    # S: (r, r), covariance of the observational data w
    # m: (d,), single center of multivariate Gaussian
    # V: (d, d), single covariance matrix of the multivariate Gaussian
    #
    # Temporary data
    # These are taken to be simple float* pointer with enough allocated momery.
    # The sizes are
    # x: (r,), difference = w - R*m
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    # As a result, incx = 1, ldT = r, and ldVRt = d.
    #
    # Output data, also float * with allocated memory. These are essentially
    # given by Eq. (25) of the original paper, with the important difference
    # that b is here in reality b_ij - m_j. This ensures better numerical
    # stability.
    # q: (1)
    # b: (d,)
    # B: (d, d)
    cdef int r=w.shape[0], d=Rt.shape[0]
    cdef int ldV=V.strides[1] // sizeof(FLOAT)
    cdef int ldRt=Rt.strides[1] // sizeof(FLOAT)
    cdef int ldVRt=d
    cdef int ldT=r
    cdef int ldB=d
    cdef int incm=m.strides[0] // sizeof(FLOAT)
    cdef int incx=1
    cdef int incb=1
    cdef int info, inc=1, n1, n2
    cdef float zero=0.0, one=1.0, _one=-1.0, a
    # Computes VRt <- VRt
    ssymm("L", "U", &d, &r, &one, &V[0,0], &ldV, &Rt[0,0], &ldRt, &zero, VRt, &ldVRt)
    # Computes T <- R*V*Rt + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = S[n2,n1]
    sgemm("T", "N", &r, &r, &d, &one, &Rt[0,0], &ldRt, VRt, &ldVRt, &one, T, &ldT)
    # Computes x <- w - R*m
    for n1 in range(r):
        x[n1] = w[n1]
    sgemv("T", &d, &r, &_one, &Rt[0,0], &ldRt, &m[0], &incm, &one, x, &incx)
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    spotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    strtri("U", "N", &r, T, &ldT, &info)
    # Computes x <- Tnew^T*x
    strmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += logf(T[n1*r+n1])
    # Computes q <- log(det Tnew) - x^T*x / 2
    q[0] = a - 0.5*sdot(&r, x, &incx, x, &incx)
    # Computes VRt <- VRt*Tnew = V*Rt*Tnew
    strmm("R", "U", "N", "N", &d, &r, &one, T, &ldT, VRt, &ldVRt)
    # Computes b <- VRt*x (the +m term has been dropped)
    for n1 in range(d):
        # b[n1] = m[n1]
        b[n1] = 0.0
    sgemv("N", &d, &r, &one, VRt, &ldVRt, x, &incx, &one, b, &incb)
    # Computes B <- V - (VRt)*(VRt)^T
    for n1 in range(d):
        for n2 in range(d):
            B[n1*d+n2] = V[n2,n1]
    ssyrk("U", "N", &d, &r, &_one, VRt, &ldVRt, &one, B, &ldB)
    for n1 in range(d):
        for n2 in range(n1):
            B[n2*d+n1] = B[n1*d+n2]
    return 0


cpdef int py_e_single_step(double[::1] w, double[:,::1] R, double[:,::1] S,
                           double[::1] m, double[:,::1] V,
                           double[::1] q, double[::1] b, double[:,::1] B):
    """
    Pure Python version of e_single_step.

    Perform a single E-step for the extreme deconvolution.

    Parameters
    ----------
    w: array-like, shape (r,)
        Single observation.

    Rt: array-like, shape (d, r)
        Transpose of a single projection matrix.

    S: array like, shape (r, r)
        Covariance of the observational data w.

    m: array-likem shape (d,)
        Single center of multivariate Gaussian.

    V: array-like, shape (d, d)
        Single covariance matrix of the multivariate Gaussian.

    Output parameters
    -----------------
    These parameters must be NumPy array of the correct shape and will be
    filled with the results of the computation.  These are essentially
    given by Eq. (25) of the original paper, with the important difference
    that b is here in reality b_ij - m_j. This ensures better numerical
    stability.

    q: array-like, shape (1,)
        The q parameter.

    b: array-like, shape (d,)
        The b vector.

    B: array-like, shape (d, d)
        The B matrix.

    """
    cdef double *x
    cdef double *T
    cdef double *VRt
    w_f = np.asfortranarray(w)
    if R is not None:
        Rt_f = np.asfortranarray(R.T)
    else:
        Rt_f = None
    S_f = np.asfortranarray(S)
    m_f = np.asfortranarray(m)
    V_f = np.asfortranarray(V)
    r = w.shape[0]
    d = m.shape[0]
    if R is not None:
        assert R.shape[1] == d, "R mismatch 1"
        assert R.shape[0] == r, "R mismatch 0"
    assert S.shape[0] == r and S.shape[1] == r, "S mismatch"
    assert V.shape[0] == d and V.shape[1] == d, "V mismatch"
    assert q.shape[0] == 1, "q mismatch"
    assert b.shape[0] == d, "b mismatch"
    assert B.shape[0] == d and B.shape[1] == d, "B mismatch"
    x = <double *>malloc(r*sizeof(DOUBLE))
    T = <double *>malloc(r*r*sizeof(DOUBLE))
    VRt = <double *>malloc(d*r*sizeof(DOUBLE))
    if R is not None:
        e_single_step_d(w_f, Rt_f, S_f, m_f, V_f, x, T, VRt, &q[0], &b[0], &B[0,0])
    else:
        e_single_step_noproj_d(w_f, S_f, m_f, V_f, x, T, VRt, &q[0], &b[0], &B[0,0])
    free(x)
    free(T)
    free(VRt)

cdef int predict_single_d(double[::1] w, double[::1,:] Rt, double[::1,:] S,
                          double[::1] m, double[::1,:] V, double[::1] k,
                          double* x, double *Rk, double* T, double* VRt,
                          double* Amean, double* Avar, double* Aweight,
                          double* Wmean, double *Wcov, double* Wvar) nogil:
    # Perform a single prediction for the extreme deconvolution.
    #
    # Input data
    # w: (r,), single observation
    # Rt: (d, r), single projection matrix
    # S: (r, r), covariance of the observational data w
    # m: (d,), single center of multivariate Gaussian
    # V: (d, d), single covariance matrix of the multivariate Gaussian
    # k: (d,), projection vector
    #
    # Temporary data
    # These are taken to be simple double* pointer with enough allocated momery.
    # The sizes are
    # x: (r,), difference = w - R*m
    # Rk: (r,), product Rk = R*k
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    # As a result, incx = 1, ldT = r, and ldVRt = d.
    #
    # Output data, also double * with allocated memory. These are 
    # Amean: (1,)
    # Avar: (1,)
    # Aweight: (1,)
    # Wmean: (d,)
    # Wcov: (d,) 
    # Wvar: (d,d)
    cdef int r=w.shape[0], d=Rt.shape[0]
    cdef int ldV=V.strides[1] // sizeof(DOUBLE)
    cdef int ldRt=Rt.strides[1] // sizeof(DOUBLE)
    cdef int ldVRt=d
    cdef int ldT=r
    cdef int ldB=d
    cdef int incm=m.strides[0] // sizeof(DOUBLE)
    cdef int incx=1
    cdef int incRk=1
    cdef int incb=1
    cdef int inck=k.strides[0] // sizeof(DOUBLE)
    cdef int incWcov=1
    cdef int ldWvar=d
    cdef int incWmean=1
    cdef int info, inc=1, n1, n2
    cdef double zero=0.0, one=1.0, _one=-1.0, a, xRk

    # Computes VRt <- V*Rt
    dsymm("L", "U", &d, &r, &one, &V[0,0], &ldV, &Rt[0,0], &ldRt, &zero, VRt, &ldVRt)
    # Computes T <- R*V*Rt + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = S[n2,n1]
    dgemm("T", "N", &r, &r, &d, &one, &Rt[0,0], &ldRt, VRt, &ldVRt, &one, T, &ldT)
    # Computes x <- w - R*m
    for n1 in range(r):
        x[n1] = w[n1]
    dgemv("T", &d, &r, &_one, &Rt[0,0], &ldRt, &m[0], &incm, &one, x, &incx)
    # Computes Rk <- R*k
    dgemv("T", &d, &r, &one, &Rt[0,0], &ldRt, &k[0], &inck, &zero, Rk, &incRk)
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    dpotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    dtrtri("U", "N", &r, T, &ldT, &info)
    # Computes x <- Tnew^T*x
    dtrmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes Rk <- Tnew^T*Rk
    dtrmv("U", "T", "N", &r, T, &ldT, Rk, &incRk)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += log(T[n1*r+n1])
    # Computes Avar <- 1 / Rk^T*Rk
    Avar[0] = 1.0 / ddot(&r, Rk, &incRk, Rk, &incRk)
    # Computes xRk <- x^T*Rk
    xRk = ddot(&r, x, &incx, Rk, &incRk)
    # Computes Amean <- Rk^T*x * Avar
    Amean[0] = xRk * Avar[0]
    # Computes Aweight <- log(Tnew) - x^T*x / 2 + xRk / (2 * ivar)
    Aweight[0] = a - (ddot(&r, x, &incx, x, &incx) - xRk * xRk * Avar[0]) * 0.5
    # Stop here if no full computations are required!
    if Wcov != NULL:
        # Computes VRt <- VRt*Tnew = V*Rt*Tnew
        dtrmm("R", "U", "N", "N", &d, &r, &one, T, &ldT, VRt, &ldVRt)
        # Computes Wcov <- VRt*Rk = V*Rt*Tnew*Tnew^T*Rk
        dgemv("N", &d, &r, &one, VRt, &ldVRt, Rk, &inck, &zero, Wcov, &incWcov)
        # Computes Wvar <- V - (VRt)*(VRt)^T + Wcov*Wcov^T * Avar
        for n1 in range(d):
            for n2 in range(d):
                Wvar[n1*d+n2] = V[n2,n1]
        dsyrk("U", "N", &d, &r, &_one, VRt, &ldVRt, &one, Wvar, &ldWvar)
        dsyr("U", &d, &Avar[0], Wcov, &incWcov, Wvar, &ldWvar)
        for n1 in range(d):
            for n2 in range(n1):
                Wvar[n2*d+n1] = Wvar[n1*d+n2]
        # Computes Wmean <- m - VRt*x - Wcov*Amean
        for n1 in range(d):
            Wmean[n1] = m[n1] - Wcov[n1]*Amean[0]
        dgemv("N", &d, &r, &one, VRt, &ldVRt, x, &incx, &one, Wmean, &incWmean)
        # Rescale Wcov <- -Wcov * Avar
        for n1 in range(d):
            Wcov[n1] *= -Avar[0]
    return 0


@cython.binding(True)
cpdef double predict_d(double[::1,:] w, double[::1,:,:] S,
                       double[::1,:] m, double[::1,:,:] V, double[::1] kvec,
                       double[::1,:] Amean, double[::1,:] Avar, double[::1,:] Aweight,
                       double[::1,:,:] Rt=None, 
                       double[::1,:,:] Wmean=None, double[::1,:,:] Wcov=None,
                       double[::1,:,:,:] Wvar=None):
    """
    Perform a vector prediction.

    Given an extreme deconvolution, this procedure computes an interence for A
    along the line w - kvec * A. Optionally, it also returns the original w 
    (and the associated errors)

    Parameters
    ----------
    Note: all array parameters are expected to be provided as Fortran
    contiguous arrays.

    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions

    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.

    m: array-like, shape (d, k)
        Centers of multivariate Gaussians.

    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians.

    Output parameters
    -----------------
    All arrays must be pre-allocated in Fortran style.

    Amean: array-like, shape (k, n)
        Centers of the inferred A Gaussians.

    Avar: array-like, shape (k, n)
        Variances of the inferred A Gaussians.

    Aweight: array-like, shape (k, n)
        Weights oof the inferred A Gaussians.

    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into
        the observed r-dimensional vector. If None, it is assumed that r=d
        and that no project is performed (equivalently: R is an array if
        identity matrices).

    Optional Outputs
    ----------------
    Wmean: array-like, shape (c, k, n)
        Array with the inferred means for the original w values.

    Wcov: array-like, shape (c, k, n)
        Array with the inferred covariances between the original w values and
        the inferred A value.

    Wvar: array-like, shape (c, c, k, n)
        Array with the inferred variances for original w values.
    """
    # Temporary data for the prediction-step
    # x: (r,), difference = w - R*m
    # Rk: (r,), product Rk = R*k
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    cdef int r=w.shape[0], n=w.shape[1], d=m.shape[0], k=m.shape[1]
    cdef int i, j 
    cdef double norm = d * log(2*PI) / 2.0
    # All these are local variables
    cdef double *x
    cdef double *Rk
    cdef double *T
    cdef double *VRt
    cdef bint full

    full = (Wmean != None) and (Wcov != None) and (Wvar != None)
    with nogil, parallel():
        # Allocates the block-local variables
        x = <double *>malloc(r*sizeof(double))
        Rk = <double *>malloc(r*sizeof(double))
        T = <double *>malloc(r*r*sizeof(double))
        VRt = <double *>malloc(r*d*sizeof(double))
        for i in prange(n, schedule='static'):
            # prediction step for i (object number) fixed
            for j in range(k):
                if Rt is None:
                    predict_single_d(w[:,i], w, S[:,:,i],
                                     m[:,j], V[:,:,j], kvec,
                                     x, Rk, T, VRt,
                                     &Amean[j,i], &Avar[j,i], &Aweight[j,i],
                                     NULL, NULL, NULL)
                else:
                    predict_single_d(w[:,i], Rt[:,:,i], S[:,:,i],
                                     m[:,j], V[:,:,j], kvec,
                                     x, Rk, T, VRt,
                                     &Amean[j,i], &Avar[j,i], &Aweight[j,i],
                                     &Wmean[0,j,i] if full else NULL, 
                                     &Wcov[0,j,i] if full else NULL, 
                                     &Wvar[0,0,j,i] if full else NULL)
                Aweight[j,i] = Aweight[j,i] - norm + log(2*PI * Avar[j,i]) / 2.0
        # Free the memory for block-local variables
        free(x)
        free(Rk)
        free(T)
        free(VRt)


cpdef int py_predict_single(double[::1] w, double[:,::1] R, double[:,::1] S,
                            double[::1] m, double[:,::1] V, double[::1] k,
                            double[::1] p, double[::1] var, double[::1] ev,
                            double[::1] M=None, double[::1] C=None,
                            double[:,::1] W=None):
    """
    #FIXME
    Pure Python version of e_single_step.

    Perform a single E-step for the extreme deconvolution.

    Parameters
    ----------
    w: array-like, shape (r,)
        Single observation.

    Rt: array-like, shape (d, r)
        Transpose of a single projection matrix.

    S: array like, shape (r, r)
        Covariance of the observational data w.

    m: array-likem shape (d,)
        Single center of multivariate Gaussian.

    V: array-like, shape (d, d)
        Single covariance matrix of the multivariate Gaussian.

    Output parameters
    -----------------
    These parameters must be NumPy array of the correct shape and will be
    filled with the results of the computation.  These are essentially
    given by Eq. (25) of the original paper, with the important difference
    that b is here in reality b_ij - m_j. This ensures better numerical
    stability.

    q: array-like, shape (1,)
        The q parameter.

    b: array-like, shape (d,)
        The b vector.

    B: array-like, shape (d, d)
        The B matrix.

    """
    cdef double *x
    cdef double *T
    cdef double *VRt
    w_f = np.asfortranarray(w)
    if R is not None:
        Rt_f = np.asfortranarray(R.T)
    else:
        Rt_f = None
    S_f = np.asfortranarray(S)
    m_f = np.asfortranarray(m)
    V_f = np.asfortranarray(V)
    k_f = np.asfortranarray(k)
    r = w.shape[0]
    d = m.shape[0]
    if R is not None:
        assert R.shape[1] == d, "R mismatch 1"
        assert R.shape[0] == r, "R mismatch 0"
    assert S.shape[0] == r and S.shape[1] == r, "S mismatch"
    assert V.shape[0] == d and V.shape[1] == d, "V mismatch"
    assert p.shape[0] == 1, "p mismatch"
    assert var.shape[0] == 1, "var mismatch"
    assert ev.shape[0] == 1, "ev mismatch"
    x = <double *>malloc(r*sizeof(DOUBLE))
    Rk = <double *>malloc(r*sizeof(DOUBLE))
    T = <double *>malloc(r*r*sizeof(DOUBLE))
    VRt = <double *>malloc(d*r*sizeof(DOUBLE))
    if R is not None:
        if M is not None:
            predict_single_d(w_f, Rt_f, S_f, m_f, V_f, k_f, x, Rk, T, VRt, 
                &p[0], &var[0], &ev[0], &M[0], &C[0], &W[0,0])
        else:
            predict_single_d(w_f, Rt_f, S_f, m_f, V_f, k_f, x, Rk, T, VRt, 
                &p[0], &var[0], &ev[0], NULL, NULL, NULL)
    else:
        raise NotImplementedError("predict_single_noproj")
        # predict_single_noproj_d(w_f, S_f, m_f, V_f, k_f, x, Rk, T, VRt, &q[0], &b[0], &B[0,0])
    free(x)
    free(T)
    free(VRt)


@cython.binding(True)
cpdef double _scores_d(double[::1,:] w, double[::1,:,:] S,
                       double[::1,:] alphaclass,
                       double[::1,:] m, double[::1,:,:] V,
                       double[::1,:] logclasses, double [::1, :] q,
                       double[::1,:,:] Rt=None):
    """
    Compute the score (log-likelihood) for each sample & component.

    Parameters
    ----------
    Note: all array parameters are expected to be provided as Fortran
    contiguous arrays.

    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions

    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.

    alphaclass: array-like, shape (c, k)
        Array with the statistical weight per class of each Gaussian. Runs
        over the k clusters and the c classes.

    m: array-like, shape (d, k)
        Centers of multivariate Gaussians.

    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians.

    logclasses: array-like, shape (c, n)
        Log-probabilities that each observation belong to a given class. Use
        logclasses = np.zeros((1,n)) to prevent the use of classes.

    Output Parameters
    -----------------
    q: array-like, shape (k, n)
        The computed score (log-likelihood) for each component and each point.
        Note that the computed log-likelihood does *not* include the term due
        to the scaling of eeach cluster (alpha), but only the one associated
        to the class probability (alphaclass).

    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into
        the observed r-dimensional vector. If None, it is assumed that r=d
        and that no project is performed (equivalently: R is an array if
        identity matrices).
    """
    # Temporary data for the E-step
    # x: (r,), difference = w - R*m
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    #
    # Results of the E-step
    # q: (k,)
    # p: (c, k)
    # b: (d, k)
    # B: (d, d, k)
    cdef int r=w.shape[0], n=w.shape[1], d=m.shape[0], k=m.shape[1]
    cdef int c=alphaclass.shape[0]
    cdef int i, j, l, n1, n2, j_, l_, n1_, n2_, noweights, numfixalpha
    cdef double norm = r * log(2 * PI) / 2.0
    cdef double[::1,:] logalphaclass = np.log(alphaclass)
    # All these are local variables
    cdef double *x
    cdef double *T
    cdef double *VRt
    cdef double *qclass
    cdef double *b
    cdef double *B

    with nogil, parallel():
        # Allocates the block-local variables
        x = <double *>malloc(r*sizeof(double))
        T = <double *>malloc(r*r*sizeof(double))
        VRt = <double *>malloc(r*d*sizeof(double))
        qclass = <double *>malloc(c*sizeof(double))
        b = <double *>malloc(k*d*sizeof(double))
        B = <double *>malloc(k*d*d*sizeof(double))
        for i in prange(n, schedule='static'):
            # E-step at i (object number) fixed
            for j in range(k):
                # Perform the E-step. Note that b return does not include
                # the m term, so it is really b_ij - m_j
                if Rt is None:
                    e_single_step_noproj_d(w[:,i], S[:,:,i],
                                           m[:,j], V[:,:,j],
                                           x, T, VRt,
                                           &q[j, i], &b[j*d], &B[j*d*d])
                else:
                    e_single_step_d(w[:,i], Rt[:,:,i], S[:,:,i],
                                    m[:,j], V[:,:,j],
                                    x, T, VRt,
                                    &q[j, i], &b[j*d], &B[j*d*d])
                for l in range(c):
                    qclass[l] = logalphaclass[l, j] + logclasses[l, i]
                q[j, i] += logsumexp_d(qclass, c) - norm
        # Free the memory for block-local variables
        free(x)
        free(T)
        free(VRt)
        free(qclass)
        free(b)
        free(B)
    return 0


@cython.binding(True)
cpdef float _scores_s(float[::1,:] w, float[::1,:,:] S,
                       float[::1,:] alphaclass,
                       float[::1,:] m, float[::1,:,:] V,
                       float[::1,:] logclasses, float [::1, :] q,
                       float[::1,:,:] Rt=None):
    """
    Compute the score (log-likelihood) for each sample & component.

    Parameters
    ----------
    Note: all array parameters are expected to be provided as Fortran
    contiguous arrays.

    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions

    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.

    alphaclass: array-like, shape (c, k)
        Array with the statistical weight per class of each Gaussian. Runs
        over the k clusters and the c classes.

    m: array-like, shape (d, k)
        Centers of multivariate Gaussians.

    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians.

    logclasses: array-like, shape (c, n)
        Log-probabilities that each observation belong to a given class. Use
        logclasses = np.zeros((1,n)) to prevent the use of classes.

    Output Parameters
    -----------------
    q: array-like, shape (k, n)
        The computed score (log-likelihood) for each component and each point.
        Note that the computed log-likelihood does *not* include the term due
        to the scaling of eeach cluster (alpha), but only the one associated
        to the class probability (alphaclass).

    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into
        the observed r-dimensional vector. If None, it is assumed that r=d
        and that no project is performed (equivalently: R is an array if
        identity matrices).
    """
    # Temporary data for the E-step
    # x: (r,), difference = w - R*m
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    #
    # Results of the E-step
    # q: (k,)
    # p: (c, k)
    # b: (d, k)
    # B: (d, d, k)
    cdef int r=w.shape[0], n=w.shape[1], d=m.shape[0], k=m.shape[1]
    cdef int c=alphaclass.shape[0]
    cdef int i, j, l, n1, n2, j_, l_, n1_, n2_, noweights, numfixalpha
    cdef float norm = r * logf(2 * PI) / 2.0
    cdef float[::1,:] logalphaclass = np.log(alphaclass)
    # All these are local variables
    cdef float *x
    cdef float *T
    cdef float *VRt
    cdef float *qclass
    cdef float *b
    cdef float *B

    with nogil, parallel():
        # Allocates the block-local variables
        x = <float *>malloc(r*sizeof(float))
        T = <float *>malloc(r*r*sizeof(float))
        VRt = <float *>malloc(r*d*sizeof(float))
        qclass = <float *>malloc(c*sizeof(float))
        b = <float *>malloc(k*d*sizeof(float))
        B = <float *>malloc(k*d*d*sizeof(float))
        for i in prange(n, schedule='static'):
            # E-step at i (object number) fixed
            for j in range(k):
                # Perform the E-step. Note that b return does not include
                # the m term, so it is really b_ij - m_j
                if Rt is None:
                    e_single_step_noproj_s(w[:,i], S[:,:,i],
                                           m[:,j], V[:,:,j],
                                           x, T, VRt,
                                           &q[j, i], &b[j*d], &B[j*d*d])
                else:
                    e_single_step_s(w[:,i], Rt[:,:,i], S[:,:,i],
                                    m[:,j], V[:,:,j],
                                    x, T, VRt,
                                    &q[j, i], &b[j*d], &B[j*d*d])
                for l in range(c):
                    qclass[l] = logalphaclass[l, j] + logclasses[l, i]
                q[j, i] += logsumexp_s(qclass, c) - norm
        # Free the memory for block-local variables
        free(x)
        free(T)
        free(VRt)
        free(qclass)
        free(b)
        free(B)
    return 0



@cython.binding(True)
cpdef double em_step_d(double[::1,:] w, double[::1,:,:] S,
                       double[::1] alpha, double[::1,:] alphaclass,
                       double[::1,:] m, double[::1,:,:] V,
                       double[::1] logweights, double[::1,:] logclasses,
                       double[::1,:,:] Rt=None,
                       uint8_t[::1] fixpars=None, double regularization=0.0):
    """
    Perform a single E-M step for the extreme deconvolution.

    Parameters
    ----------
    Note: all array parameters are expected to be provided as Fortran
    contiguous arrays.

    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions

    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.

    alpha: array-like, shape (k,)
        Array with the statistical weight of each Gaussian. Updated at
        the exit with the new weights.

    alphaclass: array-like, shape (c, k)
        Array with the statistical weight per class of each Gaussian. Updated
        at the exit with the new weights. Runs over the k clusters and the c
        classes.

    m: array-like, shape (d, k)
        Centers of multivariate Gaussians, updated at the exit with the new
        centers.

    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians, updated
        at the exit with the new covariance matrices.

    logweights: array-like, shape (n,)
        Log-weights for each observation, or None. Use logweights = np.zeros(n)
        to prevent the use of weights.

    logclasses: array-like, shape (c, n)
        Log-probabilities that each observation belong to a given class. Use
        logclasses = np.zeros((1,n)) to prevent the use of classes.

    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into
        the observed r-dimensional vector. If None, it is assumed that r=d
        and that no project is performed (equivalently: R is an array if
        identity matrices).

    fixpars: array-like, shape (k,)
        Array of bitmasks with the FIX_AMP, FIX_MEAN, and FIX_AMP combinations.

    regularization: double, default=0
        Regularization parameter (use 0 to prevent the regularization).

    """
    # Temporary data for the E-step
    # x: (r,), difference = w - R*m
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    #
    # Results of the E-step
    # q: (k,)
    # p: (c, k)
    # b: (d, k)
    # B: (d, d, k)
    #
    # Results of the M-step
    # M0: (k,), equivalent to the new alpha over all classes
    # m0: (c, k), equivalent to the new alpha
    # m1: (d, k), equivalent to the new m
    # m2: (d, d, k), equivalent to the new V
    cdef int r=w.shape[0], n=w.shape[1], d=m.shape[0], k=m.shape[1]
    cdef int c=alphaclass.shape[0]
    cdef int i, j, l, n1, n2, j_, l_, n1_, n2_, noweights, numfixalpha
    cdef double qsum, weightsum, loglike=0, exp_q, sumalpha, sumfreealpha
    cdef double norm = r * log(2 * PI) / 2.0
    cdef double[::1] logalpha = np.log(alpha)
    cdef double[::1,:] logalphaclass = np.log(alphaclass)
    # All these are local variables
    cdef double *x
    cdef double *T
    cdef double *VRt
    cdef double *q
    cdef double *qclass
    cdef double *b
    cdef double *B
    cdef double *M0_local
    cdef double *m0_local
    cdef double *m1_local
    cdef double *m2_local
    cdef double *M0
    cdef double *m0
    cdef double *m1
    cdef double *m2

    # Set the weight variables
    weightsum = 0.0
    for i in prange(n, nogil=True):
        weightsum += exp(logweights[i])

    # Set the fixed alpha variables
    sumfreealpha = 1.0
    numfixalpha = 0
    for j in range(k):
        if fixpars is not None and fixpars[j] & _FIX_AMP:
            numfixalpha += 1
            sumfreealpha -= alpha[j]
    M0 = <double *>calloc(k, sizeof(double))
    m0 = <double *>calloc(k*c, sizeof(double))
    m1 = <double *>calloc(k*d, sizeof(double))
    m2 = <double *>calloc(k*d*d, sizeof(double))
    with nogil, parallel():
        # Allocates the block-local variables
        x = <double *>malloc(r*sizeof(double))
        T = <double *>malloc(r*r*sizeof(double))
        VRt = <double *>malloc(r*d*sizeof(double))
        q = <double *>malloc(k*sizeof(double))
        qclass = <double *>malloc(c*sizeof(double))
        b = <double *>malloc(k*d*sizeof(double))
        B = <double *>malloc(k*d*d*sizeof(double))
        # Allocate the arrays for the moments
        M0_local = <double *>calloc(k, sizeof(double))
        m0_local = <double *>calloc(k*c, sizeof(double))
        m1_local = <double *>calloc(k*d, sizeof(double))
        m2_local = <double *>calloc(k*d*d, sizeof(double))
        for i in prange(n, schedule='static'):
            # E-step at i (object number) fixed
            for j in range(k):
                # Perform the E-step. Note that b return does not include
                # the m term, so it is really b_ij - m_j
                if Rt is None:
                    e_single_step_noproj_d(w[:,i], S[:,:,i],
                                           m[:,j], V[:,:,j],
                                           x, T, VRt,
                                           &q[j], &b[j*d], &B[j*d*d])
                else:
                    e_single_step_d(w[:,i], Rt[:,:,i], S[:,:,i],
                                    m[:,j], V[:,:,j],
                                    x, T, VRt,
                                    &q[j], &b[j*d], &B[j*d*d])
                for l in range(c):
                    qclass[l] = logalphaclass[l, j] + logclasses[l, i]
                q[j] = q[j] + logsumexp_d(qclass, c) + logalpha[j]
            qsum = logsumexp_d(q, k)
            loglike += qsum - norm
            for j in range(k):
                q[j] += logweights[i] - qsum
            # done! Now we can proceed with the M-step, which operates
            # a sum over all objects. We compute the moments, using the inplace
            # += operator which forces the use of a reductions for m0, m1, and m2.
            for j in range(k):
                exp_q = exp(q[j])
                # M0 is the sum of all q_ij over i, so related to alpha
                M0_local[j] += exp_q
                # m0 is similar to M0, but includes an index for the class; it is
                # more directly related to alpha
                for l in range(c):
                    m0_local[j*c+l] += exp(q[j] + logclasses[l,i])
                for n1 in range(d):
                    # m1 is the sum of q_ij (b_ij - m_j) over i, so related to m_j - m_j(old)
                    m1_local[j*d+n1] += exp_q*b[j*d+n1]
                    for n2 in range(n1, d):
                        # m2 is the sum of q_ij (b_ij - m_j)(b_ij - m_j)^T + B_ij, so related to V_j,
                        # except that the moments are computed using m_j old
                        m2_local[(j*d+n1)*d+n2] += exp_q*(b[j*d+n1]*b[j*d+n2] + B[(j*d+n2)*d+n1])
        # Collect all the reduction variables of the various threads
        with gil:
            for j_ in range(k):
                M0[j_] += M0_local[j_]
                for l_ in range(c):
                    m0[j_*c+l_] += m0_local[j_*c+l_]
                for n1_ in range(d):
                    m1[j_*d+n1_] += m1_local[j_*d+n1_]
                    for n2_ in range(n1_, d):
                        m2[(j_*d+n1_)*d+n2_] += m2_local[(j_*d+n1_)*d+n2_]
        # Free the memory for block-local variables
        free(x)
        free(T)
        free(VRt)
        free(q)
        free(qclass)
        free(b)
        free(B)
        free(M0_local)
        free(m0_local)
        free(m1_local)
        free(m2_local)

    # Good, now we have all sum moments. We need to normalize them.
    for j in range(k):
        # We check now if we need to update the mean, since m1 enters the 
        # correction of the covariance
        if fixpars is not None and fixpars[j] & _FIX_MEAN:
            for n1 in range(d):
                m1[j*d+n1] = 0
        else:
            for n1 in range(d):
                m1[j*d+n1] /= M0[j]
    # Cannot do directly m2, because it depends on the correct computation of
    # m1: therefore we need to repeat the loop
    if regularization > 0:
        for j in range(k):
            for n1 in range(d):
                m2[(j*d+n1)*d+n1] = (m2[(j*d+n1)*d+n1] - m1[j*d+n1]*m1[j*d+n1] * M0[j] + regularization) \
                    / (M0[j] + 1.0)
                for n2 in range(n1 + 1, d):
                    m2[(j*d+n1)*d+n2] = m2[(j*d+n2)*d+n1] = (m2[(j*d+n1)*d+n2] - m1[j*d+n1]*m1[j*d+n2] * M0[j]) \
                      / (M0[j] + 1.0)
    else:
        for j in range(k):
            for n1 in range(d):
                for n2 in range(n1, d):
                    m2[(j*d+n1)*d+n2] = m2[(j*d+n2)*d+n1] = (m2[(j*d+n1)*d+n2] / M0[j] - m1[j*d+n1]*m1[j*d+n2])
    # Done, save the results back
    for j in range(k):
        if fixpars is None or not fixpars[j] & _FIX_AMP:
            alpha[j] = M0[j] / weightsum
        if fixpars is None or not fixpars[j] & _FIX_CLASS:
            if M0[j] > 0:
                for l in range(c):
                    alphaclass[l, j] = m0[j*c+l] / M0[j]
            else:
                for l in range(c):
                    alphaclass[l, j] = 1.0 / c
        for n1 in range(d):
            # The += sign here is due to the use of an E-step w/o the m_j term in b_ij
            m[n1, j] += m1[j*d+n1]
            if fixpars is None or not fixpars[j] & _FIX_COVAR:
                for n2 in range(d):
                    V[n1, n2, j] = m2[(j*d+n2)*d+n1]
    # In case we have fixed some amplitudes, we need to rinormalize
    if numfixalpha > 0:
        sumalpha = 0.0
        for j in range(k):
            if not fixpars[j] & _FIX_AMP:
                sumalpha += alpha[j]
        for j in range(k):
            if not fixpars[j] & _FIX_AMP:
                alpha[j] *= sumfreealpha / sumalpha

    # Finally free the memory
    free(M0)
    free(m0)
    free(m1)
    free(m2)
    return loglike / n


@cython.binding(True)
cpdef float em_step_s(float[::1,:] w, float[::1,:,:] S,
                       float[::1] alpha, float[::1,:] alphaclass,
                       float[::1,:] m, float[::1,:,:] V,
                       float[::1] logweights, float[::1,:] logclasses,
                       float[::1,:,:] Rt=None,
                       uint8_t[::1] fixpars=None, float regularization=0.0):
    """
    Perform a single E-M step for the extreme deconvolution.

    Parameters
    ----------
    Note: all array parameters are expected to be provided as Fortran
    contiguous arrays.

    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions

    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.

    alpha: array-like, shape (k,)
        Array with the statistical weight of each Gaussian. Updated at
        the exit with the new weights.

    alphaclass: array-like, shape (c, k)
        Array with the statistical weight per class of each Gaussian. Updated
        at the exit with the new weights. Runs over the k clusters and the c
        classes.

    m: array-like, shape (d, k)
        Centers of multivariate Gaussians, updated at the exit with the new
        centers.

    V: array-like, shape (d, d, k)
        Array of covariance matrices of the multivariate Gaussians, updated
        at the exit with the new covariance matrices.

    logweights: array-like, shape (n,)
        Log-weights for each observation, or None. Use logweights = np.zeros(n)
        to prevent the use of weights.

    logclasses: array-like, shape (c, n)
        Log-probabilities that each observation belong to a given class. Use
        logclasses = np.zeros((1,n)) to prevent the use of classes.

    Optional Parameters
    -------------------
    Rt: array-like, shape (d, r, n)
        Array of projection matrices: for each datum (n), it is the transpose
        of the matrix that transforms the original d-dimensional vector into
        the observed r-dimensional vector. If None, it is assumed that r=d
        and that no project is performed (equivalently: R is an array if
        identity matrices).

    fixpars: array-like, shape (k,)
        Array of bitmasks with the FIX_AMP, FIX_MEAN, and FIX_AMP combinations.

    regularization: float, default=0
        Regularization parameter (use 0 to prevent the regularization).

    """
    # Temporary data for the E-step
    # x: (r,), difference = w - R*m
    # T: (r, r), combined projected covariance = R*V*R^T + S
    # VRt: (d, r), product VRt = V*Rt
    #
    # Results of the E-step
    # q: (k,)
    # p: (c, k)
    # b: (d, k)
    # B: (d, d, k)
    #
    # Results of the M-step
    # M0: (k,), equivalent to the new alpha over all classes
    # m0: (c, k), equivalent to the new alpha
    # m1: (d, k), equivalent to the new m
    # m2: (d, d, k), equivalent to the new V
    cdef int r=w.shape[0], n=w.shape[1], d=m.shape[0], k=m.shape[1]
    cdef int c=alphaclass.shape[0]
    cdef int i, j, l, n1, n2, j_, l_, n1_, n2_, noweights, numfixalpha
    cdef float qsum, weightsum, loglike=0, exp_q, sumalpha, sumfreealpha
    cdef float norm = r * logf(2 * PI) / 2.0
    cdef float[::1] logalpha = np.log(alpha)
    cdef float[::1,:] logalphaclass = np.log(alphaclass)
    # All these are local variables
    cdef float *x
    cdef float *T
    cdef float *VRt
    cdef float *q
    cdef float *qclass
    cdef float *b
    cdef float *B
    cdef float *M0_local
    cdef float *m0_local
    cdef float *m1_local
    cdef float *m2_local
    cdef float *M0
    cdef float *m0
    cdef float *m1
    cdef float *m2

    # Set the weight variables
    weightsum = 0.0
    for i in prange(n, nogil=True):
        weightsum += expf(logweights[i])

    # Set the fixed alpha variables
    sumfreealpha = 1.0
    numfixalpha = 0
    for j in range(k):
        if fixpars is not None and fixpars[j] & _FIX_AMP:
            numfixalpha += 1
            sumfreealpha -= alpha[j]
    M0 = <float *>calloc(k, sizeof(float))
    m0 = <float *>calloc(k*c, sizeof(float))
    m1 = <float *>calloc(k*d, sizeof(float))
    m2 = <float *>calloc(k*d*d, sizeof(float))
    with nogil, parallel():
        # Allocates the block-local variables
        x = <float *>malloc(r*sizeof(float))
        T = <float *>malloc(r*r*sizeof(float))
        VRt = <float *>malloc(r*d*sizeof(float))
        q = <float *>malloc(k*sizeof(float))
        qclass = <float *>malloc(c*sizeof(float))
        b = <float *>malloc(k*d*sizeof(float))
        B = <float *>malloc(k*d*d*sizeof(float))
        # Allocate the arrays for the moments
        M0_local = <float *>calloc(k, sizeof(float))
        m0_local = <float *>calloc(k*c, sizeof(float))
        m1_local = <float *>calloc(k*d, sizeof(float))
        m2_local = <float *>calloc(k*d*d, sizeof(float))
        for i in prange(n, schedule='static'):
            # E-step at i (object number) fixed
            for j in range(k):
                # Perform the E-step. Note that b return does not include
                # the m term, so it is really b_ij - m_j
                if Rt is None:
                    e_single_step_noproj_s(w[:,i], S[:,:,i],
                                           m[:,j], V[:,:,j],
                                           x, T, VRt,
                                           &q[j], &b[j*d], &B[j*d*d])
                else:
                    e_single_step_s(w[:,i], Rt[:,:,i], S[:,:,i],
                                    m[:,j], V[:,:,j],
                                    x, T, VRt,
                                    &q[j], &b[j*d], &B[j*d*d])
                for l in range(c):
                    qclass[l] = logalphaclass[l, j] + logclasses[l, i]
                q[j] = q[j] + logsumexp_s(qclass, c) + logalpha[j]
            qsum = logsumexp_s(q, k)
            loglike += qsum - norm
            for j in range(k):
                q[j] += logweights[i] - qsum
            # done! Now we can proceed with the M-step, which operates
            # a sum over all objects. We compute the moments, using the inplace
            # += operator which forces the use of a reductions for m0, m1, and m2.
            for j in range(k):
                exp_q = expf(q[j])
                # M0 is the sum of all q_ij over i, so related to alpha
                M0_local[j] += exp_q
                # m0 is similar to M0, but includes an index for the class; it is
                # more directly related to alpha
                for l in range(c):
                    m0_local[j*c+l] += expf(q[j] + logclasses[l,i])
                for n1 in range(d):
                    # m1 is the sum of q_ij (b_ij - m_j) over i, so related to m_j - m_j(old)
                    m1_local[j*d+n1] += exp_q*b[j*d+n1]
                    for n2 in range(n1, d):
                        # m2 is the sum of q_ij (b_ij - m_j)(b_ij - m_j)^T + B_ij, so related to V_j,
                        # except that the moments are computed using m_j old
                        m2_local[(j*d+n1)*d+n2] += exp_q*(b[j*d+n1]*b[j*d+n2] + B[(j*d+n2)*d+n1])
        # Collect all the reduction variables of the various threads
        with gil:
            for j_ in range(k):
                M0[j_] += M0_local[j_]
                for l_ in range(c):
                    m0[j_*c+l_] += m0_local[j_*c+l_]
                for n1_ in range(d):
                    m1[j_*d+n1_] += m1_local[j_*d+n1_]
                    for n2_ in range(n1_, d):
                        m2[(j_*d+n1_)*d+n2_] += m2_local[(j_*d+n1_)*d+n2_]
        # Free the memory for block-local variables
        free(x)
        free(T)
        free(VRt)
        free(q)
        free(qclass)
        free(b)
        free(B)
        free(M0_local)
        free(m0_local)
        free(m1_local)
        free(m2_local)

    # Good, now we have all sum moments. We need to normalize them.
    for j in range(k):
        # We check now if we need to update the mean, since m1 enters the 
        # correction of the covariance
        if fixpars is not None and fixpars[j] & _FIX_MEAN:
            for n1 in range(d):
                m1[j*d+n1] = 0
        else:
            for n1 in range(d):
                m1[j*d+n1] /= M0[j]
    # Cannot do directly m2, because it depends on the correct computation of
    # m1: therefore we need to repeat the loop
    if regularization > 0:
        for j in range(k):
            for n1 in range(d):
                m2[(j*d+n1)*d+n1] = (m2[(j*d+n1)*d+n1] - m1[j*d+n1]*m1[j*d+n1] * M0[j] + regularization) \
                    / (M0[j] + 1.0)
                for n2 in range(n1 + 1, d):
                    m2[(j*d+n1)*d+n2] = m2[(j*d+n2)*d+n1] = (m2[(j*d+n1)*d+n2] - m1[j*d+n1]*m1[j*d+n2] * M0[j]) \
                      / (M0[j] + 1.0)
    else:
        for j in range(k):
            for n1 in range(d):
                for n2 in range(n1, d):
                    m2[(j*d+n1)*d+n2] = m2[(j*d+n2)*d+n1] = (m2[(j*d+n1)*d+n2] / M0[j] - m1[j*d+n1]*m1[j*d+n2])
    # Done, save the results back
    for j in range(k):
        if fixpars is None or not fixpars[j] & _FIX_AMP:
            alpha[j] = M0[j] / weightsum
        if fixpars is None or not fixpars[j] & _FIX_CLASS:
            if M0[j] > 0:
                for l in range(c):
                    alphaclass[l, j] = m0[j*c+l] / M0[j]
            else:
                for l in range(c):
                    alphaclass[l, j] = 1.0 / c
        for n1 in range(d):
            # The += sign here is due to the use of an E-step w/o the m_j term in b_ij
            m[n1, j] += m1[j*d+n1]
            if fixpars is None or not fixpars[j] & _FIX_COVAR:
                for n2 in range(d):
                    V[n1, n2, j] = m2[(j*d+n2)*d+n1]
    # In case we have fixed some amplitudes, we need to rinormalize
    if numfixalpha > 0:
        sumalpha = 0.0
        for j in range(k):
            if not fixpars[j] & _FIX_AMP:
                sumalpha += alpha[j]
        for j in range(k):
            if not fixpars[j] & _FIX_AMP:
                alpha[j] *= sumfreealpha / sumalpha

    # Finally free the memory
    free(M0)
    free(m0)
    free(m1)
    free(m2)
    return loglike / n
