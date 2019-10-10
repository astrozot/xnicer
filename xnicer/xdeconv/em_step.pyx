# Compile with: python setup.py build_ext --use-cython --inplace
# cython: language_level=3, embedsignature=True
# cython: wraparound=False, boundscheck=False, initializedcheck=False, cdivision=True

import cython
from cython.parallel import prange, parallel

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport exp, log
import numpy as np
from numpy cimport float64_t, uint8_t
from numpy.math cimport INFINITY
from scipy.linalg.cython_lapack cimport dpotrf, dtrtri
from scipy.linalg.cython_blas cimport dtrmv, dtrmm, ddot, dgemv, dgemm, dsyrk, dsymm

ctypedef float64_t DOUBLE
ctypedef uint8_t BOOL

cdef int _FIX_NONE = 0
cdef int _FIX_AMP = 1
cdef int _FIX_MEAN = 2
cdef int _FIX_COVAR = 4
cdef int _FIX_ALL = _FIX_AMP + _FIX_MEAN + _FIX_COVAR

FIX_NONE = _FIX_NONE
FIX_AMP = _FIX_AMP
FIX_MEAN = _FIX_MEAN
FIX_COVAR = _FIX_COVAR
FIX_ALL = _FIX_ALL


cdef double logsumexp(double *x, int n) nogil:
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
    for i in range(n):
        result += exp(x[i] - xmax) 
    return xmax + log(result)


cpdef double log_likelihoods(double[:,:] deltas, double[:,:,:] covars, double[::1] results=None):
    cdef int n, i, j, info, alloc_results=0
    cdef int nobjs=deltas.shape[0]
    cdef int r=covars.shape[2]
    cdef int inc=1
    cdef double* C
    cdef double* d
    cdef double result=0

    if results is None:
        results = np.empty(nobjs)
    C = <double *>malloc(r*r*sizeof(DOUBLE))
    d = <double *>malloc(r*sizeof(DOUBLE))
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
        result = 0.0
        for i in range(r):
            result += log(C[i*r + i])
        # Computes log(det C) - x^T*x / 2
        results[n] = result - 0.5*ddot(&r, d, &inc, d, &inc)
    free(C)
    return logsumexp(&results[0], nobjs)


cdef int e_single_step_noproj(double[::1] w, double[::1,:] S, 
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
    # Computes T := V + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = V[n2,n1] + S[n2,n1]
    # Computes x := w - m
    for n1 in range(r):
        x[n1] = w[n1] - m[n1]
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    dpotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    dtrtri("U", "N", &r, T, &ldT, &info)
    # Computes x := Tnew^T*x
    dtrmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes VRt := V*Tnew
    for n1 in range(r):
        for n2 in range(r):
            VRt[n1*r+n2] = V[n2,n1]
    dtrmm("R", "U", "N", "N", &r, &r, &one, T, &ldT, VRt, &ldVRt)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += log(T[n1*r+n1])
    # Computes q := log(det Tnew) - x^T*x / 2)
    q[0] = a - 0.5*ddot(&r, x, &incx, &x[0], &incx)
    # Computes b := VRt*x (the +m term has been dropped)
    for n1 in range(r):
        # b[n1] = m[n1]
        b[n1] = 0.0
    dgemv("N", &r, &r, &one, VRt, &ldVRt, x, &incx, &one, b, &incb)
    # Computes B := V - RV^T*RV
    for n1 in range(r):
        for n2 in range(r):
            B[n1*r+n2] = V[n2,n1]
    dsyrk("U", "N", &r, &r, &_one, VRt, &ldVRt, &one, B, &ldB)
    for n1 in range(r):
        for n2 in range(n1):
            B[n2*r+n1] = B[n1*r+n2]
    return 0


cpdef int py_e_single_step(double[::1] w, double[:,::1] R, double[:,::1] S, 
                           double[::1] m, double[:,::1] V,
                           double[::1] q, double[::1] b, double[:,::1] B):
    """Pure Python version of e_single_step.

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
    Rt_f = np.asfortranarray(R.T)
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
    e_single_step(w_f, Rt_f, S_f, m_f, V_f, x, T, VRt, &q[0], &b[0], &B[0,0])
    free(x)
    free(T)
    free(VRt)
    

cdef int e_single_step(double[::1] w, double[::1,:] Rt, double[::1,:] S, 
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
    # Computes VRt := VRt
    dsymm("L", "U", &d, &r, &one, &V[0,0], &ldV, &Rt[0,0], &ldRt, &zero, VRt, &ldVRt)
    # Computes T := R*V*Rt + S
    for n1 in range(r):
        for n2 in range(r):
            T[n1*r+n2] = S[n2,n1]
    dgemm("T", "N", &r, &r, &d, &one, &Rt[0,0], &ldRt, VRt, &ldVRt, &one, T, &ldT)
    # Computes x := w - R*m
    for n1 in range(r):
        x[n1] = w[n1]
    dgemv("T", &d, &r, &_one, &Rt[0,0], &ldRt, &m[0], &incm, &one, x, &incx)
    # Chowlesky decomposition of T, i.e. Tnew such that Tnew^T*Tnew = Told
    dpotrf("U", &r, T, &ldT, &info)
    # Inverse of the upper triangular T (or, better, Tnew)
    dtrtri("U", "N", &r, T, &ldT, &info)
    # Computes x := Tnew^T*x
    dtrmv("U", "T", "N", &r, T, &ldT, x, &incx)
    # Computes VRt := VRt*Tnew = V*Rt*Tnew
    dtrmm("R", "U", "N", "N", &d, &r, &one, T, &ldT, VRt, &ldVRt)
    # Computes the log of the determinant of Tnew
    a = 0.0
    for n1 in range(r):
        a += log(T[n1*r+n1])
    # Computes q := log(det Tnew) - x^T*x / 2
    q[0] = a - 0.5*ddot(&r, x, &incx, x, &incx)
    # Computes b := VRt*x (the +m term has been dropped)
    for n1 in range(d):
        # b[n1] = m[n1]
        b[n1] = 0.0
    dgemv("N", &d, &r, &one, VRt, &ldVRt, x, &incx, &one, b, &incb)
    # Computes B := V - (VRt)*(VRt)^T
    for n1 in range(d):
        for n2 in range(d):
            B[n1*d+n2] = V[n2,n1]
    dsyrk("U", "N", &d, &r, &_one, VRt, &ldVRt, &one, B, &ldB)
    for n1 in range(d):
        for n2 in range(n1):
            B[n2*d+n1] = B[n1*d+n2]
    return 0
    
    
@cython.binding(True)
cpdef double em_step(double[::1,:] w, double[::1,:,:] S, 
                     double[::1,:] alpha, double[::1,:] m, 
                     double[::1,:,:] V, 
                     double[::1] logweights, double[::1,:] logclasses,
                     double[::1,:,:] Rt=None, 
                     uint8_t[::1] fixpars=None, double regularization=0.0):
    """Perform a single E-M step for the extreme deconvolution.
    
    Parameters
    ----------
    Note: all array parameters are expected to be provided as Fortran
    contiguous arrays.
    
    w: array-like, shape (r, n)
        Set of observations involving n data, each having r dimensions
    
    S: array-like, shape (r, r, n)
        Array of covariances of the observational data w.
    
    alpha: array-like, shape (c, k)
        Array with the statistical weight of each Gaussian. Updated at the
        exit with the new weights. Runs over the k clusters and the c
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
        logglasses = np.zeros((1,n)) to prevent the use of classes.

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
    cdef int c=alpha.shape[0]
    cdef int i, j, l, n1, n2, j_, l_, n1_, n2_, noweights, numfreealpha
    cdef double qsum, weightsum, loglike=0, exp_q, sumfreealpha
    cdef double[::1,:] logalpha = np.log(alpha)
    # All these are local variables
    cdef double *x
    cdef double *T
    cdef double *VRt
    cdef double *q
    cdef double *p
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
    # TODO: perhaps move the computation of weightsum to the main function
    weightsum = 0.0
    for i in prange(n, nogil=True):
        weightsum += exp(logweights[i])

    # Set the fixed alpha variables
    sumfreealpha = 1.0
    numfixalpha = 0
    for j in range(k):
        if fixpars is not None and fixpars[j] & _FIX_AMP:
            numfixalpha += 1
            for l in range(c):
                sumfreealpha -= alpha[l, j]
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
        p = <double *>malloc(k*c*sizeof(double))
        b = <double *>malloc(k*d*sizeof(double))
        B = <double *>malloc(k*d*d*sizeof(double))
        # Allocate the arrays for the moments
        M0_local = <double *>calloc(k, sizeof(double))
        m0_local = <double *>calloc(k*c, sizeof(double))
        m1_local = <double *>calloc(k*d, sizeof(double))
        m2_local = <double *>calloc(k*d*d, sizeof(double))
        for i in prange(n):
            # E-step at i (object number) fixed
            for j in range(k):
                # Perform the E-step. Note that b return does not include
                # the m term, so it is really b_ij - m_j
                if Rt is None:
                    e_single_step_noproj(w[:,i], S[:,:,i], 
                                         m[:,j], V[:,:,j], 
                                         x, T, VRt,
                                         &q[j], &b[j*d], &B[j*d*d])
                else:
                    e_single_step(w[:,i], Rt[:,:,i], S[:,:,i], 
                                  m[:,j], V[:,:,j], 
                                  x, T, VRt,
                                  &q[j], &b[j*d], &B[j*d*d])
                for l in range(c):
                    p[j*c+l] = q[j] + logalpha[l, j] + logclasses[l, i]
                q[j] = logsumexp(&p[j*c], c)
            qsum = logsumexp(q, k)
            loglike += qsum
            for j in range(k):
                q[j] += logweights[i] - qsum
                for l in range(c):
                    p[j*c+l] += logweights[i] - qsum
                    # FIXME 
                    p[j*c+l] = q[j] + logclasses[l, i]
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
                    m0_local[j*c+l] += exp(p[j*c+l])
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
        free(p)
        free(b)
        free(B)
        free(M0_local)
        free(m0_local)
        free(m1_local)
        free(m2_local)

    # Good, now we have all sum moments. We need to normalize them.
    for j in range(k):
        # We check now if we need to update the mean, since m1 enters the correction of the covariance
        if fixpars is not None and fixpars[j] & _FIX_MEAN:
            for n1 in range(d):
                m1[j*d+n1] = 0
        else:
            for n1 in range(d):
                m1[j*d+n1] /= M0[j]
    # Cannot do directly m2, because it depends on the correct computation of m1: therefore
    # we need to repeat the loop
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
            for l in range(c):
                alpha[l, j] = m0[j*c+l] / weightsum
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
                for l in range(c):
                    sumalpha += alpha[l, j]
        for j in range(k):
            if not fixpars[j] & _FIX_AMP:
                for l in range(c):
                    alpha[l, j] *= sumfreealpha / sumalpha
                
    # Finally free the memory
    free(M0)
    free(m0)
    free(m1)
    free(m2)
    return loglike / n

