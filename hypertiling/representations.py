import numpy as np
from hypertiling.check_numba import NumbaChecker
from numpy import array as nparray


@NumbaChecker("float64[::1](complex128)")
def p2w(z: np.complex128) -> np.array:
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([(1 + xx + yy), 2 * x, 2 * y])


@NumbaChecker("complex128(float64[:])")
def w2p(point: np.array) -> np.complex128:
    '''Convert Weierstraß to Poincare representation '''
    [t, x, y] = point
    factor = 1 / (1 + t)
    return np.complex128(complex(x * factor, y * factor))


@NumbaChecker("float64[::1](complex128)")
def p2w_xyt(z: np.complex128) -> np.array:
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([2 * x, 2 * y, (1 + xx + yy)])


@NumbaChecker("complex128(float64[:])")
def w2p_xyt(point: np.array) -> np.complex128:
    '''Convert Weierstraß to Poincare representation '''
    [x, y, t] = point
    factor = 1 / (1 + t)
    return np.complex128(complex(x * factor, y * factor))
