import math
from hypertiling.check_numba import NumbaChecker
from numpy import array as nparray


@NumbaChecker()
def p2w(z):
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([(1 + xx + yy), 2 * x, 2 * y])


@NumbaChecker()
def w2p(point):
    '''Convert Weierstraß to Poincare representation '''
    [t, x, y] = point
    factor = 1 / (1 + t)
    return complex(x * factor, y * factor)


@NumbaChecker()
def p2w_xyt(z):
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([2 * x, 2 * y, (1 + xx + yy)])


@NumbaChecker()
def w2p_xyt(point):
    '''Convert Weierstraß to Poincare representation '''
    [x, y, t] = point
    factor = 1 / (1 + t)
    return complex(x * factor, y * factor)
