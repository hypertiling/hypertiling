import math
from hypertiling.check_numba import check_numba
from numpy import array as nparray



@check_numba
def p2w(z):
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([(1 + xx + yy), 2 * x, 2 * y])


@check_numba
def w2p(point):
    '''Convert Weierstraß to Poincare representation '''
    [t, x, y] = point
    factor = 1 / (1 + t)
    return complex(x * factor, y * factor)


@check_numba
def p2w_xyt(z):
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([2 * x, 2 * y, (1 + xx + yy)])


@check_numba
def w2p_xyt(point):
    '''Convert Weierstraß to Poincare representation '''
    [x, y, t] = point
    factor = 1 / (1 + t)
    return complex(x * factor, y * factor)
