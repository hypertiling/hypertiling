import numpy as np
from hypertiling.check_numba import NumbaChecker
from numpy import array as nparray

def valid_weierstrass_point(point):
    """
    Check that a point is a valid Weierstrass point.

    Parameters
    ----------
    point : list
        A list containing three elements [t, x, y].

    Returns
    -------
    bool
        True if the point is a valid Weierstrass point, False otherwise.
    """
    [t, x, y] = point
    return t > 0 and (1 + x * x + y * y) * t * t > 1


@NumbaChecker("float64[::1](complex128)")
def p2w(z: np.complex128) -> np.array:
    """
    Convert Poincare to Weierstraß representation.

    Parameters
    ----------
    z : np.complex128
        A complex number.

    Returns
    -------
    np.array
        A numpy array representing Weierstraß coordinates.
    """
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([(1 + xx + yy), 2 * x, 2 * y])


@NumbaChecker("complex128(float64[:])")
def w2p(point: np.array) -> np.complex128:
    """
    Convert Weierstraß to Poincare representation.

    Parameters
    ----------
    point : np.array
        A numpy array representing Weierstraß coordinates.

    Returns
    -------
    np.complex128
        A complex number representing Poincare coordinates.
    """
    [t, x, y] = point
    factor = 1 / (1 + t)
    return np.complex128(complex(x * factor, y * factor))


@NumbaChecker("float64[::1](complex128)")
def p2w_xyt(z: np.complex128) -> np.array:
    """
    Convert Poincare to Weierstraß representation.

    Parameters
    ----------
    z : np.complex128
        A complex number.

    Returns
    -------
    np.array
        A numpy array representing Weierstraß coordinates in [x, y, t] format.
    """
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([2 * x, 2 * y, (1 + xx + yy)])


@NumbaChecker("complex128(float64[:])")
def w2p_xyt(point: np.array) -> np.complex128:
    """
    Convert Weierstraß to Poincare representation.

    Parameters
    ----------
    point : np.array
        A numpy array representing Weierstraß coordinates in [x, y, t] format.

    Returns
    -------
    np.complex128
        A complex number representing Poincare coordinates.
    """
    [x, y, t] = point
    factor = 1 / (1 + t)
    return np.complex128(complex(x * factor, y * factor))


def p2w_xyt_vector(z_list):
    """
    Convert a list of Poincare coordinates to Weierstraß representation.

    Parameters
    ----------
    z_list : list
        List of complex numbers representing Poincare coordinates.

    Returns
    -------
    np.array
        A numpy array of Weierstraß coordinates.
    """
    return np.array([p2w_xyt(x) for x in z_list])


def w2p_xyt_vector(xyt_list):
    """
    Convert a list of Weierstraß coordinates to Poincare representation.

    Parameters
    ----------
    xyt_list : list
        List of numpy arrays representing Weierstraß coordinates in [x, y, t] format.

    Returns
    -------
    np.array
        A numpy array of complex numbers representing Poincare coordinates.
    """
    return np.array([w2p_xyt(z) for z in xyt_list])
