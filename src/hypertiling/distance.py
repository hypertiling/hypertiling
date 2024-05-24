import numpy as np
import math

# define signature of the embedding three-dimensional Minkowski space 
global signature
signature = np.array([1,-1,-1])

# Common distance metrics

def lorentzian_distance(a, b):
    """
    Compute the inner product between a and b, respecting the Minkowskian signature.

    Parameters
    ----------
    a : np.array(3) or np.array((N,3))
        The first input array.
    b : np.array(3)
        The second input array.

    Returns
    -------
    np.array or scalar
        If both a and b are 1-D arrays, a scalar is returned.
        If a is a 2-D array of shape (N,3) an array of length N is returned.
    """    
    return np.dot(a, np.multiply(signature, b))


def weierstrass_distance(a, b):
    """
    Compute distance between two points given in the Weierstraß (also called hyperboloid)
    coordinate representation (t,x,y).

    Parameters
    ----------
    a : np.array
        The first point in hyperboloid coordinate representation.
    b : np.array
        The second point in hyperboloid coordinate representation.

    Returns
    -------
    float
        The distance between a and b.
    """
    
    arg = lorentzian_distance(a,b)
    if arg < 1:
        return 0
    else:
        # for scalars math.acosh is usually faster than np.arccosh
        return math.acosh(arg)


def disk_distance(z1, z2):
    """
    Compute distance between two points given in terms of their Poincare disk coordinates.

    Parameters
    ----------
    z1 : complex
        The first point in Poincare disk coordinates.
    z2 : complex
        The second point in Poincare disk coordinates.

    Returns
    -------
    float
        The distance between z1 and z2.
    """
    
    num = abs(z1-z2)
    denom = abs(1-z1*z2.conjugate())
    return 2*math.atanh(num/denom)
