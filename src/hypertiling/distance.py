import numpy as np
import math

# define signature of the embedding three-dimensional Minkowski space 
global signature
signature = np.array([1,-1,-1])

# Common distance metrics

def lorentzian_distance(a, b):
    """
    Compute the inner product between a and b, respecting the Minkowskian signature

    Parameters:
        a : np.array(3) or np.array((N,3))
        b : np.array(3)

    Returns:
        if both a and b are 1-D arrays, a scalar is returned
        if a is a 2-D array of shape (N,3) an array of length N is returned
    """    
    return np.dot(a, np.multiply(signature,b))


def weierstrass_distance(a, b):
    """
    Compute distance between two points given in the Weierstraß (also called hyperboloid)
    coordinate representation (t,x,y)
    """
    arg = lorentzian_distance(a,b)
    if arg < 1:
        return 0
    else:
        # for scalars math.acosh is usually faster than np.arccosh
        return math.acosh(arg)


def disk_distance(z1, z2):
    """
    Compute distance between two points given in terms of their Poincare disk coordiantes
    """
    num = abs(z1-z2)
    denom = abs(1-z1*z2.conjugate())
    return 2*math.atanh(num/denom)
