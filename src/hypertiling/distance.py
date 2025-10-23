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


def poincare_distance(z1: complex, z2: complex) -> float:
    """
    Compute the hyperbolic distance between two points in the Poincaré disk model.

    The Poincaré disk represents the hyperbolic plane as the open unit disk
    :math:`|z| < 1` with metric:
        d(z1, z2) = 2 * atanh( |z1 - z2| / |1 - z1 * conj(z2)| )

    Parameters
    ----------
    z1 : complex
        First point in Poincaré disk coordinates (|z1| < 1).
    z2 : complex
        Second point in Poincaré disk coordinates (|z2| < 1).

    Returns
    -------
    float
        Hyperbolic distance :math:`d(z1, z2)` between the two points.

    Raises
    ------
    ValueError
        If either point lies outside or on the boundary of the Poincaré disk (|z| ≥ 1).

    Notes
    -----
    - Returns 0 if z1 and z2 coincide.
    - Numerically clips the argument of atanh to [0, 1-1e-15] to prevent
      floating-point overshoots for points close to the boundary.
    - The distance grows unbounded as either point approaches |z| → 1.
    """
    if z1 == z2:
        return 0.0
    num = abs(z1 - z2)
    denom = abs(1 - z1 * z2.conjugate())

    # both points must be inside the open unit disk
    if abs(z1) >= 1 or abs(z2) >= 1:
        raise ValueError("Points must lie strictly inside the Poincaré disk (|z|<1).")

    x = num / denom
    # numerical safety: clip near-boundary overshoots
    x = min(max(x, 0.0), 1 - 1e-15)
    return 2.0 * math.atanh(x)

