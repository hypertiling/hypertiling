import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from .transformation import moeb_origin_trafo
from .distance import disk_distance
from .ion import htprint

# Helpers to construct geodesic lines

def minor(M, i, j):
    """
    Return the "minor" of a matrix with respect to index (i,j).

    Parameters
    ----------
    M : ndarray
        Input matrix.
    i : int
        Row index to be deleted.
    j : int
        Column index to be deleted.

    Returns
    -------
    M : ndarray
        Minor of the input matrix.

    """
    M = np.delete(M, i, 0)
    M = np.delete(M, j, 1)
    return M


def unit_circle_inversion(z):
    """
    Perform inversion of input z with respect to the unit circle.

    Parameters
    ----------
    z : complex
        Complex number to be inverted.

    Returns
    -------
    z : complex
        Inverted complex number.

    """
    denom = z.real**2 + z.imag**2
    return complex(z.real/denom, z.imag/denom)


def circle_through_three_points(z1, z2, z3, verbose=False, eps=1e-10):
    """
    Construct Euclidean circle through three points (z1, z2, z3).

    Calculates the center and radius of the circle passing through the three points.
    In case the points are collinear within a precision of "eps", a radius of -1 is returned.

    Parameters
    ----------
    z1 : complex
        First point represented as a complex number.
    z2 : complex
        Second point represented as a complex number.
    z3 : complex
        Third point represented as a complex number.
    verbose : bool, optional
        If True, prints a warning when the points are collinear. Default is False.
    eps : float, optional
        Precision for collinearity check. Default is 1e-10.

    Returns
    -------
    complex
        Center of the circle represented as a complex number.
    float
        Radius of the circle. Returns -1 if the points are collinear.

    References
    ----------
    Formulas used are derived from:
        http://web.archive.org/web/20161011113446/http://www.abecedarical.com/zenosamples/zs_circle3pts.html
    """
    x1 = z1.real
    y1 = z1.imag
    x2 = z2.real
    y2 = z2.imag
    x3 = z3.real
    y3 = z3.imag
    
    a1 = np.array([0, 0, 0, 1])
    a2 = np.array([x1*x1+y1*y1, x1, y1, 1])
    a3 = np.array([x2*x2+y2*y2, x2, y2, 1])
    a4 = np.array([x3*x3+y3*y3, x3, y3, 1])
    
    A = np.stack([a1, a2, a3, a4])
    
    M00 = np.linalg.det(minor(A, 0, 0))
    M01 = np.linalg.det(minor(A, 0, 1))
    M02 = np.linalg.det(minor(A, 0, 2))
    M03 = np.linalg.det(minor(A, 0, 3))
    
    # M00 being close to zero indicates collinearity
    if np.abs(M00) < eps:
        if verbose:
            htprint("Warning", "Points are collinear! A radius of -1 is returned.")
        return complex(0, 0), -1

    # compute center and radius
    x0 = 0.5 * M01 / M00
    y0 = - 0.5 * M02 / M00
    radius = np.sqrt(x0*x0 + y0*y0 + M03 / M00)
    
    return complex(x0, y0), radius


def geodesic_midpoint(z1, z2):
    """
    Compute the geodesic midpoint between two complex numbers, `z1` and `z2`.

    The function first applies a Möbius transformation to move `z1` and `z2`
    such that `z1` is at the origin (0). It then calculates the distance 
    between the origin and the new location of `z2` (`z2n`). 
    This distance is transformed into a Cartesian radius `r` using the hyperbolic tangent function.
    An angle is added to `r` (using Euler's formula), which is then transformed back 
    to the original location by applying the inverse Möbius transformation.

    Parameters
    ----------
    z1 : complex
        First point on the Poincare disk
    z2 : complex
        Second point on the Poincare disk

    Returns
    -------
    complex
        The geodesic midpoint of `z1` and `z2`.

    """
    z1c = z1.conjugate()
    z2c = z2.conjugate()
    
    a = 1 - z1*z1c
    b = 1 - z1*z2c
    c = 1 - z2*z1c
    d = 1 - z2*z2c
    sqrt = np.sqrt(a*b*c*d)
    num = 1 - z1*z1c*z2*z2c - sqrt
    den = z1c + z2c - (z1 + z2)*z1c*z2c
    
    return np.where(den != 0, num/den, 0+0j)


def geodesic_angles(z1, z2):
    """
    Compute the angles of a geodesic between two complex numbers `z1` and `z2`.

    This function serves as a helper for "geodesic_arc". It first checks if `z1` is not too close to the origin,
    and if so, calculates the inverse of `z1` on the unit circle and computes the circle that passes through `z1`, `z2`, and the inversion point.
    If `z1` is too close to the origin, it sets the center of the circle to infinity and the radius to -1.

    If the calculated radius is -1 (indicating that the points are collinear), it returns 0 for both angles, the center of the circle, and the radius.
    Otherwise, it calculates the angles from the center of the circle to `z1` and `z2` and returns these angles, the center of the circle, and the radius.

    Parameters
    ----------
    z1 : complex
        First point on the Poincare disk
    z2 : complex
        Second point on the Poincare disk

    Returns
    -------
    angle1 : float
        The angle from the center of the circle to `z1`.
    angle2 : float
        The angle from the center of the circle to `z2`.
    zc : complex
        The center of the circle.
    radius : float
        The radius of the circle. If the points are collinear, this is -1.

    Notes
    -----
    The origin needs some extra care since it is mapped to infinity.
    In case points are collinear, a radius of -1 is returned.
    """
    
    # handle points close to the origin
    if np.abs(z1) > 1e-14:
        z3 = unit_circle_inversion(z1)
        zc, radius = circle_through_three_points(z1, z2, z3)
    else:
        zc = np.inf
        radius = -1
    
    # in case points are collinear, return a radius of -1
    if radius == -1:
        return 0, 0, 0, -1

    ax = z1.real-zc.real
    ay = z1.imag-zc.imag
    bx = z2.real-zc.real
    by = z2.imag-zc.imag

    angle1 = np.arctan2(by, bx)
    angle2 = np.arctan2(ay, ax)
    
    return angle1, angle2, zc, radius


def geodesic_arc(z1, z2, **kwargs):
    """
    Returns a hyperbolic line segment connecting z1 and z2 as a matplotlib drawing object.

    If the points are collinear, a straight line is drawn using matplotlib.patch.Arrow. 
    Otherwise, a hyperbolic arc is drawn using matplotlib.patch.Arc.

    Parameters
    ----------
    z1 : complex
        First point represented as a complex number.
    z2 : complex
        Second point represented as a complex number.
    **kwargs : dict
        Additional parameters to be passed to the drawing function.

    Returns
    -------
    matplotlib.lines.Line2D or matplotlib.patches.Arc
        A matplotlib object representing the line or arc.

    Notes
    -----
    The function uses the geodesic_angles function to compute necessary parameters 
    for the arc. It also performs angle normalization to ensure that negative angles 
    are correctly handled.
    """
    t1, t2, zc, r = geodesic_angles(z1, z2)

    # in case the points are collinear, we use matplotlib.patch.Arrow to draw a straight line
    if r == -1:
        # line elements do not know "edgecolor", hence we rename it to "color"
        linekwargs = kwargs
        if "ec" in linekwargs:
            kwargs["color"] = kwargs.pop("ec")
        if "edgecolor" in linekwargs:
            kwargs["color"] = kwargs.pop("edgecolor")

        return mlines.Line2D(np.array([z1.real, z2.real]), np.array([z1.imag, z2.imag]), **linekwargs)
    
    # avoid negative angles
    if t1 < 0:
        t1 = 2*np.pi + t1
            
    if t2 < 0:
        t2 = 2*np.pi + t2
    
    # some gymnastics to always draw the "inner" arc
    # i.e. the one fully inside the unit circle
    t = np.sort([t1, t2])
    t1 = t[0]
    t2 = t[1]
    dt1 = t2-t1
    dt2 = t1-t2+2*np.pi
    
    # draw hyperbolic arc connection z1 and z2 as a matplotlib.patch.Arc
    if dt1<dt2:
        return mpatches.Arc((np.real(zc), np.imag(zc)), 2*r, 2*r, theta1=np.degrees(t1), theta2=np.degrees(t2), **kwargs)
    else:
        return mpatches.Arc((np.real(zc), np.imag(zc)), 2*r, 2*r, theta1=np.degrees(t2), theta2=np.degrees(t1), **kwargs)
