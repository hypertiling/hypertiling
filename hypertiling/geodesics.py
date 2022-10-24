import numpy as np

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from .transformation import moeb_origin_trafo
from .distance import disk_distance


# returns the "minor" of a matrix
def minor(M, i, j):
    M = np.delete(M, i, 0)
    M = np.delete(M, j, 1)
    return M


# perform inversion of "z" w.r.t. the unit circle
def unit_circle_inversion(z):
    denom = z.real**2 + z.imag**2
    return complex(z.real/denom, z.imag/denom)


# constructs a circle through three points
# input: three points represented as complex numbers
# output: center of the circle and radius
# if the points are collinear within a certain precision
# the functions returns a radius of -1

# formulas from here: 
# http://web.archive.org/web/20161011113446/http://www.abecedarical.com/zenosamples/zs_circle3pts.html

def circle_through_three_points(z1, z2, z3, verbose=False):
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
    if np.abs(M00) < 1e-10:
        if verbose:
            print("Error: Points are collinear!")
        return complex(0, 0), -1

    # compute center and radius
    x0 = 0.5 * M01 / M00
    y0 = - 0.5 * M02 / M00
    radius = np.sqrt(x0*x0 + y0*y0 + M03 / M00)
    
    return complex(x0, y0), radius


# return the geodesic midpoint between z1 and z2
def geodesic_midpoint(z1, z2):
    z2n = moeb_origin_trafo(z1, z2)  # move z1, z2 such that z0=0
    d = disk_distance(0, z2n)  # distance betwen 0 and z2new
    r = np.tanh(d/4)  # compute corresponding Cartesian radius
    zm = r*np.exp(1j*np.angle(z2n))  # add angle
    zm = moeb_origin_trafo(-z1, zm)  # and transform back
    return zm


# helper function for "geodesic_arc"
def geodesic_angles(z1, z2):
    z3 = unit_circle_inversion(z1)
    zc, radius = circle_through_three_points(z1, z2, z3)
    
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


# draw hyperbolic line segment connecting z1 and z2
def geodesic_arc(z1, z2, **kwargs):
    t1, t2, zc, r = geodesic_angles(z1, z2)

    # in case the points are collinear, we use matplotlib.patch.Arrow to draw a straight line
    if r == -1:
        return mlines.Line2D(np.array([z1.real, z2.real]), np.array([z1.imag, z2.imag]), **kwargs)
    
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
        return mpatches.Arc((np.real(zc), np.imag(zc)), 2*r, 2*r, 0, theta1=np.degrees(t1), theta2=np.degrees(t2), **kwargs)
    else:
        return mpatches.Arc((np.real(zc), np.imag(zc)), 2*r, 2*r, 0, theta1=np.degrees(t2), theta2=np.degrees(t1), **kwargs)


