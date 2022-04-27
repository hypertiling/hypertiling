import numpy as np
import math

def kahan(x, y):
    r = x + y
    e = y - (r - x)
    return r, e

def twosum(x, y):
    r = x + y
    t = r - x
    e = (x - (r - t)) + (y - t)
    return r, e

def twodiff(x, y):
    r = x - y
    t = r - x
    e = (x - (r - t)) - (y + t)
    return r, e

def twoproduct(x, y):
    u = x*134217729.0
    v = y*134217729.0
    s = u - (u - x)
    t = v - (v - y)
    f = x - s
    g = y - t
    r = x*y
    e = ((s*t - r) + s*g + f*t) + f*g
    return r, e

def htadd(x, dx, y, dy):
    r, e = twosum(x, y)
    e += dx + dy
    r, e = kahan(r, e)
    return r, e

def htdiff(x, dx, y, dy):
    r, e = twodiff(x, y)
    e += dx - dy
    r, e = kahan(r, e)
    return r, e

def htprod(x, dx, y, dy):
    r, e = twoproduct(x, y)
    e += x * dy + y*dx
    r, e = kahan(r, e)
    return r, e

def htdiv(x, dx, y, dy):
    r = x/y
    s, f = twoproduct(r, y)
    e = (x - s - f + dx - r*dy)/y
    r, e = kahan(r, e)
    return r, e

def p2w(z):
    x, y = z.real, z.imag
    xx = x*x
    yy = y*y
    factor = 1 / (1-xx-yy)
    return factor*np.array([(1+xx+yy), 2*x, 2*y])


def w2p(point):
    [t, x, y] = point
    factor = 1 / (1+t)
    return complex(x*factor, y*factor)

def mymoeb(z0, z):
    rez, imz = z.real, z.imag
    rez0, imz0 = z0.real, z0.imag
    return (z+z0) / complex(math.fsum([1, rez*rez0, imz*imz0]), imz*rez0-imz0*rez)
# (1+z*np.conjugate(z0))

# maps all points z such that z0 -> 0, respecting the Poincare projection
def moeb_origin_trafo(z0, z):
    return mymoeb(-z0, z)

def moeb_origin_trafo_inverse(z0, z):
    return mymoeb(z0, z)


 # rotates z by phi counter-clockwise about the origin
def moeb_rotate_trafo(z, phi): 
    return z * np.exp(complex(0, phi))


def moeb_translate_trafo(z, s):
    num = z-s
    denom = 1-z*s
    return num/denom

# reverses the previous three transformations at once
def moeb_inverse_trafo(z, z0, phi, s):  
    exp = np.exp(complex(0, phi))
    z0c = np.conjugate(z0)
    num = s+z+exp*z0*(1+s*z)
    denom = exp*(1+s*z)+z0c*(s+z)
    return num/denom

