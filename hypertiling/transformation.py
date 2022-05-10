import numpy as np
import math

def p2w_py(z):
    x, y = z.real, z.imag
    xx = x*x
    yy = y*y
    factor = 1 / (1-xx-yy)
    return factor*np.array([(1+xx+yy), 2*x, 2*y])

def w2p_py(point):
    [t, x, y] = point
    factor = 1 / (1+t)
    return complex(x*factor, y*factor)

def mymoeb_py(z0, z):
    rez, imz = z.real, z.imag
    rez0, imz0 = z0.real, z0.imag
    return (z+z0) / (1+z*np.conjugate(z0))#complex(math.fsum([1, rez*rez0, imz*imz0]), imz*rez0-imz0*rez)# (1+z*np.conjugate(z0))

# maps all points z such that z0 -> 0, respecting the Poincare projection

def moeb_origin_trafo_py(z0, z):
    return mymoeb(-z0, z)

def moeb_origin_trafo_inverse_py(z0, z):
    return mymoeb(z0, z)

 # rotates z by phi counter-clockwise about the origin
def moeb_rotate_trafo_py(z, phi): 
    return z * complex(math.cos(phi), math.sin(phi))

# If numba is present we use the numba compiled functions, else the plain ones.
try:
    import numba
    p2w = numba.njit(p2w_py)
    w2p = numba.njit(w2p_py)
    mymoeb = numba.njit(mymoeb_py)
    moeb_origin_trafo = numba.njit(moeb_origin_trafo_py)
    moeb_origin_trafo_inverse = numba.njit(moeb_origin_trafo_inverse_py)
    moeb_rotate_trafo = numba.njit(moeb_rotate_trafo_py)
except ImportError:
    p2w = p2w_py
    w2p = w2p_py
    mymoeb = mymoeb_py
    moeb_origin_trafo = moeb_origin_trafo_py
    moeb_origin_trafo_inverse = moeb_origin_trafo_inverse_py
    moeb_rotate_trafo = moeb_rotate_trafo_py

def moeb_translate_trafo(z, s):
    num = z-s
    denom = 1-z*s
    return num/denom

# reverses the previous three transformations at once
def moeb_inverse_trafo(z, z0, phi, s):  
    exp = complex(math.cos(phi), math.sin(phi))
    z0c = z0.conjugate()
    num = s+z+exp*z0*(1+s*z)
    denom = exp*(1+s*z)+z0c*(s+z)
    return num/denom

