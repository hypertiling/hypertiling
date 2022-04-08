import numpy as np


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


# maps all points z such that z0 -> 0, respecting the Poincare projection
def moeb_origin_trafo(z0, z):
    return (z-z0) / (1-z*np.conjugate(z0)) 

def moeb_origin_trafo_inverse(z0, z):
    return (z+z0) / (1+z*np.conjugate(z0))


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

