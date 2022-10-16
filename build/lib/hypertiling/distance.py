import numpy as np
import math

global signature
signature = np.array([1,-1,-1]) # signature of the embedding Minkowski space 


# computes the inner product between a and b, respecting the Minkowskian signature
# b needs to be a 1-D array of length 3
# if both a and b are 1-D arrays, a scalar is returned
# if a is a 2-D array of shape (N,3) an array of length N is returned
def lorentzian_distance(a,b):
    return np.dot(a, np.multiply(signature,b))


def weierstrass_distance(a, b):
    arg = lorentzian_distance(a,b)
    if arg < 1:
        return 0
    else:
        # for scalars math.acosh is usually faster than np.arccosh
        return math.acosh(arg)


def disk_distance(z1, z2):
    num = abs(z1-z2)
    denom = abs(1-z1*z2.conjugate())
    return 2*math.atanh(num/denom)
