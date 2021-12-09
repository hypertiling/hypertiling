import numpy as np
import math


def weierstrass_distance(a, b):
    arg = a[2] * b[2] - a[1] * b[1] - a[0] * b[0]
    if arg < 1:
        return 0
    else:
        return math.acosh(arg) 
        # for scalars math.acosh is usually faster than np.arccosh


def disk_distance(z1, z2):
    num = np.abs(z1-z2)
    denom = np.abs(1-z1*np.conj(z2))
    return 2*np.arctanh(num/denom)
