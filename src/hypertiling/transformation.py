import math
from hypertiling.arithmetics import htcplxadd, htcplxdiff, htcplxdiv, htcplxprod, htcplxprodconjb
from hypertiling.check_numba import NumbaChecker


@NumbaChecker("complex128(float64, complex128)")
def moeb_rotate_trafo(phi, z):
    '''Rotates z by phi counter-clockwise about the origin.'''
    return z * complex(math.cos(phi), math.sin(phi))


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128)"])
def mymoebint(z0, z):
    '''Internal function for performing a full Möbius transform in double-double representation.'''
    dz0 = complex(0, 0)
    dz = complex(0, 0)
    one = complex(1, 0)
    done = complex(0, 0)
    nom, dnom = htcplxadd(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxadd(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret

@NumbaChecker("complex128(complex128, complex128)")
def moeb_origin_trafo(z0, z):
    """
    Maps all points z such that z0 -> 0, respecting the Poincare projection: (z - z0)/(1 - z0 * z)
    
    Parameters:
        z0 (complex): the origin that we map back to.
        z (complex): the point that we will tr

    
    Returns:
        ret (complex): z Möbius transformed around z0: (z - z0)/(1 - z0 * z)
    """
    ret, dret = mymoebint(-z0, z)
    return ret


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def moeb_origin_trafodd(z0, dz0, z, dz):
    '''Möbius transform to the origin in double double representation'''
    one = complex(1, 0)
    done = complex(0, 0)
    nom, dnom = htcplxdiff(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxdiff(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, float64)"])
def moeb_rotate_trafodd(z, dz, phi):
    '''Rotation of a complex number'''
    ep = complex(math.cos(phi), math.sin(phi))
    ep = ep / abs(ep)  # We calculated sin and cos separately. We can't be sure that |ep| == 1
    dep = complex(0, 0)
    ret, dret = htcplxprod(z, dz, ep, dep)
    return ret, dret


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def moeb_origin_trafo_inversedd(z0, dz0, z, dz):
    '''Inverse Möbius transform to the origin in double double representation'''
    one = complex(1, 0)
    done = complex(0, 0)
    nom, dnom = htcplxadd(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxadd(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret
