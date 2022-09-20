import math
from numpy import array as nparray
from hypertiling.check_numba import check_numba


@check_numba
def kahan(x, y):
    """
    Transform the addition of two floating point numbers:

    .. math::

        x + y = r + e


    (Dekker1971) showed that this transform is exact, if abs(x) > abs(y).

    Parameters:
        x (float): a floating point number.
        y (float): a floating point number with abs(y) < abs(x).

    Returns:
        r (float): x + y
        e (float): the overflow
    """
    r = x + y
    e = y - (r - x)
    return r, e

@check_numba
def twosum(x, y):
    '''branch free transformation of addition by Knuth'''
    r = x + y
    t = r - x
    e = (x - (r - t)) + (y - t)
    return r, e

@check_numba
def twodiff(x, y):
    '''branch free transformation of subtraction'''
    r = x - y
    t = r - x
    e = (x - (r - t)) - (y + t)
    return r, e

@check_numba
def twoproduct(x, y):
    '''Product of two numbers: x*y = r + e. See Ogita et al. 2005'''
    u = x * 134217729.0  # Split input x
    v = y * 134217729.0  # Split input y
    s = u - (u - x)
    t = v - (v - y)
    f = x - s
    g = y - t
    r = x * y
    e = ((s * t - r) + s * g + f * t) + f * g
    return r, e

@check_numba
def htadd(x, dx, y, dy):  # double double add
    '''perform addition of numbers given in double double representation '''
    r, e = twosum(x, y)
    e += dx + dy
    r, e = kahan(r, e)
    return r, e

@check_numba
def htdiff(x, dx, y, dy):
    '''perform subtraction of numbers given in double double representation '''
    r, e = twodiff(x, y)
    e += dx - dy
    r, e = kahan(r, e)
    return r, e

@check_numba
def htprod(x, dx, y, dy):
    '''perform multplication of numbers given in double double representation '''
    r, e = twoproduct(x, y)
    e += x * dy + y * dx
    r, e = kahan(r, e)
    return r, e

@check_numba
def htdiv(x, dx, y, dy):
    '''perform division of numbers given in double double representation '''
    r = x / y
    s, f = twoproduct(r, y)
    e = (x - s - f + dx - r * dy) / y  # Taylor expansion
    r, e = kahan(r, e)
    return r, e

@check_numba
def htcplxprod(a, da, b, db):
    '''perform multiplication of complex double double numbers '''
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    #   We employ the Gauss/Karatsuba trick
    #   (ar + I * ai)*(br + I*bi) = ar*br - ai*bi + I*[ (ar + ai)*(br + bi) - ar*br - ai*bi ]
    r, dr = htprod(rea, drea, reb, dreb)  # ar*br
    i, di = htprod(ima, dima, imb, dimb)  # ai*bi

    fac1, dfac1 = htadd(rea, drea, ima, dima)
    fac2, dfac2 = htadd(reb, dreb, imb, dimb)
    imacc, dimacc = htprod(fac1, dfac1, fac2, dfac2)
    imacc, dimacc = htdiff(imacc, dimacc, r, dr)
    imacc, dimacc = htdiff(imacc, dimacc, i, di)

    r, dr = htdiff(r, dr, i, di)
    return complex(r, imacc), complex(dr, dimacc)

@check_numba
def htcplxprodconjb(a, da, b, db):
    '''perform multiplication of complex double double numbers: a * b^* '''
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    #   We employ the Gauss/Karatsuba trick
    #   (ar + I * ai)*(br - I*bi) = ar*br + ai*bi + I*[ (ar + ai)*(br - bi) - ar*br + ai*bi ]
    r, dr = htprod(rea, drea, reb, dreb)  # ar*br
    i, di = htprod(ima, dima, imb, dimb)  # ai*bi

    fac1, dfac1 = htadd(rea, drea, ima, dima)
    fac2, dfac2 = htdiff(reb, dreb, imb, dimb)
    imacc, dimacc = htprod(fac1, dfac1, fac2, dfac2)
    imacc, dimacc = htdiff(imacc, dimacc, r, dr)
    imacc, dimacc = htadd(imacc, dimacc, i, di)

    r, dr = htadd(r, dr, i, di)
    return complex(r, imacc), complex(dr, dimacc)

@check_numba
def htcplxadd(a, da, b, db):
    '''perform addition of complex double double numbers '''
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    r, dr = htadd(rea, drea, reb, dreb)
    i, di = htadd(ima, dima, imb, dimb)
    return complex(r, i), complex(dr, di)

@check_numba
def htcplxdiff(a, da, b, db):
    '''perform subtraction of complex double double numbers '''
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    r, dr = htdiff(rea, drea, reb, dreb)
    i, di = htdiff(ima, dima, imb, dimb)
    return complex(r, i), complex(dr, di)

@check_numba
def htcplxdiv(a, da, b, db):
    '''perform division of complex double double numbers '''
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag
    #    We make the denominator real.
    #    Hence we calculate the denominator and the nominator separately
    #    first the denominator: br^2 + bi^2
    denom, ddenom = htprod(reb, dreb, reb, dreb)
    t1, dt1 = htprod(imb, dimb, imb, dimb)
    denom, ddenom = htadd(denom, ddenom, t1, dt1)

    #    Now on to the numerator
    nom, dnom = htcplxprodconjb(a, da, b, db)

    r, dr = htdiv(nom.real, dnom.real, denom, ddenom)
    i, di = htdiv(nom.imag, dnom.imag, denom, ddenom)

    return complex(r, i), complex(dr, di)

@check_numba
def p2w(z):
    '''Convert Poincare to Weierstraß representation '''
    x, y = z.real, z.imag
    xx = x * x
    yy = y * y
    factor = 1 / (1 - xx - yy)
    return factor * nparray([(1 + xx + yy), 2 * x, 2 * y])

@check_numba
def w2p(point):
    '''Convert Weierstraß to Poincare representation '''
    [t, x, y] = point
    factor = 1 / (1 + t)
    return complex(x * factor, y * factor)

@check_numba
def mymoeb(z0, z):
    rez, imz = z.real, z.imag
    rez0, imz0 = z0.real, z0.imag
    return (z + z0) / (
                1 + z * z0.conjugate())  # complex(math.fsum([1, rez*rez0, imz*imz0]), imz*rez0-imz0*rez)# (1+z*np.conjugate(z0))


# maps all points z such that z0 -> 0, respecting the Poincare projection
@check_numba
def moeb_origin_trafo(z0, z):
    ret, dret = mymoebint(-z0, z)
    return ret

@check_numba
def moeb_origin_trafo_inverse(z0, z):
    ret, dret = mymoebint(z0, z)
    return ret


# rotates z by phi counter-clockwise about the origin
@check_numba
def moeb_rotate_trafo(phi, z):
    return z * complex(math.cos(phi), math.sin(phi))

@check_numba
def mymoebint(z0, z):
    dz0 = complex(0, 0)
    dz = complex(0, 0)
    one = complex(1, 0)
    done = complex(0, 0)
    nom, dnom = htcplxadd(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxadd(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret


@check_numba
def moeb_origin_trafodd(z0, dz0, z, dz):
    '''Möbius transform to the origin in double double representation'''
    one = complex(1, 0)
    done = complex(0, 0)
    nom, dnom = htcplxdiff(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxdiff(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret


@check_numba
def moeb_rotate_trafodd(z, dz, phi):
    '''Rotation of a complex number'''
    ep = complex(math.cos(phi), math.sin(phi))
    ep = ep / abs(ep)  # We calculated sin and cos separately. We can't be sure that |ep| == 1
    dep = complex(0, 0)
    ret, dret = htcplxprod(z, dz, ep, dep)
    return ret, dret


@check_numba
def moeb_origin_trafo_inversedd(z0, dz0, z, dz):
    '''Inverse Möbius transform to the origin in double double representation'''
    one = complex(1, 0)
    done = complex(0, 0)
    nom, dnom = htcplxadd(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxadd(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret





def moeb_translate_trafo(z, s):
    num = z - s
    denom = 1 - z * s
    return num / denom


# reverses the previous three transformations at once

def moeb_inverse_trafo(z, z0, phi, s):
    exp = complex(math.cos(phi), math.sin(phi))
    z0c = z0.conjugate()
    num = s + z + exp * z0 * (1 + s * z)
    denom = exp * (1 + s * z) + z0c * (s + z)
    return num / denom
