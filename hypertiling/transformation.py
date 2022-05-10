import numpy as np
import math
try:
   import numba
except ImportError:
   pass


@numba.njit
def kahan(x, y):
   r = x + y
   e = y - (r - x)
   return r, e

@numba.njit
def twosum(x, y):
   r = x + y
   t = r - x
   e = (x - (r - t)) + (y - t)
   return r, e

@numba.njit
def twodiff(x, y):
   r = x - y
   t = r - x
   e = (x - (r - t)) - (y + t)
   return r, e

@numba.njit
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

@numba.njit
def htadd(x, dx, y, dy): # hypertilingadd
   r, e = twosum(x, y)
   e += dx + dy
   r, e = kahan(r, e)
   return r, e

@numba.njit
def htdiff(x, dx, y, dy):
   r, e = twodiff(x, y)
   e += dx - dy
   r, e = kahan(r, e)
   return r, e

@numba.njit
def htprod(x, dx, y, dy):
   r, e = twoproduct(x, y)
   e += x * dy + y*dx
   r, e = kahan(r, e)
   return r, e

@numba.njit
def htdiv(x, dx, y, dy):
   r = x/y
   s, f = twoproduct(r, y)
   e = (x - s - f + dx - r*dy)/y
   r, e = kahan(r, e)
   return r, e

@numba.njit
def htcplxprod(a, da, b, db):
  rea, drea = a.real, da.real
  ima, dima = a.imag, da.imag
  reb, dreb = b.real, db.real
  imb, dimb = b.imag, db.imag

#   We employ the Gauss/Karatsuba trick
#   (ar + I * ai)*(br + I*bi) = ar*br - ai*bi + I*[ (ar + ai)*(br + bi) - ar*br - ai*bi ]
  r, dr = htprod(rea, drea, reb, dreb) # ar*br
  i, di = htprod(ima, dima, imb, dimb) # ai*bi

  fac1, dfac1 = htadd(rea, drea, ima, dima)
  fac2, dfac2 = htadd(reb, dreb, imb, dimb)
  imacc, dimacc = htprod(fac1, dfac1, fac2, dfac2)
  imacc, dimacc = htdiff(imacc, dimacc, r, dr)
  imacc, dimacc = htdiff(imacc, dimacc, i, di)

  r, dr = htdiff(r, dr, i, di)
  return complex(r, imacc), complex(dr, dimacc)

@numba.njit
def htcplxprodconjb(a, da, b, db):
  rea, drea = a.real, da.real
  ima, dima = a.imag, da.imag
  reb, dreb = b.real, db.real
  imb, dimb = b.imag, db.imag

#   We employ the Gauss/Karatsuba trick
#   (ar + I * ai)*(br - I*bi) = ar*br + ai*bi + I*[ (ar + ai)*(br - bi) - ar*br + ai*bi ]
  r, dr = htprod(rea, drea, reb, dreb) # ar*br
  i, di = htprod(ima, dima, imb, dimb) # ai*bi

  fac1, dfac1 = htadd(rea, drea, ima, dima)
  fac2, dfac2 = htdiff(reb, dreb, imb, dimb)
  imacc, dimacc = htprod(fac1, dfac1, fac2, dfac2)
  imacc, dimacc = htdiff(imacc, dimacc, r, dr)
  imacc, dimacc = htadd(imacc, dimacc, i, di)

  r, dr = htadd(r, dr, i, di)
  return complex(r, imacc), complex(dr, dimacc)

@numba.njit
def htcplxadd(a, da, b, db):
  rea, drea = a.real, da.real
  ima, dima = a.imag, da.imag
  reb, dreb = b.real, db.real
  imb, dimb = b.imag, db.imag

  r, dr = htadd(rea, drea, reb, dreb)
  i, di = htadd(ima, dima, imb, dimb)
  return complex(r, i), complex(dr, di)

@numba.njit
def htcplxdiff(a, da, b, db):
  rea, drea = a.real, da.real
  ima, dima = a.imag, da.imag
  reb, dreb = b.real, db.real
  imb, dimb = b.imag, db.imag

  r, dr = htdiff(rea, drea, reb, dreb)
  i, di = htdiff(ima, dima, imb, dimb)
  return complex(r, i), complex(dr, di)

@numba.njit
def htcplxdiv(a, da, b, db):
   rea, drea = a.real, da.real
   ima, dima = a.imag, da.imag
   reb, dreb = b.real, db.real
   imb, dimb = b.imag, db.imag
#    We make the denominator real.
#    Hence we calculate the denominator and the nominator separately
#    first the denominator: br^2 + bi^2
   denom, ddenom = htprod(reb, dreb, reb, dreb)
   t1, dt1 = htprod(imb, dimb, imb, dimb)
   denom, ddenom = htadd(denom, ddenom , t1, dt1)

#    Now on to the numerator
   nom, dnom = htcplxprodconjb(a, da, b, db)

   r, dr = htdiv(nom.real, dnom.real, denom, ddenom) 
   i, di = htdiv(nom.imag, dnom.imag, denom, ddenom)

   return complex(r, i), complex(dr, di)

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
    ret, dret = mymoebint(-z0, z)
    return ret

def moeb_origin_trafo_inverse_py(z0, z):
    ret, dret = mymoebint(z0, z)
    return ret

 # rotates z by phi counter-clockwise about the origin
def moeb_rotate_trafo_py(z, phi): 
    return z * complex(math.cos(phi), math.sin(phi))

def mymoebddint_py(z0, z):
    dz0 = complex(0,0)
    dz = complex(0,0)
    one = complex(1,0)
    done = complex(0,0)
    nom, dnom = htcplxadd(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxadd(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret

# If numba is present we use the numba compiled functions, else the plain ones.
try:
    import numba
    p2w = numba.njit(p2w_py)
    w2p = numba.njit(w2p_py)
    moeb_origin_trafo = numba.njit(moeb_origin_trafo_py)
    moeb_origin_trafo_inverse = numba.njit(moeb_origin_trafo_inverse_py)
    moeb_rotate_trafo = numba.njit(moeb_rotate_trafo_py)
    mymoebint = numba.njit(mymoebddint_py)
    mymoeb = numba.njit(mymoeb_py)
except ImportError:
    p2w = p2w_py
    w2p = w2p_py
    mymoeb = mymoeb_py
    moeb_origin_trafo = moeb_origin_trafo_py
    moeb_origin_trafo_inverse = moeb_origin_trafo_inverse_py
    moeb_rotate_trafo = moeb_rotate_trafo_py

@numba.njit
def mymoebdd(z0, dz0, z, dz):
    one = complex(1,0)
    done = complex(0,0)
    nom, dnom = htcplxadd(z, dz, z0, dz0)
    denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
    denom, ddenom = htcplxadd(one, done, denom, ddenom)
    ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
    return ret, dret

@numba.njit
def moeb_origin_trafodd(z0, dz0, z, dz):
   one = complex(1,0)
   done = complex(0,0)
   nom, dnom = htcplxdiff(z, dz, z0, dz0)
   denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
   denom, ddenom = htcplxdiff(one, done, denom, ddenom)
   ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
   return ret, dret
   #return mymoebdd(-z0, -dz0, z, dz)

@numba.njit
def moeb_origin_trafo_inversedd(z0, dz0, z, dz):
   one = complex(1,0)
   done = complex(0,0)
   nom, dnom = htcplxadd(z, dz, z0, dz0)
   denom, ddenom = htcplxprodconjb(z, dz, z0, dz0)
   denom, ddenom = htcplxadd(one, done, denom, ddenom)
   ret, dret = htcplxdiv(nom, dnom, denom, ddenom)
   return ret, dret
   #return mymoebdd(z0, dz0, z, dz)



@numba.njit
def moeb_rotate_trafodd(z, dz, phi):
   ep = complex(math.cos(phi), math.sin(phi))
   ep = ep/np.abs(ep)
   dep = complex(0,0)
   ret, dret = htcplxprod(z, dz, ep, dep)
#   print(np.abs(ep)- 1)
   return ret, dret

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

