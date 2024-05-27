from hypertiling.check_numba import NumbaChecker


@NumbaChecker(["UniTuple(float64, 2)(float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:])",
               "UniTuple(float32, 2)(float32, float32)",
               "UniTuple(float32[:], 2)(float32[:], float32[:])"])
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


@NumbaChecker(["UniTuple(float64, 2)(float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:])",
               "UniTuple(float32, 2)(float32, float32)",
               "UniTuple(float32[:], 2)(float32[:], float32[:])"])
def twosum(x, y):
    """
    Perform a branch-free transformation of addition according to Knuth's method.
    
    This function computes the sum of `x` and `y` in a branch-free manner, which can 
    be beneficial for performance on certain hardware architectures. The method follows 
    the approach proposed by Donald Knuth.

    Parameters
    ----------
    x : float or float[:]
        The first addend in the addition operation.
    y : float or float[:]
        The second addend in the addition operation.

    Returns
    -------
    r : float or float[:]
        The sum of `x` and `y`.
    e : float or float[:]
        The error term of the addition.

    """
    r = x + y
    t = r - x
    e = (x - (r - t)) + (y - t)
    return r, e



@NumbaChecker(["UniTuple(float64, 2)(float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:])",
               "UniTuple(float32, 2)(float32, float32)",
               "UniTuple(float32[:], 2)(float32[:], float32[:])"])
def twodiff(x, y):
    """
    Perform a branch-free transformation of subtraction.
    
    This function computes the difference (x - y) in a way that does not 
    involve conditional branches, which can be beneficial for performance 
    on certain hardware architectures.

    Parameters
    ----------
    x : float or float[:]
        The minuend of the subtraction operation.
    y : float or float[:]
        The subtrahend of the subtraction operation.

    Returns
    -------
    r : float or float[:]
        The result of the subtraction (x - y).
    e : float or float[:]
        The error term of the subtraction.

    """
    r = x - y
    t = r - x
    e = (x - (r - t)) - (y + t)
    return r, e




@NumbaChecker(["UniTuple(float64, 2)(float64, float64, float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], float64[:])",
               "UniTuple(float32, 2)(float32, float32, float32, float32)",
               "UniTuple(float32[:], 2)(float32[:], float32[:], float32[:], float32[:])"])
def htadd(x, dx, y, dy):
    """
    Perform addition of numbers (x,dx) and (y,dy) given in double double representation.

    Parameters:
        x  (float): a floating point number.
        dx (float): overflow of x
        y  (float): a floating point number.
        dy (float): overflow of y


    Returns:
        r (float): x + y + (dx + dy)
        e (float): the overflow
    """
    r, e = twosum(x, y)
    e += dx + dy
    r, e = kahan(r, e)
    return r, e


@NumbaChecker(["UniTuple(float64, 2)(float64, float64, float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], float64[:])",
               "UniTuple(float32, 2)(float32, float32, float32, float32)",
               "UniTuple(float32[:], 2)(float32[:], float32[:], float32[:], float32[:])"])
def htdiff(x, dx, y, dy):
    """
    Perform subtraction of numbers given in double double representation.

    Parameters
    ----------
    x : float or float[:]
        A floating point number or array.
    dx : float or float[:]
        Overflow of `x`.
    y : float or float[:]
        A floating point number or array.
    dy : float or float[:]
        Overflow of `y`.

    Returns
    -------
    r : float or float[:]
        The result of the subtraction (x - y).
    e : float or float[:]
        The error term of the subtraction.
    """
    r, e = twodiff(x, y)
    e += dx - dy
    r, e = kahan(r, e)
    return r, e


@NumbaChecker(["UniTuple(float64, 2)(float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:])"])
def twoproduct(x, y):
    """
    Calculate the product of two numbers.

    This function calculates the product of `x` and `y` with an error term: x*y = r + e. 
    The algorithm is referenced from Ogita et al. 2005.
    The magic numbers in this function restrict its domain to IEEE double precision numbers.

    Parameters
    ----------
    x : float or float[:]
        A floating point number or array.
    y : float or float[:]
        A floating point number or array.

    Returns
    -------
    r : float or float[:]
        The result of the multiplication (x * y).
    e : float or float[:]
        The error term of the multiplication.
    """
    u = x * 134217729.0  # Split input x
    v = y * 134217729.0  # Split input y
    s = u - (u - x)
    t = v - (v - y)
    f = x - s
    g = y - t
    r = x * y
    e = ((s * t - r) + s * g + f * t) + f * g
    return r, e


@NumbaChecker(["UniTuple(float64, 2)(float64, float64, float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], float64[:])"])
def htprod(x, dx, y, dy):
    """
    Performs multiplication of two numbers represented in double-double format.

    Parameters
    ----------
    x : float or float[:]
        A number or an array of numbers.
    dx : float or float[:]
        The error term of `x`.
    y : float or float[:]
        Another number or array of numbers.
    dy : float or float[:]
        The error term of `y`.

    Returns
    -------
    r : float or float[:]
        The product of `x` and `y`.
    e : float or float[:]
        The error term of the result.
    """
    r, e = twoproduct(x, y)
    e += x * dy + y * dx
    r, e = kahan(r, e)
    return r, e

@NumbaChecker(["UniTuple(float64, 2)(float64, float64, float64, float64)",
               "UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], float64[:])"])
def htdiv(x, dx, y, dy):
    """
    Performs division of two numbers represented in double-double format.

    Parameters
    ----------
    x : float or float[:]
        The numerator or an array of numerators.
    dx : float or float[:]
        The error term of `x`.
    y : float or float[:]
        The denominator or an array of denominators.
    dy : float or float[:]
        The error term of `y`.

    Returns
    -------
    r : float or float[:]
        The result of the division (x/y).
    e : float or float[:]
        The error term of the result.
    """
    r = x / y
    s, f = twoproduct(r, y)
    e = (x - s - f + dx - r * dy) / y  # Taylor expansion
    r, e = kahan(r, e)
    return r, e


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def htcplxprod(a, da, b, db):
    """
    Performs multiplication of complex double double numbers using the Gauss/Karatsuba trick.

    Parameters
    ----------
    a : complex128
        A complex number.
    da : complex128
        Overflow of `a`.
    b : complex128
        Another complex number.
    db : complex128
        Overflow of `b`.

    Returns
    -------
    result : complex128
        The result of the complex multiplication.
    overflow : complex128
        The overflow of the result.

    Notes
    -----
    This function employs the Gauss/Karatsuba trick for multiplication:

    .. math::

        (ar + I * ai)*(br + I*bi) = ar*br - ai*bi + I*[ (ar + ai)*(br + bi) - ar*br - ai*bi ]

    """
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    r, dr = htprod(rea, drea, reb, dreb)  # ar*br
    i, di = htprod(ima, dima, imb, dimb)  # ai*bi

    fac1, dfac1 = htadd(rea, drea, ima, dima)
    fac2, dfac2 = htadd(reb, dreb, imb, dimb)
    imacc, dimacc = htprod(fac1, dfac1, fac2, dfac2)
    imacc, dimacc = htdiff(imacc, dimacc, r, dr)
    imacc, dimacc = htdiff(imacc, dimacc, i, di)

    r, dr = htdiff(r, dr, i, di)
    return complex(r, imacc), complex(dr, dimacc)


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def htcplxprodconjb(a, da, b, db):
    """
    Perform multiplication of complex double double numbers where b is conjugated.

    .. math::

        (a, da) * (b, db)^* = (r, dr)

    Parameters:
        a  : float 
           a floating point number.
        da : float
           overflow of a
        b  : float
           a floating point number.
        db : float
           overflow of b

    Returns:
        r : float

        dr : float
           the overflow

    This function employs the Gauss/Karatsuba trick for multiplication:

    .. math::

        (ar + I * ai)*(br + I*bi) = ar*br - ai*bi + I*[ (ar + ai)*(br + bi) - ar*br - ai*bi ]

    """
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


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def htcplxadd(a, da, b, db):
    """
    Perform addition of complex double-double numbers.
    
    Parameters
    ----------
    a, da, b, db : complex128
        Complex double-double numbers to add.
        
    Returns
    -------
    complex128
        A tuple of the sum and its overflow.

    """
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    r, dr = htadd(rea, drea, reb, dreb)
    i, di = htadd(ima, dima, imb, dimb)
    return complex(r, i), complex(dr, di)


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def htcplxdiff(a, da, b, db):
    """
    Perform subtraction of complex double-double numbers.

    Parameters
    ----------
    a, da, b, db : complex128
        Complex double-double numbers to subtract.

    Returns
    -------
    complex128
        A tuple of the difference and its overflow.
        
    """
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    r, dr = htdiff(rea, drea, reb, dreb)
    i, di = htdiff(ima, dima, imb, dimb)
    return complex(r, i), complex(dr, di)


@NumbaChecker(["UniTuple(complex128, 2)(complex128, complex128, complex128, complex128)"])
def htcplxdiv(a, da, b, db):
    """
    Perform division of complex double-double numbers.

    Parameters
    ----------
    a, da : complex128
        The dividend complex double-double number.
    b, db : complex128
        The divisor complex double-double number.

    Returns
    -------
    complex128
        A tuple of the quotient and its overflow.
    """

    # We make the denominator real. Hence we calculate the denominator and the nominator separately.
    # First the denominator: :math:`br^2 + bi^2`.
    rea, drea = a.real, da.real
    ima, dima = a.imag, da.imag
    reb, dreb = b.real, db.real
    imb, dimb = b.imag, db.imag

    denom, ddenom = htprod(reb, dreb, reb, dreb)
    t1, dt1 = htprod(imb, dimb, imb, dimb)
    denom, ddenom = htadd(denom, ddenom, t1, dt1)

    # Now on to the numerator
    nom, dnom = htcplxprodconjb(a, da, b, db)

    r, dr = htdiv(nom.real, dnom.real, denom, ddenom)
    i, di = htdiv(nom.imag, dnom.imag, denom, ddenom)

    return complex(r, i), complex(dr, di)
