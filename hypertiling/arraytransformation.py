import hypertiling.transformation as trans


def morigin_py(p, z0, verticesP):
    """
    Apply Moebius transform to an array of length (p+1) of vertices.
    
    Arguments:
    -----------
    p : int
        Number of outer vertices.
    z0 : complex128
        Vertex that we transform around.
    verticesP : Hyperpolygon
        Array of vertices + the center that make up the polygon.
    """

    for i in range(p + 1):
        z = trans.moeb_origin_trafo(z0, verticesP[i])
        verticesP[i] = z


def morigin_inv_py(p, z0, verticesP):
    """
    Apply inverse Moebius trafo to an array of length (p+1) of vertices.
    
    Arguments:
    -----------
    p : int
        Number of outer vertices.
    z0 : complex128
        Vertex that we transform around.
    verticesP : complex[]
        Array of vertices + the center that make up the polygon.
    """

    for i in range(p + 1):
        z = trans.moeb_origin_trafo_inverse(z0, verticesP[i])
        verticesP[i] = z


def mrotate_py(p, phi, verticesP):
    """
    Rotate an array of length (p + 1) of complex vertices.
    
    Arguments:
    -----------
    p : int
        Number of outer vertices.
    phi : float
        Angle of rotation
    verticesP : complex[]
        Array of vertices + the center that make up the polygon.
    """
    for i in range(p + 1):
        # FIXME: I do not like the - in front of phi as it makes the behaviour more hidden
        z = trans.moeb_rotate_trafo(-phi, verticesP[i])
        verticesP[i] = z


def mfull_point_py(z0, phi, p):
    """
    Apply all transformations(origin, rotate, inv_origin) to a single vertex.
    
    Arguments:
    -----------
    z0 : complex128
        Vertex that we transform around.
    phi : float
        Angle of rotation.
    p : complex128
        The vertex that we want to fully transform.
    """

    z = trans.moeb_origin_trafo(z0, p)
    z = trans.moeb_rotate_trafo(-phi, z)
    return trans.moeb_origin_trafo_inverse(z0, z)


def mfull_py(p, phi, ind, verticesP):
    """ 
    Apply all transformations(origin, rotate, inv_origin) in dd precision to the vertices of an entire polygon.

    Arguments:
    -----------
    p : int
        Number of outer vertices.
    phi : float
        Angle of roatation
    ind : int
        Index of vertex that defines the Moebius Transform
    verticesP : Hyperpolygon
        Array of vertices + the center that make up the polygon.
    """
    z0 = verticesP[ind]
    dz0 = complex(0, 0)

    for i in range(p + 1):
        z, dz = trans.moeb_origin_trafodd(z0, dz0, verticesP[i], dz0)
        z, dz = trans.moeb_rotate_trafodd(z, dz, -phi)
        z, dz = trans.moeb_origin_trafo_inversedd(z0, dz0, z, dz)
        verticesP[i] = z
        # verticesdP[i] = dz


# try to use numba
try:
    import numba

    morigin = numba.njit(morigin_py)
    morigin_inv = numba.njit(morigin_inv_py)
    mrotate = numba.njit(mrotate_py)
    mfull_point = numba.njit(mfull_point_py)
    mfull = numba.njit(mfull_py)
except ImportError:
    morigin = morigin_py
    morigin_inv = morigin_inv_py
    mrotate = mrotate_py
    mfull_point = mfull_point_py
    mfull = mfull_py
