import numpy as np
import hypertiling.transformation as trans
from hypertiling.check_numba import NumbaChecker


@NumbaChecker("(int64, complex128, complex128[:])")
def morigin(p, z0, verticesP):
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


@NumbaChecker("(int64, complex128, complex128[:])")
def moeb_origin_vector(p, z0, points):
    """
    Apply Moebius translation to an array of length p
    
    Arguments:
    -----------
    p : int
        Length of point list
    z0 : complex128
        Vertex that we transform around.
    points : complex128[]
        List of points in the Poincare disk
    """

    for i in range(p):
        z = trans.moeb_origin_trafo(z0, points[i])
        points[i] = z


@NumbaChecker("(int64, float64, complex128[:])")
def mrotate(p, phi, verticesP):
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


@NumbaChecker("complex128(complex128, float64, complex128)")
def mfull_point(z0, phi, p):
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
    return trans.moeb_origin_trafo(-z0, z)



@NumbaChecker("complex128[::1](uint16, float64, complex128, complex128)")
def multi_rotation_around_vertex(qn, dqhi, z0, p):
    """
    Perform qn discrete rotations by i*dqhi of p around z0
    """

    # transform to origin
    z = trans.moeb_origin_trafo(z0, p)

    # apply successive rotations
    retvals = np.zeros(qn, dtype=np.complex128)

    for k in range(qn):
        zn = trans.moeb_rotate_trafo(-(k*dqhi), z)
        retvals[k] = zn
    
    # transform all points back
    moeb_origin_vector(qn, -z0, retvals)

    return retvals



@NumbaChecker("(int64, float64, int64, complex128[:])")
def mfull(p, phi, ind, verticesP):
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
