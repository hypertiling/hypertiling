import numpy as np
from ..representations import p2w_xyt, w2p_xyt
from .SR_base import HyperPolygon

def transformW_poly(polygon: HyperPolygon, transformation):
    """
    Apply Weierstraß transformation matrix to entire HyperPolygon, i.e. vertices and center coordiantes
    """
    new_verts = np.zeros_like(polygon.verticesP)
    for i, pointP in enumerate(polygon.verticesP):
        new_verts[i] = transformW_site(pointP, transformation)
    polygon.verticesP = new_verts


def transformW_site(pointP: np.complex128, transformation):
    """
    Apply Weierstraß transformation to Poincare site
    1. Transform site from Poincare to Weierstraß
    2. Apply Weierstraß transformation
    3. Transform back
    """
    return w2p_xyt(transformation @ p2w_xyt(pointP))
