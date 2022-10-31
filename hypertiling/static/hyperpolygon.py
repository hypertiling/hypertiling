import math
import numpy as np
from hypertiling.representations import p2w
from hypertiling.arraytransformation import mfull, mrotate, morigin


# defines a hyperbolic polygon

class HyperPolygon:
    """
    Hyperbolic polygon object

    Attributes
    ----------

    p : int
        number of outer vertices (edges)

    verticesP : ndarray
        1D array of np.complex128 type, containting positions of vertices and the polygon center
        in Poincare disk coordinates

    idx : int
        auxiliary scalar index; can be used, e.g, for easy identifaction inside a tiling

    layer : int
        encodes in which layer of a tessellation this polygons is located
        
    sector : int
        index of the sector this polygons is located

    angle : float
        angle between center and the positive x-axis

    val : float
        assign a value (useful in any application)

    orientation : float
        the angle between the line defined by the center and vertices 0, and the abscissa


    Methods
    -------

    centerP()
        returns the center of the polygon in Poincare coordinates

    centerW()
        returns the center of the polygon in Weierstrass coordinates
    
    __equal__()
        checks whether two polygons are equal by comparing centers and orientations

    transform(tmat)
        apply Moebius transformation matrix "tmat" to  all points (vertices + center) of the polygon

    tf_full(ind, phi)
        transforms the entire polygon: to the origin, rotate it and back again


    ... to be completed


    """

    def __init__(self, p):

        self.p = p
        self.idx = 1
        self.layer = 1
        self.sector = 0
        self.angle = 0
        self.val = 0
        self.orientation = 0

        # Poincare disk coordinates
        self.verticesP = np.zeros(shape=self.p + 1, dtype=np.complex128)  # vertices + center


    def centerP(self):
        return self.verticesP[self.p]

    def centerW(self):
        return p2w(self.verticesP[self.p])

    # checks whether two polygons are equal
    def __eq__(self, other):
        if isinstance(other, HyperPolygon):
            centers = cmath.isclose(self.centerP, other.centerP)
            if not centers:
                return False
            orientations = cmath.isclose(self.orientation, other.orientation)
            if not orientations:
                return False
            if self.p == other.p:
                return True
        return False

    # transforms the entire polygon: to the origin, rotate it and back again
    def tf_full(self, ind, phi):
        mfull(self.p, phi, ind, self.verticesP)

    # transforms the entire polygon such that z0 is mapped to origin
    def moeb_origin(self, z0):
        morigin(self.p, z0, self.verticesP)

    def moeb_rotate(self, phi):  # rotates each point of the polygon by phi
        mrotate(self.p, phi, self.verticesP)

    def moeb_translate(self, s):
        for i in range(self.p + 1):
            z = moeb_translate_trafo(self.verticesP[i], s)
            self.verticesP[i] = z

    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        for i in range(self.p + 1):
            z = self.verticesP[i]
            z = z * rotation
            self.verticesP[i] = z

    # compute angle between center and the positive x-axis
    def find_angle(self):
        self.angle = math.degrees(math.atan2(self.centerP().imag, self.centerP().real))
        self.angle += 360 if self.angle < 0 else 0

    def find_sector(self, k, offset=0):
        """ 
        compute in which sector out of k sectors the polygon resides

        Arguments
        ---------
        k : int
            number of equal-sized sectors
        offset : float, optional
            rotate sectors by an angle
        """

        self.sector = math.floor((self.angle - offset) / (360 / k))

    # mirror on the x-axis
    def mirror(self):
        for i in range(self.p + 1):
            self.verticesP[i] = complex(self.verticesP[i].real, -self.verticesP[i].imag)
        self.find_angle()

    # returns value between -pi and pi
    def find_orientation(self):
        self.orientation = np.angle(self.verticesP[0] - self.centerP())
