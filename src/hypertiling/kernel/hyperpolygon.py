import math, cmath
import numpy as np
from ..arraytransformation import mrotate, mfull, morigin
from ..representations import p2w
PI2 = 2 * np.pi


class HyperPolygon:
    """
    Represents a hyperbolic polygon with operations suitable for tiling hyperbolic planes.
    
    Attributes:
        p (int): The number of edges of the polygon.
        idx (int): The unique identifier of the polygon.
        layer (int): The layer number in the tiling where the polygon belongs.
        sector (int): The sector of the tiling where the polygon is located.
        angle (float): The internal angle of the polygon.
        val (float): The value associated with the polygon (custom usage).
        orientation (float): The orientation of the polygon in the tiling.
        _vertices (np.array[np.complex128]): The center and vertices in Poincare disk coordinates.
    """

    
    def __init__(self, p, vertices=None, idx=None, layer=None, sector=None, angle=None, val=None, orientation=None):
        
        self.p = p
        self.idx = idx
        self.layer = layer
        self.sector = sector
        self.angle = angle
        self.val = val
        self.orientation = orientation
        self.edges = None

        if vertices is not None:
            if len(vertices) != self.p + 1 or not isinstance(vertices, np.ndarray):
                raise ValueError("[hypertiling] Error: Argument 'vertices' must be a numpy array of length p + 1 (center + vertices)!")
            self._vertices = vertices
        else:
            self._vertices = np.zeros(shape=self.p + 1, dtype=np.complex128) # vertices + center



    # returns the center of the polygon in Poincare coordinates
    def get_center(self):
        return self._vertices[self.p]
    

    # returns an array containing the outer vertices in Poincare coordinates
    def get_vertices(self):
        return self._vertices[:-1]
    

    # returns an array containing center + outer vertices in Poincare coordinates
    def get_polygon(self):
        return self._vertices
    

    # sets the center of the polygon in Poincare coordinates
    def set_center(self, center):
        self._vertices[self.p] = center


    # sets the outer vertices of the polygon in Poincare coordinates
    def set_vertices(self, vertices):
        if len(vertices) != self.p:
            raise ValueError(f"[hypertiling] Error: Expected {self.p} vertices, got {len(vertices)}")
        self._vertices[:-1] = vertices


    # sets the entire polygon: center + outer vertices in Poincare coordinates
    def set_polygon(self, polygon):
        if len(polygon) != self.p + 1:
            raise ValueError(f"[hypertiling] Error: Expected {self.p + 1} points, got {len(polygon)}")
        self._vertices = polygon


    # returns the center of the polygon in Weierstrass coordinates
    def centerW(self):
        return p2w(self._vertices[self.p])


    # checks whether two polygons are equal
    def __eq__(self, other):
        if isinstance(other, HyperPolygon):

            if self.p != other.p:
                return False
            
            centers_close = cmath.isclose(self.get_center, other.get_center)
            orientations_close = cmath.isclose(self.orientation, other.orientation)
            if centers_close and orientations_close:
                return True
            else:
                return False


    # transforms the entire polygon: to the origin, rotate it and back again
    def tf_full(self, ind, phi):
        mfull(self.p, phi, ind, self._vertices)

    # transforms the entire polygon such that z0 is mapped to origin
    def moeb_origin(self, z0):
        morigin(self.p, z0, self._vertices)
        
    # rotates each point of the polygon by phi
    def moeb_rotate(self, phi):  
        mrotate(self.p, phi, self._vertices)

    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        self._vertices = [z * rotation for z in self._vertices]

    # compute angle between center and the positive x-axis
    def find_angle(self):
        self.angle = math.atan2(self.get_center().imag, self.get_center().real)
        self.angle += PI2 if self.angle < 0 else 0

    def find_sector(self, k, offset=0):
        """ 
        Compute - based on complex angle -in which sector out of k sectors the polygon is located

        Arguments
        ---------
        k : int
            number of equal-sized sectors
        offset : float, optional
            rotate sectors by an angle
        """

        self.sector = math.floor((self.angle - offset) / (PI2 / k))

    # mirror on the x-axis
    def mirror(self):
        for i in range(self.p + 1):
            self._vertices[i] = complex(self._vertices[i].real, -self._vertices[i].imag)
        self.find_angle()

    # returns value between -pi and pi
    def find_orientation(self):
        self.orientation = np.angle(self._vertices[0] - self.get_center())