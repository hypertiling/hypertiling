import math, cmath
import numpy as np
from ..arraytransformation import mrotate, mfull, morigin
from ..representations import p2w
PI2 = 2 * np.pi


class HyperPolygon:
    """
    Represents a hyperbolic polygon with operations suitable for tiling hyperbolic planes.

    Attributes
    ----------
    p : int
        The number of edges of the polygon.
    idx : int
        The unique identifier of the polygon.
    layer : int
        The layer number in the tiling where the polygon belongs.
    sector : int
        The sector of the tiling where the polygon is located.
    angle : float
        The internal angle of the polygon.
    val : float
        The value associated with the polygon (custom usage).
    orientation : float
        The orientation of the polygon in the tiling.
    _vertices : np.ndarray[np.complex128]
        The center and vertices in Poincare disk coordinates.
    """


    
    def __init__(self, p, vertices=None, idx=None, layer=None, sector=None, angle=None, val=None, orientation=None):
        """
        Initialize the HyperPolygon with given parameters.
        """

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



    def get_center(self):
        """
        Returns the center of the polygon in Poincare coordinates.

        Returns
        -------
        np.complex128
            The center of the polygon.
        """
        return self._vertices[self.p]
    

    def get_vertices(self):
        """
        Returns an array containing the outer vertices in Poincare coordinates.

        Returns
        -------
        np.array[np.complex128]
            An array of the outer vertices of the polygon.
        """
        return self._vertices[:-1]
    

    def get_polygon(self):
        """
        Returns an array containing center + outer vertices in Poincare coordinates.

        Returns
        -------
        np.array[np.complex128]
            An array of the center and outer vertices of the polygon.
        """
        return self._vertices
    

    def set_center(self, center):
        """
        Sets the center of the polygon in Poincare coordinates.

        Parameters
        ----------
        center : np.complex128
            The center of the polygon in Poincare coordinates.
        """
        self._vertices[self.p] = center


    def set_vertices(self, vertices):
        """
        Sets the outer vertices of the polygon in Poincare coordinates.

        Parameters
        ----------
        vertices : np.array[np.complex128]
            An array of the outer vertices of the polygon.

        Raises
        ------
        ValueError
            If the number of vertices provided is not equal to the number of polygon edges.
        """
        if len(vertices) != self.p:
            raise ValueError(f"[hypertiling] Error: Expected {self.p} vertices, got {len(vertices)}")
        self._vertices[:-1] = vertices


    def set_polygon(self, polygon):
        """
        Sets the entire polygon: center + outer vertices in Poincare coordinates.

        Parameters
        ----------
        polygon : np.array[np.complex128]
            An array of the center and outer vertices of the polygon.

        Raises
        ------
        ValueError
            If the number of points provided is not equal to the number of polygon edges plus one (center point).
        """
        if len(polygon) != self.p + 1:
            raise ValueError(f"[hypertiling] Error: Expected {self.p + 1} points, got {len(polygon)}")
        self._vertices = polygon


    def centerW(self):
        """
        Returns the center of the polygon in Weierstrass coordinates.

        Returns
        -------
        Weierstrass coordinate
            The Weierstrass coordinate of the polygon center.
        """
        return p2w(self._vertices[self.p])


    def __eq__(self, other):
        """
        Checks whether two polygons are equal.

        Parameters
        ----------
        other : HyperPolygon
            The other HyperPolygon to compare with.

        Returns
        -------
        bool
            True if the two polygons are equal, False otherwise.
        """
        if isinstance(other, HyperPolygon):

            if self.p != other.p:
                return False
            
            centers_close = cmath.isclose(self.get_center, other.get_center)
            orientations_close = cmath.isclose(self.orientation, other.orientation)
            if centers_close and orientations_close:
                return True
            else:
                return False


    def tf_full(self, ind, phi):
        """
        Transforms the entire polygon: to the origin, rotate it and back again.

        Parameters
        ----------
        ind : int
            Index of the vertex that defines the Moebius Transform.
        phi : float
            Angle of rotation.
        """
        mfull(self.p, phi, ind, self._vertices)


    def moeb_origin(self, z0):
        """
        Transforms the entire polygon such that z0 is mapped to origin.
        
        Parameters
        ----------
        z0 : complex
            The point that is to be mapped to the origin.
        """
        morigin(self.p, z0, self._vertices)
        

    def moeb_rotate(self, phi):  
        """
        Rotates each point of the polygon by phi.
        
        Parameters
        ----------
        phi : float
            The angle of rotation in radians.
        """
        mrotate(self.p, phi, self._vertices)


    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        self._vertices = [z * rotation for z in self._vertices]


    def find_angle(self):
        """
        Compute angle between center and the positive x-axis.
        The angle is adjusted to ensure it's non-negative. The result is stored as an instance variable.
        """
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


    def mirror(self):
        """
        Mirror on the x-axis.
        
        This function mirrors the polygon on the x-axis and updates the angle attribute.
        """
        for i in range(self.p + 1):
            self._vertices[i] = complex(self._vertices[i].real, -self._vertices[i].imag)
        self.find_angle()


    def find_orientation(self):
        """
        Finds the orientation of the polygon.
        
        This function computes the orientation of the polygon (the angle between the first vertex and the center of the polygon) 
        and stores the result in the orientation attribute. The returned value is between -pi and pi.
        """
        self.orientation = np.angle(self._vertices[0] - self.get_center())

