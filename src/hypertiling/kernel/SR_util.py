import math, cmath
import numpy as np
from ..arraytransformation import mrotate, mfull, morigin
from ..representations import p2w
PI2 = 2 * np.pi


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
        
    # rotates each point of the polygon by phi
    def moeb_rotate(self, phi):  
        mrotate(self.p, phi, self.verticesP)

    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        for i in range(self.p + 1):
            z = self.verticesP[i]
            z = z * rotation
            self.verticesP[i] = z

    # compute angle between center and the positive x-axis
    def find_angle(self):
        self.angle = math.atan2(self.centerP().imag, self.centerP().real)
        self.angle += PI2 if self.angle < 0 else 0

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

        self.sector = math.floor((self.angle - offset) / (PI2 / k))

    # mirror on the x-axis
    def mirror(self):
        for i in range(self.p + 1):
            self.verticesP[i] = complex(self.verticesP[i].real, -self.verticesP[i].imag)
        self.find_angle()

    # returns value between -pi and pi
    def find_orientation(self):
        self.orientation = np.angle(self.verticesP[0] - self.centerP())




class HTCenter:
    '''This helper class wraps a complex and enables comparison based on the angle'''

    def __init__(self, *args):
        """The constructor.

            Parameters(Option 1):
                z (complex) : a complex

            Parameters(Option 2):
                r (real) : magnitude
                phi (real) : angle
        """
        if len(args) == 1:
            self.z = args[0]
            self.angle = math.atan2(self.z.imag, self.z.real)
        elif len(args) == 2:
            self.z = args[0] * complex(math.cos(args[1]), math.sin(args[1]))
            self.angle = args[1]

    def __le__(self, other):
        return self.angle <= other.angle

    def __lt__(self, other):
        return self.angle < other.angle

    def __ge__(self, other):
        return self.angle >= other.angle

    def __gt__(self, other):
        return self.angle > other.angle

    def __eq__(self, other):
        return self.z == other.z

    def __ne__(self, other):
        return self.z != other.z


try:
    from sortedcontainers import SortedList


    class DuplicateContainer:
        """
            A Container to store complex numbers and to efficiently decide
            whether a floating point representative of a given complex number is already present.
        """

        def __init__(self, linlength, r, phi):
            # Note to self, think of numpy in the alternative implementation
            self.maxlinlength = linlength  # the maximum linear length
            self.dangle = 0.1  # controls the width of the angle interval and is adapted by repeated searches

            self.centers = SortedList([HTCenter(r, phi)])

        def add(self, z):
            '''
                Add z to the container.

                Parameters:
                    z (complex): A complex number. should not be 0+0*I...
            '''
            self.centers.add(HTCenter(z))

        def __len__(self):
            '''
                Returns the length of the container and should enable use of the len() builtin on this container.
            '''
            return len(self.centers)

        def is_duplicate(self, z):
            '''
                Checks whether a representative of z has already been stored.

                Parameter:
                    z (complex): the number to check.

                Returns:
                    true if a number that is as close as 1E-12 to z has already been stored
                    else false. 1E-12 is deemed sufficient since on the hyperbolic lattice the numbers pile up near |z| ~ 1
            '''
            nangle = math.atan2(z.imag, z.real)
            centerarray_iterator = self.centers.irange(HTCenter(1, nangle * (1 - self.dangle)),
                                                       HTCenter(1, nangle * (1 + self.dangle)))
            incontainer = False
            iterlen = 0  # since we cannot apply len() on the irange iterator we have to determine the length ourselves
            for c in centerarray_iterator:
                iterlen += 1
                if abs(z - c.z) < 1E-12:  # 1E-12 is the relative acuuracy here, since for the hyperbolic lattice vertices pile up near |z|~1
                    incontainer = True
                    break
            if iterlen > self.maxlinlength:
                self.dangle /= 2.0
            return incontainer


except ImportError:
    import bisect


    class DuplicateContainer:
        '''
            A Container to store complex numbers and to efficiently decide
            whether a floating point representative of a given complex number is already present.
        '''

        def __init__(self, linlength, r, phi):
            # Note to self, think of numpy in the alternative implementation
            self.maxlinlength = linlength  # the maximum linear length
            self.dangle = 0.1  # controls the width of the angle interval and is adapted by repeated searches
            self.centers = [HTCenter(r, phi)]

        def add(self, z):
            '''
                Add z to the container
                
                Parameters:
                    z (complex): A complex number. It should not be 0+0*I...
            '''
            temp = HTCenter(z)
            pos = bisect.bisect_left(self.centers, temp)
            self.centers.insert(pos, temp)

        def __len__(self):
            '''
                Returns the length of the container and should enable use of the len() builtin on this container.
            '''
            return len(self.centers)

        def is_duplicate(self, z):
            '''
                Checks whether a representative of z has already been stored
                
                Parameter:
                    z (complex): the number to check.
                    
                Returns:
                    true if a number that is as close as 1E-12 to z has already been stored
                    else false.
            '''
            nangle = math.atan2(z.imag, z.real)
            lpos = bisect.bisect_left(self.centers, HTCenter(1, nangle * (1 - self.dangle)))
            upos = bisect.bisect_left(self.centers, HTCenter(1, nangle * (1 + self.dangle)))
            if (upos - lpos) > self.maxlinlength:
                self.dangle /= 2.0
            return any(abs(c.z - z) < 1E-12 for c in self.centers[lpos:upos])