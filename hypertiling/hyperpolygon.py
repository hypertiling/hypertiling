from math import floor
from .transformation import *

def morigin_py(p, z0, dz0, verticesP, verticesW):
    for i in range(p + 1):
        z = moeb_origin_trafo(z0, verticesP[i])
#        z, dz = moeb_origin_trafodd(z0, dz0, verticesP[i], verticesdP[i])
        verticesP[i] = z
#        verticesdP[i] = dz
        verticesW[:, i] = p2w(z)

def morigin_inv_py(p, z0, dz0, verticesP, verticesW):
    for i in range(p + 1):
        z = moeb_origin_trafo_inverse(z0, verticesP[i])
#        z, dz = moeb_origin_trafo_inversedd(z0, dz0, verticesP[i], verticesdP[i])
        verticesP[i] = z
#        verticesdP[i] = dz
        verticesW[:, i] = p2w(z)

def mrotate_py(p, phi, verticesP, verticesW):
    for i in range(p + 1):
#        z, dz = moeb_rotate_trafodd(verticesP[i], verticesdP[i], -phi)
        z = moeb_rotate_trafo(verticesP[i], -phi)
        verticesP[i] = z
#        verticesdP[i] = dz
        verticesW[:, i] = p2w(z)

def mfull_py(p, phi, ind, verticesP, verticesdP, verticesW):
        z0 =  verticesP[ind]
        dz0 = verticesdP[ind]
        
        for i in range(p + 1):
            z, dz = moeb_origin_trafodd(z0, dz0, verticesP[i], dz0)
            z, dz = moeb_rotate_trafodd(z, dz, -phi)
            z, dz = moeb_origin_trafo_inversedd(z0, dz0, z, dz)
            verticesP[i] = z
            verticesdP[i] = dz
            verticesW[:, i] = p2w(z)


try:
    import numba
    morigin = numba.njit(morigin_py)
    morigin_inv = numba.njit(morigin_inv_py)
    mrotate = numba.njit(mrotate_py)
    mfull = numba.njit(mfull_py)
except ImportError:
    morigin = morigin_py
    morigin_inv = morigin_inv_py
    mrotate = mrotate_py
    mfull = mfull_py

# defines a hyperbolic polygon
class HyperPolygon:
    def __init__(self, p):

# The centers are at the end of the arrays
        self.p = p  # number of edges

        # Poincare disk coordinates
#        self.centerP = complex(0, 0)  # center
#        self.dcenterP = complex(0, 0)
        self.verticesP = np.zeros(shape=self.p+1, dtype=np.complex128)  # vertices
        self.verticesdP = np.zeros(shape=self.p+1, dtype=np.complex128)

        # Weierstrass (hyperboloid) coordinates
#        self.centerW = np.array([1, 0, 0]) # center
        self.verticesW = np.zeros((3, self.p+1))  # vertices
        self.verticesW[0,-1] = 1 # center
        
        self.idx         = 1  # auxiliary scalar index; can be used, e.g, for easy identifaction inside a tessellation
        self.layer       = 1  # encodes in which layer of a tessellation this polygons is located
        self.sector      = 0  # index of the sector this polygons is located; can be used for finding neighbours more efficiently
        self.angle       = 0  # angle between self.centerP and the positive x-axis
        self.val         = 0  # assign a value (useful in any application)
        self.orientation = 0 # the angle between the line defined by the center and vertices 0, and the abscissa

        self.edges = []  # compare self.populate_edge_list

    def centerP(self):
        return self.verticesP[self.p]

    def centerW(self):
        return self.verticesW[:,-1]


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


    # transforms all points of the polygon by matrix tmat
    # only used by HyperbolicTilingDunham
    def transform(self, tmat):
        for i in range(self.p + 1):
            self.verticesW[:, i] = tmat @ self.verticesW[:, i]
            self.verticesP[i] = w2p(self.verticesW[:, i])
        self.find_angle(360)


    # transforms the entire polygon: to the origin, rotate it and back again
    def tf_full(self, ind, phi):
        mfull(self.p, phi, ind, self.verticesP, self.verticesdP, self.verticesW)

    # transforms the entire polygon such that z0 is mapped to origin
    def moeb_origin(self, z0, dz0):
        morigin(self.p, z0, dz0, self.verticesP, self.verticesW)

    def moeb_rotate(self, phi):  # rotates each point of the polygon by phi
        mrotate(self.p, phi, self.verticesP, self.verticesW)

    def moeb_translate(self, s):
        for i in range(self.p + 1):
            z = moeb_translate_trafo(self.verticesP[i], s)
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])


    def moeb_inverse(self, z0, dz0):
        morigin_inv(self.p, z0, dz0, self.verticesP, self.verticesW)

    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        for i in range(self.p + 1):
            z = self.verticesP[i]
            z = z*rotation
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])


    def find_edges(self):  # finds twice the amount of necessary edges!
        xedges = []
        yedges = []
        for i in range(self.p):
            epW1 = self.verticesW[:, i]  # edge point Weierstrass 1 and 2
            epW2 = self.verticesW[:, (i+1) % self.p]
            epP1 = w2p(epW1)  # edge point Poincare 1 and 2
            epP2 = w2p(epW2)
            xedges.append(epP1.real)
            xedges.append(epP2.real)
            yedges.append(epP1.imag)
            yedges.append(epP2.imag)

        return [xedges, yedges]


    def find_angle(self):
        self.angle = math.degrees(math.atan2(self.centerP().imag, self.centerP().real))#np.angle(self.centerP(), deg=True)
        self.angle += 360 if self.angle < 0 else 0

    def find_sector(self, k):
        self.sector = floor(self.angle/(360/k))

    def mirror(self):
        for i in range(self.p + 1):
            self.verticesP[i] = complex(self.verticesP[i].real, -self.verticesP[i].imag)
            self.verticesW = p2w(self.verticesP)
        self.find_angle(360)

    # returns value between -pi and pi
    def find_orientation(self):
        self.orientation = np.angle(self.verticesP[0]-self.centerP())

