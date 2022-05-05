from math import floor
from .transformation import *

@numba.njit
def morigin(p, z0, dz0, verticesP, verticesdP, verticesW):
    for i in range(p + 1):
        z, dz = moeb_origin_trafodd(z0, dz0, verticesP[i], verticesdP[i])
        verticesP[i] = z
        verticesdP[i] = dz
        verticesW[:, i] = p2w(z)

@numba.njit
def morigin_inv(p, z0, dz0, verticesP, verticesdP, verticesW):
    for i in range(p + 1):
        z, dz = moeb_origin_trafo_inversedd(z0, dz0, verticesP[i], verticesdP[i])
        verticesP[i] = z
        verticesdP[i] = dz
        verticesW[:, i] = p2w(z)

@numba.njit
def mrotate(p, phi, verticesP, verticesdP, verticesW):
    for i in range(p + 1):
        z, dz = moeb_rotate_trafodd(verticesP[i], verticesdP[i], -phi)
        verticesP[i] = z
        verticesdP[i] = dz
        verticesW[:, i] = p2w(z)

# defines a hyperbolic polygon
class HyperPolygon:
    def __init__(self, p, q):
        self.p = p  # number of edges
        self.q = q  # number of adjacent polygons per vertex

# The centers are at the end of the arrays

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
    
    # checks whether two polygons are equal (within given numerical precision)
    # untested!
    def __eq__(self, other, digits=7):  
        re1 = self.centerP().real
        im1 = self.centerP().imag
        re2 = other.centerP().real
        im2 = other.centerP().imag
        c1 = complex(round(re1, digits), round(im1, digits))
        c2 = complex(round(re2, digits), round(im2, digits))
        print("Warning: Equality operator of the HyperPolygon class is untested!")  
        return c1 == c2


    # transforms all points of the polygon by matrix tmat
    # only used by HyperbolicTilingDunham
    def transform(self, tmat):
        for i in range(self.p + 1):
            self.verticesW[:, i] = tmat @ self.verticesW[:, i]
            self.verticesP[i] = w2p(self.verticesW[:, i])
        self.find_angle(360)



    # transforms the entire polygon such that z0 is mapped to origin
    def moeb_origin(self, z0, dz0):
        morigin(self.p, z0, dz0, self.verticesP, self.verticesdP, self.verticesW)
#        for i in range(self.p + 1):
#            z, dz = moeb_origin_trafodd(z0, dz0, self.verticesP[i], self.verticesdP[i])
#            self.verticesP[i] = z
#            self.verticesdP[i] = dz
#            self.verticesW[:, i] = p2w(self.verticesP[i])
        # self.find_angle(360)  # this might be superfluous


    def moeb_rotate(self, phi):  # rotates each point of the polygon by phi
        mrotate(self.p, phi, self.verticesP, self.verticesdP, self.verticesW)
#        for i in range(self.p + 1):
#            z, dz = moeb_rotate_trafodd(self.verticesP[i], self.verticesdP[i], -phi)
#            self.verticesP[i] = z
#            self.verticesdP[i] = dz
#            self.verticesW[:, i] = p2w(self.verticesP[i])


    def moeb_translate(self, s):
        for i in range(self.p + 1):
            z = moeb_translate_trafo(self.verticesP[i], s)
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])


    def moeb_inverse(self, z0, dz0):
        morigin_inv(self.p, z0, dz0, self.verticesP, self.verticesdP, self.verticesW)
#        for i in range(self.p + 1):
#            z, dz = moeb_origin_trafo_inversedd(z0, dz0, self.verticesP[i], self.verticesdP[i])
#            self.verticesP[i] = z
#            self.verticesdP[i] = dz
#            self.verticesW[:, i] = p2w(self.verticesP[i])


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

    def find_sector(self):
        self.sector = floor(self.angle/(360/self.p))


    def mirror(self):
        for i in range(self.p + 1):
            self.verticesP[i] = complex(self.verticesP[i].real, -self.verticesP[i].imag)
            self.verticesW = p2w(self.verticesP)
        self.find_angle(360)

    # returns value between -pi and pi
    def find_orientation(self):
        self.orientation = np.angle(self.verticesP[0]-self.centerP())

