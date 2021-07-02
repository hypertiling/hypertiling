from math import floor
from .transformation import *


class HyperPolygon:
    def __init__(self, p, q):
        self.p = p  # no. of edges
        self.q = q  # no. of adjacent polygons per vertex

        self.centerP = complex(0, 0)  # poincare coordinates of the center
        self.centerW = np.array([0, 0, 1])

        self.verticesP = np.zeros(shape=self.p, dtype=np.complex128)  # POINCARE Disk 2D
        self.verticesW = np.zeros((3, self.p))  # WEIERSTRASS 3D

        self.number = 1  # for counting them and coloring them successively
        self.layer = 1
        self.angle = 0  # angle between self.centerP and the positive x-axis
        self.sector = 0  # sector for finding NN more efficiently
        self.val = 0  # field value for Ising/PDEQ solver

    def __eq__(self, other):  # checks if two polygons are equivalent; currently not being used
        re1 = self.centerP.real
        im1 = self.centerP.imag
        re2 = other.centerP.real
        im2 = other.centerP.imag
        c1 = complex(round(re1, 5), round(im1, 5))
        c2 = complex(round(re2, 5), round(im2, 5))
        print("__ eq__ was used")  # in case it happens by accident
        return c1 == c2

    # transforms all points of the polygon by matrix tmat; only used by dunhams tiling class
    def transform(self, tmat):
        self.centerW = tmat @ self.centerW
        self.centerP = w2p(self.centerW)
        self.find_angle(360)
        for i in range(self.p):
            self.verticesW[:, i] = tmat @ self.verticesW[:, i]
            self.verticesP[i] = w2p(self.verticesW[:, i])

    def moeb_origin(self, z0):  # transforms the whole polygon such that z0 is mapped to origin
        self.centerP = moeb_origin_trafo(z0, self.centerP)
        self.centerW = p2w(self.centerP)
        # self.find_angle(360)  # this might be superfluous
        for i in range(self.p):
            z = moeb_origin_trafo(z0, self.verticesP[i])
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])

    def moeb_rotate(self, phi):  # rotates each point of the polygon by phi
        self.centerP = moeb_rotate_trafo(self.centerP, -phi)  # these two lines might be redundant
        self.centerW = p2w(self.centerP)
        # self.find_angle(360)   # i think this is not needed anymore
        for i in range(self.p):
            z = moeb_rotate_trafo(self.verticesP[i], -phi)
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])

    def moeb_translate(self, s):
        self.centerP = moeb_translate_trafo(self.centerP, s)
        self.centerW = p2w(self.centerP)
        # self.find_angle(360)  # i think this is not needed anymore
        for i in range(self.p):
            z = moeb_translate_trafo(self.verticesP[i], s)
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])

    def moeb_inverse(self, z0, phi=0, s=0):
        self.centerP = moeb_inverse_trafo(self.centerP, z0, -phi, s)
        self.centerW = p2w(self.centerP)
        # self.find_angle(360)  # superfluous
        for i in range(self.p):
            z = moeb_inverse_trafo(self.verticesP[i], z0, -phi, s)
            self.verticesP[i] = z
            self.verticesW[:, i] = p2w(self.verticesP[i])

    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        z = self.centerP
        z = z*rotation
        self.centerP = z
        self.centerW = p2w(self.centerP)
        for i in range(self.p):
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

    def find_angle(self, k, offset=0):   # has to be called after the inverse trafo...
        self.angle = np.arctan2(self.centerP.imag, self.centerP.real)*180/np.pi-offset  # requires y first
        self.angle = np.round(self.angle, 10)
        self.angle += 360 if self.angle < 0 else 0
        self.sector = floor(self.angle/(360/(k*self.p)))  # k*p sectors; insert offset +0.1 here?!

    def mirror(self):
        self.centerP = complex(self.centerP.real, - self.centerP.imag)  # mirror on axis Im(z)=0
        self.centerW = p2w(self.centerP)
        self.find_angle(360)
        for i in range(self.p):
            self.verticesP[i] = complex(self.verticesP[i].real, (-1)*self.verticesP[i].imag)
            self.verticesW = p2w(self.verticesP)
