import numpy as np
import math
import numba
import bisect

from sortedcontainers import SortedList

from .util import fund_radius

#@numba.njit
#def add_center_if_new(centerset, lpos, upos, centerangles, z, angle):
    #addpgon = True
    #idx = 0
    #for cen in centerset[lpos:upos]:
        #if abs(cen - z) < 1E-12:
            #addpgon = False
            #break
    #return addpgon

#@numba.njit
#def add_center_if_new_sc(centerset, z):
    #addpgon = True
    #for cen in centerset:
        #if abs(cen - z) < 1E-12:
            #addpgon = False
            #break
    #return addpgon

class HTCenter:
    def __init__(self, *args):        
        if len(args) == 1:
            self.z = args[0]
            self.angle = math.atan2(self.z.imag, self.z.real)
        elif len(args) == 2:
            self.z = args[0]*complex(math.cos(args[1]), math.sin(args[1]))
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

class CenterContainer:
    def __init__(self, p, q, phi):
        # Note to self, think of numpy in the alternative implementation
        self.p = p
        self.q = q
        self.dangle = 0.1 # controls the width of the angle interval and is adapted by repeated searches
        self.centers = SortedList([HTCenter(fund_radius(self.p, self.q), phi/2)]) # We arbitrarily set the initial fundamental Polygon to have an angle of phi/2
        
    def add(self, z):
        self.centers.add(HTCenter(z))
        
    def fp_has(self, z):
        nangle = math.atan2(z.imag, z.real)
        centerarray_iterator = self.centers.irange(HTCenter(1, nangle*(1-self.dangle)), HTCenter(1, nangle*(1+self.dangle)))
        addpgon = True
        iterlen = 0 # since we cannot apply len() on the irange iterator we have to determine the length ourselves
        for c in centerarray_iterator:
            iterlen += 1
            if abs(z - c.z) < 1E-12:
                addpgon = False
                break
        if iterlen > self.p*self.q:
            self.dangle /= 2
        return addpgon

