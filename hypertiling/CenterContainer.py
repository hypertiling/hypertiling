import numpy as np
import math
import copy
import numba
import bisect

from numba.typed import List
from sortedcontainers import SortedList

from .util import fund_radius

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
    def __init__(self, p, q, phi/2):
        self.centers = SortedList([HTCenter(fund_radius(self.p, self.q), self.phi/2)])
        
    def add(z):
        self.centers.add(z)
        
    def fp_has(z):
        nangle = math.atan2(z.imag, z.real)
        centerarray_iterator = centers.irange(HTCenter(1, nangle*(1-anglefudge)), HTCenter(1, nangle*(1+anglefudge)))
        addpgon = True
        iterlen = 0 # since we cannot apply len() on the irange iterator we have to determine the length ourselves
        for c in centerarray_iterator:
            iterlen += 1
            if abs(z - c.z) < 1E-12:
                addpgon = False
                break
        if iterlen > self.p*self.q:
            anglefudge /= 2
        return addpgon

