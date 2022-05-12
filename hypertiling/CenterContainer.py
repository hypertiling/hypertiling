import math

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

try:
    from sortedcontainers import SortedList
    class CenterContainer:
        def __init__(self, p, q, phi):
            # Note to self, think of numpy in the alternative implementation
            self.maxlinlength = p*q# the maximum linear length
            self.dangle = 0.1 # controls the width of the angle interval and is adapted by repeated searches
            self.centers = SortedList([HTCenter(fund_radius(p, q), phi/2)]) # We arbitrarily set the initial fundamental Polygon to have an angle of phi/2
        
        def add(self, z):
            self.centers.add(HTCenter(z))
        
        def fp_has(self, z):
            nangle = math.atan2(z.imag, z.real)
            centerarray_iterator = self.centers.irange(HTCenter(1, nangle*(1-self.dangle)), HTCenter(1, nangle*(1+self.dangle)))
            addpgon = False
            iterlen = 0 # since we cannot apply len() on the irange iterator we have to determine the length ourselves
            for c in centerarray_iterator:
                iterlen += 1
                if abs(z - c.z) < 1E-12:
                    addpgon = True
                    break
            if iterlen > seolf.maxlinlength:
                self.dangle /= 2.0
            return addpgon
except ImportError:
    import bisect

    class CenterContainer:
        def __init__(self, p, q, phi):
            # Note to self, think of numpy in this alternative implementation
            self.maxlinlength = p*q# the maximum linear length
            self.dangle = 0.1 # controls the width of the angle interval and is adapted by repeated searches
            self.centers = List(HTCenter(fund_radius(p, q), phi/2)) # We arbitrarily set the initial fundamental Polygon to have an angle of phi/2
        
        def add(self, z):
            temp = HTCenter(z)
            pos = bisect.bisect_left(self.centers, temp)
            self.centers.insert(temp)
        
        def fp_has(self, z):
            nangle = math.atan2(z.imag, z.real)
            lpos = bisect.bisect_left(self.centers, HTCenter(1, nangle*(1-self.dangle)))
            upos = bisect.bisect_left(self.centers, HTCenter(1, nangle*(1+self.dangle)))
            if (upos - lpos) > self.maxlinlength:
                self.dangle /= 2.0
            return any(abs(c.z - z) < 1E-12 for c in centers[lpos:upos])
