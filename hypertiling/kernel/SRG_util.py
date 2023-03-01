import math
import numpy as np
import bisect

from ..ion import htprint

PI2 = 6.2831853071795864769

class HTCenterTuple:
    '''
    This helper class wraps a complex and enables comparison based on the angle
    It provides an extension of the class "HTCenter" hosting tuples of coordinate
    and index instead of only a coordinate
    '''

    def __init__(self, *args):
        """The constructor.

            Parameters (Option 1):
                z (complex) : a complex
                idx (integer) : the index of z in the tiling

            Parameters (Option 2):
                r (real) : magnitude
                phi (real) : angle
                idx (integer) : the index of z in the tiling
        """
        if len(args) == 2:
            self.z = args[0]
            self.angle = math.atan2(self.z.imag, self.z.real)
            self.angle += PI2 if self.angle<0 else 0 
            self.idx = args[1]
        elif len(args) == 3:
            self.z = args[0] * complex(math.cos(args[1]), math.sin(args[1]))
            self.angle = args[1]
            self.angle += PI2 if self.angle<0 else 0 
            self.idx = args[2]

    def __le__(self, other):
        return self.angle <= other.angle

    def __lt__(self, other):
        return self.angle < other.angle

    def __ge__(self, other):
        return self.angle >= other.angle

    def __gt__(self, other):
        return self.angle > other.angle

    def __eq__(self, other):
        return self.idx == other.idx

    def __ne__(self, other):
        return self.idx != other.idx



class DuplicateContainerCircularCommon:
    """
    Common methods used in both versions of the circular DuplicateContainer (the default and the fallback version)
    """
    def __init__(self):
        pass        

    def __len__(self):
        '''
        Returns the length of the container and should enable use of the len() builtin on this container.
        '''
        return len(self.centers)


    def _angle_windows(self, z):
        """
        computes the bounds of a region of angles around the angle of the complex number z
        in case the region crosses 2*pi, it is split into two
        """

        windows = []

        nangle = math.atan2(z.imag, z.real)
        nangle += PI2 if nangle<0 else 0

        angle_upper = nangle + self.dangle
        angle_lower = nangle - self.dangle

        # in case the angle domain crosses 0, we need to split into two checks
        second_window = False
        if angle_lower < 0:
            angle_lower = 0
            second_window = True
            angle_lower_extra = angle_lower + PI2
            angle_upper_extra = PI2

        elif angle_upper > PI2:
            angle_upper = PI2
            second_window = True
            angle_lower_extra = 0
            angle_upper_extra = angle_upper - PI2

        windows.append([angle_lower, angle_upper])

        if second_window:
            windows.append([angle_lower_extra,angle_upper_extra])

        return windows


try:
    from sortedcontainers import SortedList

    htprint("Status", "Found package 'sortedcontainers'!")

    # default
    class DuplicateContainerCircular(DuplicateContainerCircularCommon):
        """
        A Container to store complex numbers and to efficiently decide
        whether a floating point representative of a given complex number is already present.
        """

        def __init__(self, linlength):#,: r, phi, idx):
            # the maximum linear length
            self.maxlinlength = linlength  
            # controls the width of the angle interval and is adapted by repeated searches
            self.dangle = 0.1  
            # 1E-12 is the relative acuuracy here, since for the hyperbolic lattice vertices pile up near |z|~1
            self.eps = 1e-12
            # array where the actual data is stored
            self.centers = SortedList()#[HTCenterTuple(r, phi, idx)])


        def add(self, z, idx):
            '''
            Add z to the container.

            Parameters:
                z (complex): A complex number. should not be 0+0*I...
            '''
            self.centers.add(HTCenterTuple(z,idx))


        def remove_by_idx(self, z, idx):
            pos = self.centers.bisect_left(HTCenterTuple(z,idx))

            if self.centers[pos-1].idx == idx:
                self.centers.pop(pos-1)
            elif self.centers[pos].idx == idx:
                self.centers.pop(pos)
            elif self.centers[pos+1].idx == idx:
                self.centers.pop(pos+1)
            else:
                htprint("Warning", "Something unexpected happend!")   



        def is_duplicate(self, z):
            '''
            Checks whether a representative of z has already been stored.

            Parameter:
                z (complex): the number to check.

            Returns:
                true if a number (representing a cell center) that is as close as eps to z has 
                already been stored; also returns the index of that number
                else false
            '''
                        
            # compute regions to be checked
            windows = self._angle_windows(z)

            # loop over regions
            for w in windows:

                iterator = self.centers.irange(HTCenterTuple(1, w[0], -1), HTCenterTuple(1, w[1], -1))
                # since we cannot apply len() on the irange iterator we have to determine the length ourselves
                iterlen = 0  
                # loop over region
                for c in iterator:
                    iterlen += 1
                    if np.abs(z - c.z) < self.eps:
                        return True, c.idx
                if iterlen > self.maxlinlength:
                    self.dangle /= 2.0
                
            return False, -1


except ImportError:
    import bisect

    htprint("Status", "Package 'sortedcontainers' not found, using fallback implementation!")


    # fallback
    class DuplicateContainerCircular(DuplicateContainerCircularCommon):
        '''
            A Container to store complex numbers and to efficiently decide
            whether a floating point representative of a given complex number is already present.
            Fallback implementation in case SortedListed is not available
        '''

        def __init__(self, linlength):
            # the maximum linear length
            self.maxlinlength = linlength  
            # controls the width of the angle interval and is adapted by repeated searches
            self.dangle = 0.1  
            # 1E-12 is the relative acuuracy here, since for the hyperbolic lattice vertices pile up near |z|~1
            self.eps = 1e-12
            # list where the actual data is stored
            self.centers = []


        def add(self, z, idx):
            '''
                Add z to the container
                
                Parameters:
                    z (complex): A complex number. It should not be 0+0*I...
            '''
            temp = HTCenterTuple(z, idx)
            pos = bisect.bisect_left(self.centers, temp)
            self.centers.insert(pos, temp)


        def remove_by_idx(self, z, idx):

            temp = HTCenterTuple(z, idx) # create temp object 
            pos = bisect.bisect_left(self.centers, temp)

            if self.centers[pos-1].idx == idx:
                self.centers.pop(pos-1)
            elif self.centers[pos].idx == idx:
                self.centers.pop(pos)
            elif self.centers[pos+1].idx == idx:
                self.centers.pop(pos+1)
            else:
                htprint("Warning", "Something unexpected happend!")   
        


        def is_duplicate(self, z):
            '''
                Checks whether a representative of z has already been stored
                
                Parameter:
                    z (complex): the number to check.
                    
                Returns:
                    true if a number (cell) that is as close as eps to z has already been stored
                    as well as the index of that number (cell)
                    else false
                    
            '''
            # compute regions to be checked
            windows = self._angle_windows(z)
            
            # loop over region
            for w in windows:
                lpos = bisect.bisect_left(self.centers, HTCenterTuple(1, w[0], -1))
                upos = bisect.bisect_left(self.centers, HTCenterTuple(1, w[1], -1))
                if (upos - lpos) > self.maxlinlength:
                    self.dangle /= 2.0

                for c in self.centers[lpos:upos]:
                    if np.abs(z - c.z) < self.eps:
                        return True, c.idx
            return False, -1
