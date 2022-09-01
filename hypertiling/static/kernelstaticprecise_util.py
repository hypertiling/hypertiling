import math


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


    class CenterContainer:
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

        def fp_has(self, z):
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


    class CenterContainer:
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

        def fp_has(self, z):
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
