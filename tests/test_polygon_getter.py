import unittest
from hypertiling import HyperbolicTiling
from hypertiling.kernel.hyperpolygon import HyperPolygon

class TestPolygonGetter(unittest.TestCase):


    def test_polygon_getter(self):
        # test whether polygon getter is implemented
        for kernel in ["SRG", "SRS", "DUN07", "DUN07X", "GR"]:
            
            print(kernel)
            
            T = HyperbolicTiling(4,7,2, kernel=kernel)
            T.get_polygon(3)


    def test_polygon_getter_type(self):
        # test whether polygon getter returns correct type
        for kernel in ["SRG", "SRS", "DUN07", "DUN07X", "GR"]:
            
            print(kernel)
            
            T = HyperbolicTiling(4,7,2, kernel=kernel)
            poly = T.get_polygon(3)
            
            if not isinstance(poly, HyperPolygon):
                raise TypeError("Must be an instance of HyperPolygon")
