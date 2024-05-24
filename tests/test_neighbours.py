import unittest
from hypertiling import HyperbolicTiling
from hypertiling import TilingKernels
from hypertiling.ion import set_verbosity_level

set_verbosity_level("Warning")

class TestCore(unittest.TestCase):
    def test_num_neighbours(self):
        nlayer = 3
        for p,q in [(7,3), (8,3), (4,5), (4,6), (4,7), (4,8), (5,5), (5,6)]:
            for nl in range(2, nlayer):
                for cen in ["cell", "vertex"]:
                    T = HyperbolicTiling(p, q, nl, kernel=TilingKernels.StaticRotationalSector, center=cen)
                    
                    print(p, q, nl, cen, "RBF")
                    nbrs = T.get_nbrs_list(method="RBF")
                    for n in nbrs:
                        self.assertFalse(len(n) > p)
                        self.assertFalse(len(n) < 1)
                    
                    print(p, q, nl, cen, "RO")
                    nbrs = T.get_nbrs_list(method="RO")
                    for n in nbrs:
                        self.assertFalse(len(n) > p)
                        self.assertFalse(len(n) < 1)

                    print(p, q, nl, cen, "EMO")
                    nbrs = T.get_nbrs_list(method="EMO")
                    for n in nbrs:
                        self.assertFalse(len(n) > p)
                        self.assertFalse(len(n) < 1)

    


if __name__ == '__main__':
    unittest.main()
