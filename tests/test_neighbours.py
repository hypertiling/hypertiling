import unittest
from hypertiling import HyperbolicTiling
from hypertiling.neighbours import find



class TestCore(unittest.TestCase):
    def test_num_neighbours(self):
        nlayer = 4
        p, q = 3, 7
        kernels = ["manu", "flo"]
        nn_algorithms = ["optimized"]
        for k in kernels:
            for which in nn_algorithms:
                for nl in range(2,nlayer):
                    for cen in ["cell", "vertex"]:
                        print("Constructing lattice (p,q,n) = ", p, q, nl, ", center = ", cen, ", kernel =", k)
                        T = HyperbolicTiling(p, q, nl, kernel=k, center=cen)
                        T.generate()
                        nbrs = find(T, which=which)
                        for n in nbrs:
                            self.assertFalse(len(n) > p)
                            self.assertFalse(len(n) < 1)



        nlayer = 4
        p, q = 7, 3
        kernels = ["manu", "flo"]
        nn_algorithms = ["optimized"]
        for k in kernels:
            for which in nn_algorithms:
                for nl in range(2,nlayer):
                    for cen in ["cell", "vertex"]:
                        print("Constructing lattice (p,q,n) = ", p, q, nl, ", center = ", cen, ", kernel =", k)
                        T = HyperbolicTiling(p, q, nl, kernel=k, center=cen)
                        T.generate()
                        nbrs = find(T, which=which)
                        for n in nbrs:
                            self.assertFalse(len(n) > p)
                            self.assertFalse(len(n) < 1)


if __name__ == '__main__':
    unittest.main()
