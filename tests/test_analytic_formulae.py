import unittest

from hypertiling.util import n_cell_centered, n_vertex_centered

class TestFormulas(unittest.TestCase):
    """
    This class tests the analytic formulae for the number of polygons in a regular tiling
    """

    def test_n_cell_centered(self):
        self.assertEqual(n_cell_centered(3,7,1), 1)
        self.assertEqual(n_cell_centered(3,7,2), 16)
        self.assertEqual(n_cell_centered(3,7,3), 61)
        self.assertEqual(n_cell_centered(3,7,4), 181)
        self.assertEqual(n_cell_centered(3,7,5), 496)
        self.assertEqual(n_cell_centered(3,7,10), 62701)
        self.assertEqual(n_cell_centered(3,7,15), 7713421)
        self.assertEqual(n_cell_centered(3,7,20), 948689776)

    def test_n_vertex_centered(self):
        self.assertEqual(n_vertex_centered(3,7,1), 7)
        self.assertEqual(n_vertex_centered(3,7,2), 35)
        self.assertEqual(n_vertex_centered(3,7,3), 112)   
        self.assertEqual(n_vertex_centered(3,7,5), 847)   
        self.assertEqual(n_vertex_centered(3,7,10), 105875)   
        self.assertEqual(n_vertex_centered(3,7,15), 13023472)   
