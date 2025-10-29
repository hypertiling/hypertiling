# test_hyperbolic_metrics_unittest.py
import math
import unittest
from hypertiling.util import _check_hyperbolic

"""
Unit tests for hyperbolic geometry primitives used in regular {p,q} tilings
within the Poincaré disk model.

These tests validate analytic expressions for:
    - Geodesic edge length h^(p,q)
    - Dual edge length h^(q,p)
    - Geodesic circumradius (cell radius) h_r
    - Euclidean circumradius r_0
    - Geodesic inradius r

Reference values are taken from hypertiling release publication
Additional tests enforce geometric identities such as:
    h_dual(p,q) = h(q,p)  and  h_dual(p,q) = 2 * inradius(p,q),
and ensure functions correctly raise ValueError for non-hyperbolic {p,q}.
"""

from hypertiling.util import (
    edge_length_geodesic,
    dual_edge_length_geodesic,
    cell_radius_geodesic,
    fundamental_radius,
    inradius_geodesic
)

# Reference values from the hypertiling release publication
# (p, q, h_(p,q), h_(q,p), h_r, r0)
CASES = [
    (3, 7, 0.566256, 1.090550, 0.620672, 0.300743),
    (4, 5, 1.061275, 1.253739, 0.842481, 0.397975),
    (5, 4, 1.253739, 1.061275, 0.842481, 0.397975),
    (6, 4, 1.762747, 1.316958, 1.146216, 0.517638),
    (7, 3, 1.090550, 0.566256, 0.620673, 0.300743),
    (8, 4, 2.448452, 1.528571, 1.528570, 0.643594),
    (12, 10, 3.951080, 3.612418, 3.132385, 0.916418),
]

RTOL = 1e-5
ATOL = 1e-5


class TestHyperbolicMetrics(unittest.TestCase):
    def test_edge_lengths_match_table(self):
        for p, q, h_pq, h_qp, h_r, r0 in CASES:
            with self.subTest(p=p, q=q, what="edge_length"):
                self.assertTrue(
                    math.isclose(edge_length_geodesic(p, q), h_pq, rel_tol=RTOL, abs_tol=ATOL),
                    msg=f"h_(p,q) mismatch for {{p,q}}={{ {p},{q} }}",
                )
            with self.subTest(p=p, q=q, what="dual_edge_length"):
                self.assertTrue(
                    math.isclose(dual_edge_length_geodesic(p, q), h_qp, rel_tol=RTOL, abs_tol=ATOL),
                    msg=f"h_(q,p) mismatch for dual of {{p,q}}={{ {p},{q} }}",
                )

    def test_cell_radius_and_fundamental_radius_match_table(self):
        for p, q, h_pq, h_qp, h_r, r0 in CASES:
            with self.subTest(p=p, q=q, what="cell_radius_geodesic"):
                self.assertTrue(
                    math.isclose(cell_radius_geodesic(p, q), h_r, rel_tol=RTOL, abs_tol=ATOL),
                    msg=f"R mismatch for {{p,q}}={{ {p},{q} }}",
                )
            with self.subTest(p=p, q=q, what="fundamental_radius"):
                self.assertTrue(
                    math.isclose(fundamental_radius(p, q), r0, rel_tol=RTOL, abs_tol=ATOL),
                    msg=f"r0 mismatch for {{p,q}}={{ {p},{q} }}",
                )

    def test_dual_equals_swapped_edge(self):
        for p, q, *_ in CASES:
            with self.subTest(p=p, q=q, what="dual_identity"):
                self.assertTrue(
                    math.isclose(
                        dual_edge_length_geodesic(p, q),
                        edge_length_geodesic(q, p),
                        rel_tol=RTOL,
                        abs_tol=ATOL,
                    ),
                    msg=f"Dual identity failed for {{p,q}}={{ {p},{q} }}",
                )

    def test_fundamental_radius_consistency(self):
        for p, q, *_ in CASES:
            with self.subTest(p=p, q=q, what="rho=tanh(R/2)"):
                R = cell_radius_geodesic(p, q)
                rho = fundamental_radius(p, q)
                self.assertTrue(
                    math.isclose(rho, math.tanh(R / 2.0), rel_tol=RTOL, abs_tol=ATOL),
                    msg=f"rho != tanh(R/2) for {{p,q}}={{ {p},{q} }}",
                )
    def test_check_hyperbolic_raises_on_non_hyperbolic(self):
        # Boundary/Euclidean or spherical cases should raise (p-2)(q-2) <= 4
        for p, q in [(3, 6), (3, 3), (6, 3)]:
            with self.subTest(p=p, q=q):
                with self.assertRaises(ValueError):
                    _check_hyperbolic(p, q)


    def test_dual_edge_equals_twice_inradius(self):
        for p, q, *_ in CASES:
            with self.subTest(p=p, q=q, what="dual_edge == 2 * inradius"):
                dual = dual_edge_length_geodesic(p, q)
                r = inradius_geodesic(p, q)
                self.assertTrue(
                    math.isclose(dual, 2 * r, rel_tol=RTOL, abs_tol=ATOL),
                    msg=f"h_dual != 2*r for {{p,q}}={{ {p},{q} }}",
                )



if __name__ == "__main__":
    unittest.main()
