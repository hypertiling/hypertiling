import numpy as np
from .svg_base import to_px, build_svg_attrs, SvgElement

def geodesic_arc(z1: complex, z2: complex, tol: float = 1e-14):
    """
    Return geometric parameters of the Poincaré-disk geodesic through z1 and z2.

    The geodesic is:
      • a straight diameter (if z1, z2, 0 are colinear), or
      • an Euclidean circle orthogonal to the unit circle passing through z1, z2.

    Parameters
    ----------
    z1, z2 : complex
        Points inside or on the unit disk.
    tol : float, optional
        Numerical tolerance for detecting a diameter.

    Returns
    -------
    dict
        For a diameter:
            {"type": "line"}
        For a circle:
            {"type": "circle", "center": complex, "radius": float}
    """
    # identical points → degenerate
    if abs(z1 - z2) < tol:
        return {"type": "line"}

    # Diameter test: Im(z1 * conj(z2)) ≈ 0 → same radial line
    if abs(np.imag(z1 * np.conj(z2))) < tol:
        return {"type": "line"}

    # Circle orthogonal to |z|=1 through z1 and z2:
    # Re(c * conj(z)) = (1 + |z|^2) / 2  for each point z on the circle
    u1, v1 = z1.real, z1.imag
    u2, v2 = z2.real, z2.imag
    b1 = 0.5 * (1.0 + u1*u1 + v1*v1)
    b2 = 0.5 * (1.0 + u2*u2 + v2*v2)
    A = np.array([[u1, v1],
                  [u2, v2]], dtype=float)
    b = np.array([b1, b2], dtype=float)

    try:
        cx, cy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # fallback to diameter if numerical degeneracy
        return {"type": "line"}

    R2 = cx*cx + cy*cy - 1.0  # from orthogonality |c|^2 = R^2 + 1
    R2 = max(R2, 0.0)
    return {"type": "circle", "center": complex(cx, cy), "radius": np.sqrt(R2)}

class Geodesic(SvgElement):
    """A geodesic arc in the Poincaré disk."""
    
    def __init__(
        self,
        z1: complex,
        z2: complex,
        edgecolor: str = "black",
        lw: float = 1.0,
        digits: int = 5,
        **svg_attrs,
    ):
        super().__init__("none", edgecolor, lw, digits, **svg_attrs)
        self.z1 = z1
        self.z2 = z2
    
    def _get_repr_attrs(self) -> dict:
        return {
            "z1": self.z1,
            "z2": self.z2,
            "edgecolor": self.edgecolor,
            "lw": self.lw,
        }
    
    def _get_str_repr(self) -> str:
        return f"Geodesic({self.z1:.2f} → {self.z2:.2f})"
    
    def to_svg(self) -> str:
        """Generate the SVG <path> element."""
        if abs(self.z1 - self.z2) < 1e-12:
            return ""
        
        arc = geodesic_arc(self.z1, self.z2)
        
        z1_svg = np.conj(self.z1)
        z2_svg = np.conj(self.z2)
        
        x1, y1 = to_px(z1_svg)
        x2, y2 = to_px(z2_svg)
        
        base_attrs = self._build_base_attrs()
        
        if arc["type"] == "line":
            base_attrs["d"] = (
                f"M {np.round(x1, self.digits)} {np.round(y1, self.digits)} "
                f"L {np.round(x2, self.digits)} {np.round(y2, self.digits)}"
            )
            attrs_str = build_svg_attrs(base_attrs, **self.svg_attrs)
            return f"<path {attrs_str} />"
        
        c: complex = arc["center"]
        r: float = arc["radius"]
        c_svg = np.conj(c)
        
        Cx, Cy = to_px(c_svg)
        r_px = float(np.hypot(x1 - Cx, y1 - Cy))
        
        th1 = np.arctan2((z1_svg - c_svg).imag, (z1_svg - c_svg).real)
        th2 = np.arctan2((z2_svg - c_svg).imag, (z2_svg - c_svg).real)
        dth = (th2 - th1) % (2*np.pi)
        
        sweep = 0 if dth <= np.pi else 1
        
        th_mid = th1 + (0.5 * dth if sweep == 1 else -0.5 * (2*np.pi - dth))
        z_mid = c_svg + r * np.exp(1j * th_mid)
        if abs(np.conj(z_mid)) >= 1 - 1e-12:
            sweep ^= 1
        
        base_attrs["d"] = (
            f"M {np.round(x1, self.digits)} {np.round(y1, self.digits)} "
            f"A {np.round(r_px, self.digits)} {np.round(r_px, self.digits)} 0 "
            f"0 {sweep} {np.round(x2, self.digits)} {np.round(y2, self.digits)}"
        )
        
        attrs_str = build_svg_attrs(base_attrs, **self.svg_attrs)
        return f"<path {attrs_str} />"
    
    def set_endpoints(self, z1: complex, z2: complex):
        self.z1 = z1
        self.z2 = z2
        return self
    
    def set_start(self, z1: complex):
        self.z1 = z1
        return self
    
    def set_end(self, z2: complex):
        self.z2 = z2
        return self