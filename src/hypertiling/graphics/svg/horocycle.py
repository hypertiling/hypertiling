import numpy as np
from .svg_base import to_px, build_svg_attrs, SVGElement
class Horocycle(SVGElement):
    """A horocycle in the Poincaré disk model."""
    
    def __init__(
        self,
        z1: complex,
        R: float,
        fill: str = "none",
        edgecolor: str = "black",
        lw: float = 1.0,
        digits: int = 5,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        self.z1 = z1
        self.R = R
    
    def _get_repr_attrs(self) -> dict:
        return {
            "z1": self.z1,
            "R": self.R,
            "fill": self.fill,
            "edgecolor": self.edgecolor,
            "lw": self.lw,
        }
    
    def _get_str_repr(self) -> str:
        return f"Horocycle(boundary={self.z1:.2f}, R={self.R:.2f})"
    
    def _compute_euclidean_params(self) -> tuple[complex, float]:
        r_e = 0.5 * (1 / np.cosh(self.R / 2.0)) ** 2
        c_e = (1 - r_e) * self.z1
        return c_e, r_e
    
    def to_svg(self) -> str:
        c_e, r_e = self._compute_euclidean_params()
        
        cx, cy = to_px(np.conj(c_e))
        px, py = to_px(np.conj(c_e + r_e))
        r_px = np.hypot(px - cx, py - cy)
        
        base_attrs = self._build_base_attrs({
            "cx": np.round(cx, self.digits),
            "cy": np.round(cy, self.digits),
            "r": np.round(r_px, self.digits),
        })
        
        attrs_str = build_svg_attrs(base_attrs, **self.svg_attrs)
        return f"<circle {attrs_str} />"
    
    def set_boundary_point(self, z1: complex):
        self.z1 = z1
        return self
    
    def set_distance(self, R: float):
        self.R = R
        return self