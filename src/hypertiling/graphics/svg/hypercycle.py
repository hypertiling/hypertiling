import numpy as np
from .svg_base import to_px, build_svg_attrs, SVGElement
class Hypercycle(SVGElement):
    """A hypercycle in the Poincaré disk (NOT YET IMPLEMENTED)."""
    
    def __init__(
        self,
        angle1: float,
        angle2: float,
        R: float,
        fill: str = "none",
        edgecolor: str = "black",
        lw: float = 1.0,
        digits: int = 5,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        self.angle1 = angle1
        self.angle2 = angle2
        self.R = R
    
    def _get_repr_attrs(self) -> dict:
        return {
            "angle1": self.angle1,
            "angle2": self.angle2,
            "R": self.R,
            "edgecolor": self.edgecolor,
        }
    
    def _get_str_repr(self) -> str:
        return f"Hypercycle(axis=[{self.angle1:.2f}, {self.angle2:.2f}], R={self.R:.2f}) [NOT IMPLEMENTED]"
    
    def to_svg(self) -> str:
        raise NotImplementedError(
            "Hypercycle.to_svg() is not yet implemented. "
            "The correct mathematical formula needs to be derived."
        )
    
    def set_axis(self, angle1: float, angle2: float):
        self.angle1 = angle1
        self.angle2 = angle2
        return self
    
    def set_distance(self, R: float):
        self.R = R
        return self