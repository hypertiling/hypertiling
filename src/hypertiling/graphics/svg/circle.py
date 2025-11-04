import numpy as np
from .svg_base import to_px, build_svg_attrs, SVGElement


class UnitCircle(SVGElement):
    """The unit circle boundary of the Poincaré disk.

    Parameters
    ----------
    center : tuple[float, float]
        The center of the unit circle.
    radius : float
        The radius of the unit circle.
    fill : str, optional
        The fill color of the unit circle. Defaults to "none".
    edgecolor : str, optional
        The edge color of the unit circle. Defaults to "black".
    lw : float, optional
        The line width of the unit circle. Defaults to 1.0.
    digits : int, optional
        The number of digits to round the coordinates to. Defaults to 7.
    **svg_attrs
        Additional SVG attributes.

    Attributes
    ----------
    center : tuple[float, float]
        The center of the unit circle.
    radius : float
        The radius of the unit circle.
    """
    
    def __init__(
        self,
        center: tuple[float, float] = (0, 0),
        radius: float = 1.0,
        fill: str = "none",
        edgecolor: str = "black",
        lw: float = 1.0,
        digits: int = 7,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        self.center = center
        self.radius = radius
    
    def _get_repr_attrs(self) -> dict:
        return {
            "center": self.center,
            "radius": self.radius,
            "edgecolor": self.edgecolor,
            "lw": self.lw,
        }
    
    def _get_str_repr(self) -> str:
        return f"UnitCircle(r={self.radius}, center={self.center})"
    
    def to_svg(self) -> str:
        """Generate the SVG <circle> element."""
        cx, cy = self.center
        r_px = to_px(complex(self.radius, 0))[0] - to_px(complex(0, 0))[0]
        cx_px, cy_px = to_px(complex(cx, cy))
        
        base_attrs = self._build_base_attrs({
            "cx": np.round(cx_px, self.digits),
            "cy": np.round(cy_px, self.digits),
            "r": np.round(r_px, self.digits),
        })
        
        attrs_str = build_svg_attrs(base_attrs, **self.svg_attrs)
        return f"<circle {attrs_str} />"
    
    def set_center(self, center: tuple[float, float]):
        """Set the center of the unit circle."""
        self.center = center
        return self
    
    def set_radius(self, radius: float):
        """Set the radius of the unit circle."""
        self.radius = radius
        return self


class HyperbolicCircle(SVGElement):
    """A hyperbolic circle in the Poincaré disk.

    Parameters
    ----------
    z0 : complex
        The center of the hyperbolic circle.
    R : float
        The Euclidean distance from the center of the hyperbolic circle.
    fill : str, optional
        The fill color of the hyperbolic circle. Defaults to "none".
    edgecolor : str, optional
        The edge color of the hyperbolic circle. Defaults to "black".
    lw : float, optional
        The line width of the hyperbolic circle. Defaults to 1.0.
    digits : int, optional
        The number of digits to round the coordinates to. Defaults to 7.
    **svg_attrs
        Additional SVG attributes.

    Attributes
    ----------
    z0 : complex
        The center of the hyperbolic circle.
    R : float
        The Euclidean distance from the center of the hyperbolic circle.
    """
    
    def __init__(
        self,
        z0: complex,
        R: float,
        fill: str = "none",
        edgecolor: str = "black",
        lw: float = 1.0,
        digits: int = 7,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        self.z0 = z0
        self.R = R
    
    def _get_repr_attrs(self) -> dict:
        return {
            "z0": self.z0,
            "R": self.R,
            "fill": self.fill,
            "edgecolor": self.edgecolor,
            "lw": self.lw,
        }
    
    def _get_str_repr(self) -> str:
        return f"HyperbolicCircle(center={self.z0:.2f}, R={self.R:.2f})"
    
    def _compute_euclidean_params(self) -> tuple[complex, float]:
        """Compute the Euclidean parameters of the hyperbolic circle."""
        rho = np.tanh(self.R / 2.0)
        a2 = abs(self.z0)**2
        denom = (1 - (rho**2) * a2)
        
        c_e = ((1 - rho**2) * self.z0) / denom
        r_e = ((1 - a2) * rho) / denom
        
        return c_e, r_e
    
    def to_svg(self) -> str:
        """Generate the SVG <circle> element."""
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
    
    def set_center(self, z0: complex):
        """Set the center of the hyperbolic circle."""
        self.z0 = z0
        return self
    
    def set_radius(self, R: float):
        """Set the Euclidean distance from the center of the hyperbolic circle."""
        self.R = R
        return self
