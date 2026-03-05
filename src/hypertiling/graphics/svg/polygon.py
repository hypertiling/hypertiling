from typing import Sequence
import numpy as np
from .svg_base import to_px, build_svg_attrs, SVGElement
from .geodesic import geodesic_arc


def _arc_segment(z1: complex, z2: complex, digits: int) -> str:
    """
    Build an SVG arc segment from z1 to z2 (already conjugated for y-flip).
    
    Parameters
    ----------
    z1, z2 : complex
        Start and end points (already conjugated in SVG coordinate space).
    digits : int
        Decimal precision.
    
    Returns
    -------
    str
        SVG path segment (e.g., "L x y" or "A rx ry 0 large sweep x y").
    """
    # Note: z1 and z2 are ALREADY conjugated when this function is called
    # We need to un-conjugate them to use geodesic_arc, then re-conjugate
    z1_disk = np.conj(z1)
    z2_disk = np.conj(z2)
    
    arc = geodesic_arc(z1_disk, z2_disk)
    
    x1, y1 = to_px(z1)
    x2, y2 = to_px(z2)

    if arc["type"] == "line":
        return f" L {np.round(x2, digits)} {np.round(y2, digits)} "

    c = arc["center"]
    r = arc["radius"]
    
    # Conjugate center for SVG space
    c_svg = np.conj(c)
    
    cx_px, cy_px = to_px(c_svg)
    r_px = np.hypot(x1 - cx_px, y1 - cy_px)

    # Use conjugated coordinates for angle calculation
    th1 = np.arctan2((z1 - c_svg).imag, (z1 - c_svg).real)
    th2 = np.arctan2((z2 - c_svg).imag, (z2 - c_svg).real)
    dth = (th2 - th1) % (2*np.pi)

    sweep = 0 if dth <= np.pi else 1
    
    # Check if midpoint is inside disk (same logic as Geodesic)
    th_mid = th1 + (0.5 * dth if sweep == 1 else -0.5 * (2*np.pi - dth))
    z_mid = c_svg + r * np.exp(1j * th_mid)
    if abs(np.conj(z_mid)) >= 1 - 1e-12:  # Check in original disk space
        sweep ^= 1

    return (
        f" A {np.round(r_px, digits)} {np.round(r_px, digits)} 0 "
        f"0 {sweep} {np.round(x2, digits)} {np.round(y2, digits)} "
    )


class Polygon(SVGElement):
    """
    A hyperbolic polygon in the Poincaré disk.

    Parameters
    ----------
    vertices : Sequence[complex]
        The vertices of the polygon.
    fill : str, optional
        The fill color of the polygon. Defaults to "white".
    edgecolor : str, optional
        The edge color of the polygon. Defaults to "black".
    lw : float, optional
        The line width of the polygon. Defaults to 0.3.
    digits : int, optional
        The number of digits to round the coordinates to. Defaults to 7.
    skip_first : bool, optional
        If True, the first vertex is considered as the center of the polygon and is skipped. Defaults to True.
    **svg_attrs
        Additional SVG attributes.

    Attributes
    ----------
    center : complex or None
        The center of the polygon if skip_first=True, otherwise None.
    vertices : List[complex]
        The vertices of the polygon.

    Methods
    -------
    _get_repr_attrs()
        Get the attributes that should be included in the __repr__ method.
    _get_str_repr()
        Get the string that should be included in the __str__ method.
    _build_path()
        Build the SVG path 'd' attribute.
    to_svg()
        Generate the SVG <path> element.
    """

    def __init__(
        self,
        vertices: Sequence[complex],
        fill: str = "white",
        edgecolor: str = "black",
        lw: float = 0.3,
        digits: int = 7,
        skip_first: bool = True,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        
        verts = list(vertices)
        if skip_first:
            if len(verts) < 4:
                raise ValueError("[hypertiling] Error: Polygon with skip_first=True must have at least 4 elements")
            self.center = verts[0]
            self.vertices = verts[1:]
        else:
            self.center = None
            self.vertices = verts
        
        if len(self.vertices) < 3:
            raise ValueError("[hypertiling] Error: Polygon must have at least 3 vertices")
    
    def _get_repr_attrs(self) -> dict:
        attrs = {
            "n_vertices": len(self.vertices),
            "fill": self.fill,
            "edgecolor": self.edgecolor,
        }
        if self.center is not None:
            attrs["center"] = self.center
        return attrs
    
    def _get_str_repr(self) -> str:
        return f"Polygon({len(self.vertices)} vertices)"
    
    def _build_path(self) -> str:
        """Build the SVG path 'd' attribute."""
        verts = [np.conj(v) for v in self.vertices]
        x0, y0 = to_px(verts[0])
        parts = [f"M {np.round(x0, self.digits)} {np.round(y0, self.digits)}"]
        for i in range(len(verts)):
            z1 = verts[i]
            z2 = verts[(i + 1) % len(verts)]
            parts.append(_arc_segment(z1, z2, self.digits))
        return " ".join(parts)
    
    def to_svg(self) -> str:
        """Generate the SVG <path> element."""
        d_attr = self._build_path()
        base_attrs = self._build_base_attrs({"d": d_attr})
        attrs_str = build_svg_attrs(base_attrs, **self.svg_attrs)
        return f"<path {attrs_str} />"
    

