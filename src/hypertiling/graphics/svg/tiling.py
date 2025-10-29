from typing import Iterable, Sequence, Optional, Tuple, Union
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from .svg_base import to_px, build_svg_attrs, SvgElement
from .geodesic import geodesic_arc


ColorLike = Union[str, Sequence[float], Sequence[Tuple[float, float, float]]]

# --- geometry → SVG helpers ---------------------------------------------------

def _arc_segment(z1: complex, z2: complex, digits: int) -> str:
    arc = geodesic_arc(z1, z2)
    x1, y1 = to_px(z1)
    x2, y2 = to_px(z2)

    if arc["type"] == "line":
        return f" L {np.round(x2, digits)} {np.round(y2, digits)} "

    c = arc["center"]
    r = arc["radius"]
    cx_px, cy_px = to_px(c)
    r_px = np.hypot(x1 - cx_px, y1 - cy_px)

    th1 = np.arctan2((z1 - c).imag, (z1 - c).real)
    th2 = np.arctan2((z2 - c).imag, (z2 - c).real)
    dth = (th2 - th1) % (2*np.pi)

    large_arc_flag = 0
    sweep_flag = int(dth > np.pi)  # flipped if your arcs bent the wrong way

    return (
        f" A {np.round(r_px, digits)} {np.round(r_px, digits)} 0 "
        f"{large_arc_flag} {sweep_flag} {np.round(x2, digits)} {np.round(y2, digits)} "
    )

    """
    A hyperbolic polygon in the Poincaré disk.
    
    Parameters
    ----------
    vertices : Sequence[complex]
        Vertices of the polygon in the unit disk.
    fill : str, optional
        Fill color.
    edgecolor : str, optional
        Edge color.
    lw : float, optional
        Line width.
    digits : int, optional
        Decimal precision for SVG coordinates.
    skip_first : bool, optional
        If True, skip the first vertex (e.g., if it's a center coordinate).
        Default is False.
    **svg_attrs
        Additional SVG attributes.
    """
    
    def __init__(
        self,
        vertices: Sequence[complex],
        fill: str = "white",
        edgecolor: str = "black",
        lw: float = 0.3,
        digits: int = 5,
        skip_first: bool = False,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        
        # Handle skip_first
        verts = list(vertices)
        if skip_first:
            if len(verts) < 4:  # Need at least 3 vertices after skipping
                raise ValueError("Polygon with skip_first=True must have at least 4 elements")
            self.center = verts[0]  # Store center if needed
            self.vertices = verts[1:]
        else:
            self.center = None
            self.vertices = verts
        
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
    
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


class Polygon(SvgElement):
    """A hyperbolic polygon in the Poincaré disk."""
    
    def __init__(
        self,
        vertices: Sequence[complex],
        fill: str = "white",
        edgecolor: str = "black",
        lw: float = 0.3,
        digits: int = 5,
        skip_first: bool = False,
        **svg_attrs,
    ):
        super().__init__(fill, edgecolor, lw, digits, **svg_attrs)
        
        verts = list(vertices)
        if skip_first:
            if len(verts) < 4:
                raise ValueError("Polygon with skip_first=True must have at least 4 elements")
            self.center = verts[0]
            self.vertices = verts[1:]
        else:
            self.center = None
            self.vertices = verts
        
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
    
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
# --- color resolution ---------------------------------------------------------

def _is_scalar_sequence(x, n: int) -> bool:
    try:
        a = np.asarray(x)
        return a.ndim == 1 and a.size == n
    except Exception:
        return False

def _is_rgb_sequence(x, n: int) -> bool:
    try:
        a = np.asarray(x, dtype=float)
        return a.ndim == 2 and a.shape == (n, 3)
    except Exception:
        return False

def _rgb_to_css(rgb: Sequence[float]) -> str:
    arr = np.asarray(rgb, dtype=float)
    if arr.max() <= 1.0:
        arr = np.round(arr * 255.0)
    return f"rgb({int(arr[0])},{int(arr[1])},{int(arr[2])})"

def _resolve_facecolors(facecolors: ColorLike, n: int, cmap: str) -> Tuple[bool, Optional[Sequence[str]]]:
    """
    Returns (individual, colors_css). If individual=False, fill is group-level.
    """
    if isinstance(facecolors, str):
        return False, None
    if _is_rgb_sequence(facecolors, n):
        return True, [_rgb_to_css(rgb) for rgb in facecolors]
    if _is_scalar_sequence(facecolors, n):
        vals = np.asarray(facecolors, dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        vals = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        ccmap = plt.get_cmap(cmap)
        rgba = ccmap(vals)[:, :3]
        return True, [_rgb_to_css(255 * c) for c in rgba]
    return False, None

# --- main element factory -----------------------------------------------------

class Tiling(SvgElement):
    """A collection of hyperbolic polygons forming a tiling."""
    
    def __init__(
        self,
        polygons: Iterable[Sequence[complex]],
        facecolors: ColorLike = "white",
        edgecolor: str = "black",
        lw: float = 0.3,
        cmap: str = "RdYlGn",
        digits: int = 5,
        link: str = "",
        skip_first: bool = False,
        **svg_attrs,
    ):
        super().__init__("none", edgecolor, lw, digits, **svg_attrs)
        
        polygons_list = list(polygons)
        n = len(polygons_list)
        
        individual, colors_css = _resolve_facecolors(facecolors, n, cmap)
        
        self.polygons = []
        for i, verts in enumerate(polygons_list):
            if individual and colors_css is not None:
                fill = colors_css[i]
            elif isinstance(facecolors, str):
                fill = facecolors
            else:
                fill = "white"
            
            poly = Polygon(
                verts,
                fill=fill,
                edgecolor=edgecolor,
                lw=lw,
                digits=digits,
                skip_first=skip_first,
            )
            self.polygons.append(poly)
        
        self.link = link
        self.use_pattern = bool(link)
        self.skip_first = skip_first
    
    def _get_repr_attrs(self) -> dict:
        return {
            "n_polygons": len(self.polygons),
            "edgecolor": self.edgecolor,
            "lw": self.lw,
        }
    
    def _get_str_repr(self) -> str:
        return f"Tiling({len(self.polygons)} polygons)"
    
    def to_svg(self) -> str:
        """Generate SVG <g> element containing all polygons."""
        group_attrs = {
            "stroke": self.edgecolor,
            "stroke-width": self.lw,
        }
        
        if self.polygons and all(p.fill == self.polygons[0].fill for p in self.polygons):
            group_attrs["fill"] = self.polygons[0].fill
        
        group_attrs_str = build_svg_attrs(group_attrs, **self.svg_attrs)
        out = [f"<g {group_attrs_str}>\r"]
        
        if self.use_pattern:
            out.append(
                "<defs>\r"
                "  <pattern id='img1' width='5' height='5'>\r"
                f"    <image href='{self.link}' x='0' y='0' width='45' height='45'/>\r"
                "  </pattern>\r"
                "</defs>\r"
            )
        
        for poly in self.polygons:
            if self.use_pattern:
                poly.fill = "url(#img1)"
            out.append(poly.to_svg() + "\r")
        
        out.append("</g>\r")
        return "".join(out)
    
    def __getitem__(self, idx):
        return self.polygons[idx]
    
    def __len__(self):
        return len(self.polygons)
    
    def __iter__(self):
        return iter(self.polygons)
    
    def set_edgecolor(self, color: str):
        self.edgecolor = color
        for poly in self.polygons:
            poly.set_edgecolor(color)
        return self
    
    def set_linewidth(self, lw: float):
        self.lw = lw
        for poly in self.polygons:
            poly.set_linewidth(lw)
        return self