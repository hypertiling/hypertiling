from typing import Iterable, Sequence
import numpy as np
from .svg_base import to_px, build_svg_attrs, SVGElement
from .polygon import Polygon
from .geodesic import geodesic_arc
from .color import _resolve_facecolors, ColorLike


class Tiling(SVGElement):
    """A collection of hyperbolic polygons forming a tiling.

    Parameters
    ----------
    polygons : Iterable[Sequence[complex]]
        A sequence of sequences of complex numbers representing the vertices of the polygons.
    facecolors : ColorLike, optional
        A color or a sequence of colors to use for the polygons. If individual, each polygon will be assigned a color from the sequence.
        If not individual, all polygons will be assigned the same color.
    edgecolor : str, optional
        The color of the polygon edges.
    lw : float, optional
        The linewidth of the polygon edges.
    cmap : str, optional
        The matplotlib colormap key string to use for generating colors.
    digits : int, optional
        The number of digits to round SVG coordinates to.
    link : str, optional
        A link to an image to use as a pattern.
    skip_first : bool, optional
        Whether or not to skip the first polygon when generating SVG elements.
    **svg_attrs
        Additional SVG attributes to add to the generated SVG element.

    Attributes
    ----------
    polygons : List[Polygon]
        A list of Polygon objects representing the polygons in the tiling.
    edgecolor : str
        The color of the polygon edges.
    lw : float
        The linewidth of the polygon edges.
    digits : int
        The number of digits to round SVG coordinates to.
    link : str
        A link to an image to use as a pattern.
    skip_first : bool
        Whether or not to skip the first polygon when generating SVG elements.
    use_pattern : bool
        Whether or not to use a pattern.

    Methods
    -------
    to_svg : str
        Generate SVG <g> element containing all polygons.
    __getitem__ : Polygon
        Get a Polygon object from the tiling by index.
    __len__ : int
        Get the number of polygons in the tiling.
    __iter__ : Iterable[Polygon]
        Iterate over the polygons in the tiling.
    set_edgecolor : Tiling
        Set the edgecolor of all polygons in the tiling.
    set_linewidth : Tiling
        Set the linewidth of all polygons in the tiling.
    """
    
    def __init__(
        self,
        polygons: Iterable[Sequence[complex]],
        facecolors: ColorLike = "white",
        edgecolor: str = "black",
        lw: float = 0.3,
        cmap: str = "RdYlGn",
        digits: int = 7,
        link: str = "",
        skip_first: bool = True,
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