"""
SVG drawing utilities for hyperbolic tilings in the Poincaré disk.

This module provides object-oriented SVG elements and a canvas for rendering
hyperbolic geometric objects.

Examples
--------
>>> from hypertiling.graphics.svg import SVGCanvas, Tiling, Geodesic, UnitCircle
>>> from hypertiling import HyperbolicTiling
>>> 
>>> T = HyperbolicTiling(5, 4, 5)
>>> canvas = SVGCanvas(width=600, height=600)
>>> canvas.add(UnitCircle())
SVGCanvas(1 elements)
>>> canvas.add(Tiling(T, skip_first=True))
SVGCanvas(2 elements)
>>> draw_svg(canvas.render())
<IPython.core.display.SVG object>
"""

from .svg_base import SVGElement, SVGCanvas, SVGGroup, draw_svg, build_svg_attrs
from .tiling import Tiling, Polygon
from .circle import HyperbolicCircle, UnitCircle
from .geodesic import Geodesic
from .horocycle import Horocycle
from .hypercycle import Hypercycle
from .svg import make_svg, write_svg


__all__ = [
    # Base classes
    'SVGElement',
    'SVGCanvas',
    'SVGGroup',
    'draw_svg',
    
    # Geometric elements
    'Polygon',
    'Tiling',
    'Geodesic',
    'HyperbolicCircle',
    'Horocycle',
    'Hypercycle',
    'UnitCircle',
    
    # Utilities
    'build_svg_attrs',

    # Tombstones
    'make_svg',
    'write_svg',

]