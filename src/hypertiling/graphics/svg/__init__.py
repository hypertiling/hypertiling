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
>>> canvas.add(UnitCircle())                    # doctest: +ELLIPSIS
>>> canvas.add(Tiling(T, skip_first=True))      # doctest: +ELLIPSIS
>>> display(canvas)
"""

from .svg_base import SVGElement, SVGCanvas, display, build_svg_attrs
from .tiling import Tiling, Polygon
from .circle import HyperbolicCircle, UnitCircle
from .geodesic import Geodesic
from .horocycle import Horocycle
from .hypercycle import Hypercycle

__all__ = [
    # Base classes
    'SVGElement',
    'SVGCanvas',
    'display',
    
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
]