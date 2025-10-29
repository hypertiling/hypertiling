from .svg_base import to_px, build_svg_attrs
import numpy as np


def svg_horocycle(
    z1: complex,
    R: float,
    edgecolor: str = "black",
    fill: str = "none",
    lw: float = 1.0,
    digits: int = 5,
    **svg_attrs,
):
    """
    Build an SVG <circle> element for a *horocycle* in the Poincaré disk model.

    A horocycle is the limiting case of a hyperbolic circle whose center lies
    on the boundary (|z1| = 1).  It appears as an Euclidean circle tangent to
    the unit circle at z1 and entirely contained within it.

    The Euclidean circle parameters are:
        r_e = 0.5 * sech^2(R / 2)
        c_e = (1 - r_e) * z1

    where R is the (inward) hyperbolic distance from the boundary point z1 to
    the horocycle itself.  The direction of z1 determines the point of tangency.

    Parameters
    ----------
    z1 : complex
        Boundary point on the unit circle where the horocycle is tangent (|z1| ≈ 1).
    R : float
        Hyperbolic distance of the horocycle from the boundary along its normal.
    edgecolor : str, optional
        Stroke color.
    fill : str, optional
        Fill color or pattern (e.g., 'none' or 'url(#img1)').
    lw : float, optional
        Stroke width in pixels.
    digits : int, optional
        Decimal digits for SVG numeric attributes.
    **svg_attrs
        Additional SVG attributes (e.g., opacity=0.5, stroke_dasharray="3,3").

    Returns
    -------
    str
        SVG <circle> element string.
    """
    # Euclidean parameters for the horocycle
    r_e = 0.5 * (1 / np.cosh(R / 2.0)) ** 2  # = 0.5 * sech^2(R/2)
    c_e = (1 - r_e) * z1

    # map to pixel space (with y-flip via conjugation)
    cx, cy = to_px(np.conj(c_e))
    px, py = to_px(np.conj(c_e + r_e))
    r_px = np.hypot(px - cx, py - cy)

    # Basis-Attribute
    base_attrs = {
        "cx": np.round(cx, digits),
        "cy": np.round(cy, digits),
        "r": np.round(r_px, digits),
        "fill": fill,
        "stroke": edgecolor,
        "stroke-width": lw,
    }
    
    # Wrapper nutzen
    attrs_str = build_svg_attrs(base_attrs, **svg_attrs)
    
    return f"<circle {attrs_str} />"

