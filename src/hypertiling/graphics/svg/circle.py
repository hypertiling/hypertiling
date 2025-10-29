import numpy as np
from .svg_base import to_px, build_svg_attrs

def svg_hyperbolic_circle(
    z0: complex,
    R: float,
    edgecolor: str = "black",
    fill: str = "none",
    lw: float = 1.0,
    digits: int = 5,
    **svg_attrs,
):
    """
    Build an SVG <circle> element for a hyperbolic circle in the Poincaré disk.

    The hyperbolic circle with (hyperbolic) center z0 (|z0|<1) and
    hyperbolic radius R maps to an Euclidean circle with center c_e and radius r_e:
        ρ = tanh(R / 2)
        c_e = ((1 - ρ**2) * z0) / (1 - ρ**2 * |z0|**2)
        r_e = ((1 - |z0|**2) * ρ) / (1 - ρ**2 * |z0|**2)

    This function returns a ready-to-insert SVG <circle> element **in pixel space**,
    using the existing `to_px` mapping.

    Parameters
    ----------
    z0 : complex
        Hyperbolic center (Euclidean coordinate in the unit disk), |z0| < 1.
    R : float
        Hyperbolic radius (geodesic distance in the Poincaré metric).
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
        An SVG <circle> element string.
    """
    # Euclidean circle parameters for a hyperbolic circle
    rho = np.tanh(R / 2.0)
    a2 = abs(z0)**2
    denom = (1 - (rho**2) * a2)

    # center and radius in Euclidean (disk) coordinates
    c_e = ((1 - rho**2) * z0) / denom
    r_e = ((1 - a2) * rho) / denom

    # map to pixel space; follow the same y-flip convention as polygons
    cx, cy = to_px(np.conj(c_e))
    # take a point to the +x direction to measure radius in pixels (affine map)
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