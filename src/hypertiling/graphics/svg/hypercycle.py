import numpy as np
from .svg_base import to_px, build_svg_attrs


def svg_hypercycle(
    phi: float,
    R: float,
    side: int = +1,
    edgecolor: str = "black",
    fill: str = "none",
    lw: float = 1.0,
    digits: int = 5,
    **svg_attrs,
):
    """
    Build an SVG <circle> element for a *hypercycle* in the Poincaré disk.

    Definition
    ----------
    A hypercycle is the locus of points at signed hyperbolic distance R from a
    given geodesic (the "axis"). Here the axis is the diameter through angle φ,
    i.e. the geodesic whose ideal endpoints are e^{iφ} and e^{i(φ+π)}.

    Method
    ------
    1) In the UHP model, hypercycles at distance R from the vertical geodesic x=0
       are the rays: x = ±sinh(R) * y  (sign picks the side).
    2) Map three such points to the disk via the inverse Cayley transform
         z = (w - i) / (w + i)   with w = x + i y,
       then rotate by e^{iφ} to align the chosen axis.
    3) Fit the unique Euclidean circle through the three disk points.
    4) Convert that circle to pixel space and return an SVG <circle> element.

    Parameters
    ----------
    phi : float
        Axis direction in radians. The axis geodesic is the diameter through angle φ.
    R : float
        Signed hyperbolic offset from the axis geodesic. Positive/negative selects
        the two branches (together with `side`).
    side : {+1, -1}, optional
        Choose which side of the axis (the two hypercycle branches). Default +1.
    edgecolor : str, optional
        Stroke color.
    fill : str, optional
        Fill color or pattern (e.g. 'none' or 'url(#img1)').
    lw : float, optional
        Stroke width in pixels.
    digits : int, optional
        Decimal digits for SVG attributes.
    **svg_attrs
        Additional SVG attributes (e.g., opacity=0.5, stroke_dasharray="3,3").

    Returns
    -------
    str
        SVG <circle> element string.

    Notes
    -----
    - As R → 0, the hypercycle tends to the axis geodesic (a diameter).
      As |R| grows, the hypercycle approaches a horocycle.
    """
    # --- helper: inverse Cayley transform (UHP → disk) ---
    def uhp_to_disk(w: complex) -> complex:
        return (w - 1j) / (w + 1j)

    # --- sample three UHP points on the canonical hypercycle x = s * y ---
    s = np.sinh(R) * (1 if side >= 0 else -1)
    ys = np.array([0.4, 0.8, 1.6])  # any positive values spanning a scale
    ws = s * ys + 1j * ys

    # map to disk and rotate to set the axis direction φ
    rot = np.exp(1j * phi)
    zs = [rot * uhp_to_disk(w) for w in ws]

    # --- fit Euclidean circle through three points in the disk ---
    (x1, y1), (x2, y2), (x3, y3) = [(z.real, z.imag) for z in zs]

    A = np.array([
        [2*(x2 - x1), 2*(y2 - y1), x2**2 + y2**2 - x1**2 - y1**2],
        [2*(x3 - x2), 2*(y3 - y2), x3**2 + y3**2 - x2**2 - y2**2],
    ], dtype=float)

    # Solve for center (cx, cy) in disk coords: A[:, :2] @ [cx, cy] = A[:, 2]
    sol = np.linalg.lstsq(A[:, :2], A[:, 2], rcond=None)[0]
    c_e = complex(sol[0], sol[1])
    r_e = abs(c_e - zs[0])  # radius in disk coords

    # --- map to SVG pixel space ---
    cx, cy = to_px(np.conj(c_e))
    px, py = to_px(np.conj(c_e + r_e))
    r_px = float(np.hypot(px - cx, py - cy))

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