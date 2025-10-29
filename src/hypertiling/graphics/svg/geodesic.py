import numpy as np
from .svg_base import to_px, build_svg_attrs

def geodesic_arc(z1: complex, z2: complex, tol: float = 1e-14):
    """
    Return geometric parameters of the Poincaré-disk geodesic through z1 and z2.

    The geodesic is:
      • a straight diameter (if z1, z2, 0 are colinear), or
      • an Euclidean circle orthogonal to the unit circle passing through z1, z2.

    Parameters
    ----------
    z1, z2 : complex
        Points inside or on the unit disk.
    tol : float, optional
        Numerical tolerance for detecting a diameter.

    Returns
    -------
    dict
        For a diameter:
            {"type": "line"}
        For a circle:
            {"type": "circle", "center": complex, "radius": float}
    """
    # identical points → degenerate
    if abs(z1 - z2) < tol:
        return {"type": "line"}

    # Diameter test: Im(z1 * conj(z2)) ≈ 0 → same radial line
    if abs(np.imag(z1 * np.conj(z2))) < tol:
        return {"type": "line"}

    # Circle orthogonal to |z|=1 through z1 and z2:
    # Re(c * conj(z)) = (1 + |z|^2) / 2  for each point z on the circle
    u1, v1 = z1.real, z1.imag
    u2, v2 = z2.real, z2.imag
    b1 = 0.5 * (1.0 + u1*u1 + v1*v1)
    b2 = 0.5 * (1.0 + u2*u2 + v2*v2)
    A = np.array([[u1, v1],
                  [u2, v2]], dtype=float)
    b = np.array([b1, b2], dtype=float)

    try:
        cx, cy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # fallback to diameter if numerical degeneracy
        return {"type": "line"}

    R2 = cx*cx + cy*cy - 1.0  # from orthogonality |c|^2 = R^2 + 1
    R2 = max(R2, 0.0)
    return {"type": "circle", "center": complex(cx, cy), "radius": np.sqrt(R2)}


def svg_geodesic(
    z1: complex,
    z2: complex,
    edgecolor: str = "black",
    lw: float = 1.0,
    digits: int = 5,
    **svg_attrs,
) -> str:
    """
    SVG <path> for the hyperbolic geodesic between z1 and z2 in the Poincaré disk.
    Uses `geodesic_arc(z1, z2)` to get the geometric primitive (line or circle),
    then emits either an 'L' segment or an 'A' arc. Supports ideal points (|z|≈1).
    
    Parameters
    ----------
    z1, z2 : complex
        Points in the Poincaré disk.
    edgecolor : str, optional
        Stroke color.
    lw : float, optional
        Stroke width.
    digits : int, optional
        Decimal digits for SVG numeric attributes.
    **svg_attrs
        Additional SVG attributes (e.g., opacity=0.5, stroke_dasharray="3,3").
    """
    if abs(z1 - z2) < 1e-12:
        return ""  # degenerate

    arc = geodesic_arc(z1, z2)

    # Conjugate for y-flip
    z1_svg = np.conj(z1)
    z2_svg = np.conj(z2)
    
    x1, y1 = to_px(z1_svg)
    x2, y2 = to_px(z2_svg)

    # Basis-Attribute
    base_attrs = {
        "fill": "none",
        "stroke": edgecolor,
        "stroke-width": lw,
    }
    attrs_str = build_svg_attrs(base_attrs, **svg_attrs)

    if arc["type"] == "line":
        return (
            f"<path d='M {np.round(x1, digits)} {np.round(y1, digits)} "
            f"L {np.round(x2, digits)} {np.round(y2, digits)}' "
            f"{attrs_str} />"
        )

    # Circle case
    c: complex = arc["center"]
    r: float = arc["radius"]
    c_svg = np.conj(c)  # Also conjugate the center!

    Cx, Cy = to_px(c_svg)
    r_px = float(np.hypot(x1 - Cx, y1 - Cy))

    # Use CONJUGATED coordinates for angle calculations
    th1 = np.arctan2((z1_svg - c_svg).imag, (z1_svg - c_svg).real)
    th2 = np.arctan2((z2_svg - c_svg).imag, (z2_svg - c_svg).real)
    dth = (th2 - th1) % (2*np.pi)

    sweep = 0 if dth <= np.pi else 1

    # Test midpoint with conjugated values
    th_mid = th1 + (0.5 * dth if sweep == 1 else -0.5 * (2*np.pi - dth))
    z_mid = c_svg + r * np.exp(1j * th_mid)
    if abs(np.conj(z_mid)) >= 1 - 1e-12:  # Check in original space
        sweep ^= 1

    return (
        f"<path d='M {np.round(x1, digits)} {np.round(y1, digits)} "
        f"A {np.round(r_px, digits)} {np.round(r_px, digits)} 0 "
        f"0 {sweep} {np.round(x2, digits)} {np.round(y2, digits)}' "
        f"{attrs_str} />"
    )