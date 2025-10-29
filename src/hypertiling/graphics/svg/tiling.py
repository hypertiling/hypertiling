from typing import Iterable, Sequence, Optional, Tuple, Union
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from .svg_base import to_px, build_svg_attrs
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


def svg_cell_path(pgon: Sequence[complex], digits: int) -> str:
    """
    Build one polygon cell as an SVG <path> 'd' attribute (no styling).
    Assumes pgon[1:] are the vertices (complex, unit disk). Uses y-flip via conj.
    """
    verts = [np.conj(v) for v in pgon[1:]]
    x0, y0 = to_px(verts[0])
    parts = [f"M {np.round(x0, digits)} {np.round(y0, digits)}"]
    for i in range(len(verts)):
        z1 = verts[i]
        z2 = verts[(i + 1) % len(verts)]
        parts.append(_arc_segment(z1, z2, digits))
    return " ".join(parts)

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


def svg_tiling(
    tiling: Iterable[Sequence[complex]],
    facecolors: ColorLike = "white",
    edgecolor: str = "black",
    lw: float = 0.3,
    cmap: str = "RdYlGn",
    digits: int = 5,
    link: str = "",
    **svg_attrs,
) -> str:
    """
    Return a *single SVG element* (<g>…</g>) containing all tiling cells as <path>s.
    No <svg> header/footer; ready to append into an SvgBuilder or another group.
    """
    tiling = list(tiling)
    n = len(tiling)
    individual, colors_css = _resolve_facecolors(facecolors, n, cmap)

    # Basis-Attribute für <g>
    group_attrs = {
        "stroke": edgecolor,
        "stroke-width": lw,
    }
    if not individual:
        group_fill = facecolors if isinstance(facecolors, str) else "white"
        group_attrs["fill"] = group_fill
    
    # Wrapper nutzen - vereinheitlicht!
    group_attrs_str = build_svg_attrs(group_attrs, **svg_attrs)
    out = [f"<g {group_attrs_str}>\r"]

    # optional pattern fill
    use_pattern = bool(link)
    if use_pattern:
        out.append(
            "<defs>\r"
            "  <pattern id='img1' width='5' height='5'>\r"
            f"    <image href='{link}' x='0' y='0' width='45' height='45'/>\r"
            "  </pattern>\r"
            "</defs>\r"
        )

    # cells
    for i, pgon in enumerate(tiling):
        path_attrs = {}
        if use_pattern:
            path_attrs["fill"] = "url(#img1)"
        elif individual and colors_css is not None:
            path_attrs["fill"] = colors_css[i]
        
        d_attr = svg_cell_path(pgon, digits)
        path_attrs["d"] = d_attr
        
        path_attrs_str = build_svg_attrs(path_attrs)
        out.append(f"<path {path_attrs_str} />\r")

    out.append("</g>\r")
    return "".join(out)