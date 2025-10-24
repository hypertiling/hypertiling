import os
import numpy as np
from ..geodesics import geodesic_arc
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import SVG, display


def to_px(z, factor=100, offset=1):
    """
    Transforms complex number to px coordinates

    Arguments:
    ----------
    z : np.complex
        coordinate in the complex plane
    factor : int
        some large scaling factor to conform to px scale
    offset : int
        offset plot region
        
    """
    x = np.real(z) + offset
    x *= factor
    y = np.imag(z) + offset
    y *= factor
    return x, y


class svgString():
    """
    Helper class, makes working with strings more convenient
    
    """

    def __init__(self):
        header = f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' " \
                 f"width='500px' height='500px' viewBox='0 0 200 200'>" + "\r\n"
        self.string = header

    def write(self, string):
        self.string += string

    def newline(self):
        self.string += "\n"

    def tabstop(self):
        self.string += "\t"

    def print(self):
        return self.string


from typing import Iterable, Sequence, Optional, Tuple, Union

ColorLike = Union[str, Sequence[float], Sequence[Tuple[float, float, float]]]

def make_svg(
    tiling,
    facecolors: ColorLike = "white",      # str | list of scalars | list of (r,g,b)
    edgecolor: str = "black",
    lw: float = 0.3,
    cmap: str = "RdYlGn",
    digits: int = 5,
    unitcircle: bool = False,
    link: str = "",
):
    """
    Render a hyperbolic tiling (Poincaré disk representation) as an SVG image.

    Each cell of the tiling is drawn as an SVG `<path>` composed of geodesic arcs,
    which appear as circle segments orthogonal to the unit circle or as diameters.
    Arcs are oriented correctly for all layers by determining the sweep direction
    in the undistorted disk coordinates rather than pixel space.

    Parameters
    ----------
    tiling : HyperbolicTiling
        Iterable of polygons representing the tiling. Each polygon must provide
        its vertices as complex numbers inside the unit disk (pgon[1:] are vertices).
    facecolors : str | array-like
        Color specification for polygon interiors.
        • Single CSS color string → same fill for all polygons.
        • Sequence of scalar values → normalized and mapped through `cmap`.
        • Sequence of RGB tuples (in [0,1] or [0,255]) → used directly per cell.
    edgecolor : str, optional
        Color of polygon edges (stroke).
    lw : float, optional
        Line width in pixels for polygon edges.
    cmap : str, optional
        Matplotlib colormap key used when `facecolors` is a sequence of scalars.
    digits : int, optional
        Number of decimal digits used when writing SVG coordinates.
    unitcircle : bool, optional
        If True, draw a circle marking the Poincaré boundary (radius = 1).
    link : str, optional
        Path or URL to an image used as a repeating pattern fill (`url(#img1)`).

    Returns
    -------
    str
        The SVG markup as a string.

    Notes
    -----
    - Arc sweep directions are computed geometrically from the circle center of
    each geodesic, ensuring correct curvature even near the disk boundary.
    """


    # ---- helpers -------------------------------------------------------------

    def _is_scalar_sequence(x) -> bool:
        try:
            x = np.asarray(x)
            return x.ndim == 1 and x.size == len(tiling)
        except Exception:
            return False

    def _is_rgb_sequence(x) -> bool:
        try:
            a = np.asarray(x, dtype=float)
            return a.ndim == 2 and a.shape[1] == 3 and a.shape[0] == len(tiling)
        except Exception:
            return False

    def _rgb_to_css(rgb: Sequence[float]) -> str:
        # accept [0..1] or [0..255]
        arr = np.asarray(rgb, dtype=float)
        if arr.max() <= 1.0:  # assume [0..1]
            arr = np.round(arr * 255.0)
        return f"rgb({int(arr[0])},{int(arr[1])},{int(arr[2])})"

    def _resolve_facecolors(facecolors: ColorLike, cmap: str) -> Tuple[bool, Optional[Sequence[str]]]:
        """
        Returns:
          (individual, colors_css)
          - individual=False means a single fill color (handled at group level).
          - individual=True and colors_css is a list of 'rgb(r,g,b)' strings per polygon.
        """
        # single CSS color string — group level fill
        if isinstance(facecolors, str):
            return False, None

        # list of RGBs
        if _is_rgb_sequence(facecolors):
            cols = [_rgb_to_css(rgb) for rgb in facecolors]  # per polygon
            return True, cols

        # list of scalars -> cmap
        if _is_scalar_sequence(facecolors):
            vals = np.asarray(facecolors, dtype=float)
            # normalize to [0,1] safely
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax > vmin:
                vals = (vals - vmin) / (vmax - vmin)
            else:
                vals = np.zeros_like(vals)
            ccmap = plt.get_cmap(cmap)
            rgba = ccmap(vals)[:, :3]  # Nx3 in [0,1]
            cols = [_rgb_to_css(255 * c) for c in rgba]
            return True, cols

        # fallback: treat as a single color string via str()
        return False, None

    def _arc_segment(z1: complex, z2: complex, digits: int) -> str:
        """
        Build the SVG path segment ('L ...' or 'A ...') for a hyperbolic edge.
        - Decides CW/CCW in *disk coords* using the geodesic circle center.
        - Forces the minor arc (large-arc-flag=0) for proper geodesics.
        """
        x1, y1 = to_px(z1)
        x2, y2 = to_px(z2)
        arc = geodesic_arc(z1, z2)

        if isinstance(arc, mlines.Line2D):
            # diameter geodesic
            return f" L {np.round(x2, digits)} {np.round(y2, digits)} "

        # circle arc: center in disk coords
        try:
            cx_d, cy_d = arc.get_center()
        except AttributeError:
            cx_d, cy_d = arc.center  # mpl fallback
        c = complex(cx_d, cy_d)

        # angles around the center in disk coords (math y-up)
        th1 = np.arctan2((z1 - c).imag, (z1 - c).real)
        th2 = np.arctan2((z2 - c).imag, (z2 - c).real)
        dth = (th2 - th1) % (2 * np.pi)

        large_arc_flag = 0                 # always take the minor arc
        sweep_flag = int(dth <= np.pi)     # CCW in math coords

        # radius in pixels from pixel-center to pixel-endpoint
        cx_px, cy_px = to_px(c)
        r_px = np.hypot(x1 - cx_px, y1 - cy_px)

        return (
            f" A {np.round(r_px, digits)} {np.round(r_px, digits)} 0 "
            f"{large_arc_flag} {sweep_flag} {np.round(x2, digits)} {np.round(y2, digits)} "
        )

    def _polygon_path(pgon: Sequence[complex], digits: int) -> str:
        """
        Build the 'd' attribute for a polygon path from its vertex list (complex in disk coords).
        Assumes pgon[1:] are the vertices in order
        Applies conjugation to flip y for SVG screen coords.
        """
        verts = [np.conj(v) for v in pgon[1:]] 
        x0, y0 = to_px(verts[0])
        d = [f"M {np.round(x0, digits)} {np.round(y0, digits)}"]
        for i in range(len(verts)):
            z1 = verts[i]
            z2 = verts[(i + 1) % len(verts)]
            d.append(_arc_segment(z1, z2, digits))
        return " ".join(d)

    # ---- start building SVG --------------------------------------------------

    svg = svgString()

    individual, colors_css = _resolve_facecolors(facecolors, cmap)

    # group style; put stroke on the group, fill either on group (single) or per-path (individual)
    if not individual:
        group_fill = facecolors if isinstance(facecolors, str) else "white"
        group_open = f"<g style='stroke:{edgecolor}; stroke-width:{lw}px; fill:{group_fill}'>\r"
    else:
        group_open = f"<g style='stroke:{edgecolor}; stroke-width:{lw}px'>\r"

    svg.write(group_open)

    # optional pattern fill (overrides facecolor per path)
    use_pattern = bool(link)
    if use_pattern:
        pattern = (
            "<defs>\r"
            "  <pattern id='img1' width='5' height='5'>\r"
            f"    <image href='{link}' x='0' y='0' width='45' height='45'/>\r"
            "  </pattern>\r"
            "</defs>\r"
        )
        svg.write(pattern)

    # draw cells
    for idx, pgon in enumerate(tiling):
        attrs = ["\t<path"]

        # per-path fill
        if use_pattern:
            attrs.append("fill='url(#img1)'")
        elif individual and colors_css is not None:
            attrs.append(f"style='fill:{colors_css[idx]}'")

        # path data
        d_attr = _polygon_path(pgon, digits)
        attrs.append(f"d='{d_attr}'")
        attrs.append("/>\r")

        svg.write(" ".join(attrs))

    # unit circle (optional)
    #if unitcircle:
    #    svg.write("<circle cx='100' cy='100' r='99.9999' fill='none' />")

    #svg.write("</g>\r</svg>")
    return svg.print()


def svg_close(unitcircle: bool = False) -> str:
    """
    Return closing tags for the current SVG document.

    Parameters
    ----------
    unitcircle : bool, optional
        If True, draws the boundary circle of the Poincaré disk before closing.

    Returns
    -------
    str
        Closing SVG markup (including optional unit circle and closing tags).
    """
    parts = []
    if unitcircle:
        parts.append("<circle cx='100' cy='100' r='99.9999' fill='none' />")
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def svg_hyperbolic_circle(
    z0: complex,
    R: float,
    edgecolor: str = "black",
    fill: str = "none",
    lw: float = 1.0,
    digits: int = 5,
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

    Returns
    -------
    str
        An SVG <circle> element string with cx, cy, r, stroke, fill, and stroke-width.
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

    # build SVG element
    return (
        f"<circle cx='{np.round(cx, digits)}' "
        f"cy='{np.round(cy, digits)}' "
        f"r='{np.round(r_px, digits)}' "
        f"fill='{fill}' stroke='{edgecolor}' stroke-width='{lw}' />"
    )


def svg_horocycle(
    z1: complex,
    R: float,
    edgecolor: str = "black",
    fill: str = "none",
    lw: float = 1.0,
    digits: int = 5,
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

    Returns
    -------
    str
        SVG <circle> element string with cx, cy, r, stroke, fill, and stroke-width.

    """

    # Euclidean parameters for the horocycle
    r_e = 0.5 * (1 / np.cosh(R / 2.0)) ** 2  # = 0.5 * sech^2(R/2)
    c_e = (1 - r_e) * z1

    # map to pixel space (with y-flip via conjugation)
    cx, cy = to_px(np.conj(c_e))
    px, py = to_px(np.conj(c_e + r_e))
    r_px = np.hypot(px - cx, py - cy)

    # build SVG element
    return (
        f"<circle cx='{np.round(cx, digits)}' "
        f"cy='{np.round(cy, digits)}' "
        f"r='{np.round(r_px, digits)}' "
        f"fill='{fill}' stroke='{edgecolor}' stroke-width='{lw}' />"
    )





def svg_hypercycle(
    phi: float,
    R: float,
    side: int = +1,
    edgecolor: str = "black",
    fill: str = "none",
    lw: float = 1.0,
    digits: int = 5,
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

    Returns
    -------
    str
        SVG <circle> element string (cx, cy, r, stroke, fill, stroke-width).

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

    return (
        f"<circle cx='{np.round(cx, digits)}' "
        f"cy='{np.round(cy, digits)}' "
        f"r='{np.round(r_px, digits)}' "
        f"fill='{fill}' stroke='{edgecolor}' stroke-width='{lw}' />"
    )





def draw_svg(content: str):
    """
    Use IPython display API for displaying SVG
    """
    display(SVG(content))


def write_svg(fname: str, content: svgString):
    """
    Write svgString to file
    """
    os.remove(fname) if os.path.exists(fname) else None
    svgfile = open(fname, 'w')
    svgfile.write(content)
    svgfile.close()


def norm_0_1(x, cmin=None, cmax=None):
    """
    Normalize an array like x linearly between 0 and 1

    Arguments:
    __________
    x : 1d array like
        contains data to be normalized between 0 and 1
    cmin : float, default = None
        the value that is mapped to 0
        if None, the minimal value of x is taken
    cmax : float, default = None
        the value that is mapped to 1
        if None, the maximal value of x is taken

    """
    if not cmin:
        cmin = min(x)
    else:
        cmin = cmin
    if not cmax:
        cmax = max(x)
    else:
        cmax = cmax
    x = np.array(x)
    return (x - cmin) / (cmax - cmin)


def array_to_rgb(x, cmap):
    """
    Takes an array like in the range of [0,1] and return a 2d array containing the rgb values in the range [0, 255]
    in respect to cmap

    Arguments:
    __________
    x : 1d array like
        contains data in the range [0,1] to be mapped to rgb values
    cmap :  matplotlib.colors.LinearSegmentedColormap
        the colormap that is used to calculate the rgb values

    """
    rgb = np.zeros((len(x), 3))
    for idx, val in enumerate(x):
        rgb[idx] = cmap(val)[:3]
    return (rgb * 255).astype(int)
