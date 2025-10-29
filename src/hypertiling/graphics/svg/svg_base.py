import os
import numpy as np
from ...geodesics import geodesic_arc
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import SVG, display
from typing import Optional



class SvgBuilder:
    """Mutable SVG string builder with a friendly API."""
    def __init__(self): self.parts = []
    def write(self, s: str): self.parts.append(s)
    append = write  # alias
    def render(self) -> str: return "".join(self.parts)



def svg_open(
    center: tuple[float, float] = (100.0, 100.0),
    radius: float = 100.0,
    padding: float = 0.0,
    width: Optional[float] = None,
    height: Optional[float] = None,
    group_style: str = "",
) -> str:
    """
    Create an <svg> header with a viewBox centered at `center` and
    extending to `radius` (plus optional `padding`) on all sides.

    Parameters
    ----------
    center : (float, float)
        Pixel-space coordinates of the disk center.
    radius : float
        Pixel radius of the unit circle (|z|=1).
    padding : float
        Extra margin around the drawing in same units.
    width, height : float, optional
        Output size in px or other units; purely stylistic.
    group_style : str, optional
        CSS style string applied to the root <g> element.

    Returns
    -------
    str
        The opening <svg> and <g> tags.
    """
    cx, cy = center
    half = radius + padding
    x0, y0 = cx - half, cy - half
    w = h = 2 * half
    vb = f"{x0} {y0} {w} {h}"

    size_attr = []
    if width is not None:
        size_attr.append(f"width='{width}'")
    if height is not None:
        size_attr.append(f"height='{height}'")
    size_str = " ".join(size_attr)

    gs = f" style='{group_style}'" if group_style else ""
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{vb}' {size_str}>\n<g{gs}>\n"


def svg_close(
    unitcircle: bool = False,
    center: tuple[float, float] = (100.0, 100.0),
    radius: float = 99.9999,
    stroke: str = "black",
    stroke_width: float = 0.5,
    fill: str = "none",
    opacity: float = 1.0,
    dasharray: Optional[str] = None,
) -> str:
    """
    Return closing tags for the SVG document, optionally drawing the boundary
    (unit) circle with user-specified styling.

    Parameters
    ----------
    unitcircle : bool, optional
        If True, include a boundary circle before closing.
    center : (float, float), optional
        Pixel coordinates (cx, cy) of the disk center.
    radius : float, optional
        Pixel radius of the boundary circle.
    stroke : str, optional
        Stroke color of the boundary circle.
    stroke_width : float, optional
        Stroke width in pixels.
    fill : str, optional
        Fill color (typically "none").
    opacity : float, optional
        Overall opacity in [0, 1].
    dasharray : str, optional
        SVG stroke-dasharray pattern (e.g., "2,2" for dashed lines).

    Returns
    -------
    str
        Closing SVG markup including optional boundary circle.
    """
    parts = []
    if unitcircle:
        cx, cy = center
        dash = f" stroke-dasharray='{dasharray}'" if dasharray else ""
        parts.append(
            f"<circle cx='{cx}' cy='{cy}' r='{radius}' "
            f"fill='{fill}' stroke='{stroke}' stroke-width='{stroke_width}' "
            f"opacity='{opacity}'{dash} />\n"
        )
    parts.append("</g>\n</svg>\n")
    return "".join(parts)



def build_svg_attrs(base_attrs: dict, **svg_attrs) -> str:
    """
    Merge base attributes with additional SVG attributes.
    Converts underscores to hyphens (stroke_width -> stroke-width).
    svg_attrs override base_attrs.
    
    Example:
        build_svg_attrs({"fill": "red", "stroke": "black"}, opacity=0.5)
        # Returns: "fill='red' stroke='black' opacity='0.5'"
    """
    base_attrs.update({k.replace('_', '-'): v for k, v in svg_attrs.items()})
    return " ".join(f"{k}='{v}'" for k, v in base_attrs.items())


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





# def svg_close(unitcircle: bool = False) -> str:
#     """
#     Return closing tags for the current SVG document.

#     Parameters
#     ----------
#     unitcircle : bool, optional
#         If True, draws the boundary circle of the Poincaré disk before closing.

#     Returns
#     -------
#     str
#         Closing SVG markup (including optional unit circle and closing tags).
#     """
#     parts = []
#     if unitcircle:
#         parts.append("<circle cx='100' cy='100' r='99.9999' fill='none' />")
#     parts.append("</g>")
#     parts.append("</svg>")
#     return "\n".join(parts)




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
