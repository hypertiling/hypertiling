import os
import numpy as np
from ..geodesics import geodesic_arc
import matplotlib.lines as mlines
from matplotlib import cm
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




def make_svg(tiling, facecolors="white", edgecolor="black", lw=0.3, cmap="RdYlGn", digits=5, unitcircle=False, link=""):
    """
    Creates an scalable vector graphic (SVG) plot of the tiling

    Arguments:
    -----------
    tiling : HyperbolicTiling object
        An object containing the tiling.
    facecolor : str or array-like of length len(tiling)
        The background color of each polygon as a sequence of numbers
    edgecolor : string
        The color the polygon edges
    lw : float
        The linewidth the polygon edges
    cmap : str
        matplotlib colormap key string
    digits : int
        number of digits SVG coordinates are rounded to
    unitcircle: bool
        whether or not the unit circle is added to the plot
    """

    # preparations
    svg = svgString()
    pi2 = 2 * np.pi
    vs = [_ for _ in range(1, tiling.p)] + [0]

    # one color vs. colormap
    individual_colors = True
    if isinstance(facecolors, str):
        individual_colors = False
    else:
        ccmap = cm.get_cmap(f"{cmap}")
        colors = array_to_rgb(norm_0_1(facecolors), ccmap)

    # attribute group
    if individual_colors:
        group_open = f"<g style='stroke:{edgecolor}; stroke-width:{lw}px'>\r"
    else:
        group_open = f"<g style='stroke:{edgecolor}; stroke-width:{lw}px; fill:{facecolors}'>\r"
    svg.write(group_open)

    if link != '':  
        pattern = f"<defs>\r <pattern id='img1' width='5' height='5'>\r" \
                  f"  <image href='{link}' x='0' y='0' width='45' height='45'/>\r </pattern>\r</defs>"
        svg.write(pattern + "\r\n")
        facecolors = 'transparent'


    # loop through tiling
    for idx, pgon in enumerate(tiling):
        if individual_colors:
            start = f"\t<path   style='fill:rgb{colors[idx,0], colors[idx,1], colors[idx,2]}' "
        else:
            start = f"\t<path  "
        svg.write(start + "\r")
        z0 = np.conj(tiling.get_vertices(idx)[0])
        x0, y0 = to_px(z0)
        path = f"       d = 'M {np.round(x0,digits)} {np.round(y0,digits)} "
        for v1, v2 in enumerate(vs):
            z1 = np.conj(tiling.get_vertices(idx)[v1])
            z2 = np.conj(tiling.get_vertices(idx)[v2])
            orientation = False
            a1 = np.angle(z1) + pi2 if np.angle(z1) < 0 else np.angle(z1)
            a2 = np.angle(z2) + pi2 if np.angle(z2) < 0 else np.angle(z2)

            # if second point is left of first point: swap values
            if a2 < a1:  
                orientation = np.invert(orientation)
            # for edges that intersect the x-axis: swap values
            if np.imag(z1) * np.imag(z2) < 0 < np.real(z1):  
                orientation = np.invert(orientation)

            # calculate svg data
            arc = geodesic_arc(z1, z2)
            if type(arc) == mlines.Line2D:  # if r -> \infty
                r = 1e9  # some large number
            else:
                r = arc.get_width() / 2  # = height

            q = r / abs(z2 - z1)  # scale factor between coordinates and pixels
            x1, y1 = to_px(z1)
            x2, y2 = to_px(z2)
            r_px = q * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            path += f" A {np.round(r_px,digits)} {np.round(r_px,digits)} 0 0 {int(orientation)} {np.round(x2,digits)} {np.round(y2,digits)} "
        path += "'\r        fill = 'url(#img1)'/>" if link != '' else "'/>\r"
        svg.write(path + "\r\n")
    if unitcircle:
        svg.write('<circle cx="100" cy="100" r="99.9999" fill="none" />')
    svg.write("</g>")
    svg.write("\r</svg>")
    return svg.print()


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
