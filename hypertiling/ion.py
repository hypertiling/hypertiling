import csv
import os
import numpy as np
from .geodesics import geodesic_arc
import matplotlib.lines as mlines
from matplotlib import cm


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




def write_svg(fname, tiling, edgecolor="black", facecolor="transparent", lw=.5,  link='', cmap=None):
    """
        Saves a plot of the geodesic edges as a .svg-file.

        Arguments:
        -----------
        fname : string
            Output file name including directory
        tiling : HyperbolicTiling object
            An object containing the tiling.
        edgecolor : string
            The color of each geodesic.
        facecolor : string
            The background color of each polygon
        lw : float
            The line width of each geodesic.
        link : string
            A hyperlink referencing an image to fill each polygon with.


        TODO: This is slow and the svg turns out to be huge, needs improvement
    """

    # preparations
    os.remove(fname) if os.path.exists(fname) else None
    head = f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' " \
           f"width='500px' height='500px' viewBox='0 0 200 200'>" + "\r\n"
    svg = open(fname, 'w')
    svg.write(head)

    # if background image is provided
    if link != '':  
        pattern = f"<defs>\r <pattern id='img1' width='5' height='5'>\r" \
                  f"  <image href='{link}' " \
                  "x='0' y='0' width='45' height='45'/>\r </pattern>\r</defs>"
        svg.write(pattern + "\r\n")
        if facecolor == 'transparent':
            facecolor = ''
            fill_individual = False
    if hasattr(facecolor, "__len__") and len(facecolor) == len(tiling):
        fill_individual = True
        if not cmap:
            cmap = cm.get_cmap("RdYlGn")
        else:
            cmap = cm.get_cmap(f"{cmap}")
        colors = array_to_rgb(norm_0_1(facecolor), cmap)
    else:
        print("facecolor must be either an SVG fill command or an array like of size len(tiling) "
              "containing ints or floats")
        return

    pi2 = 2 * np.pi
    vs = [_ for _ in range(1, tiling.p)] + [0]
    for idx, pgon in enumerate(tiling.polygons):
        if fill_individual:
            start = f"   <path style='stroke:{edgecolor}; stroke-width:{lw}px; " \
                    f"fill:rgb{colors[idx,0], colors[idx,1], colors[idx,2]}' "
            print(start)
        else:
            start = f"   <path style='stroke:{edgecolor}; stroke-width:{lw}px; fill:{facecolor}' "
        svg.write(start + "\r")
        z0 = pgon.verticesP[0]
        x0, y0 = to_px(z0)
        path = f"       d = 'M {x0} {y0} "
        for v1, v2 in enumerate(vs):
            z1 = pgon.verticesP[v1]
            z2 = pgon.verticesP[v2]
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
            path += f" A {r_px} {r_px} 0 0 {int(orientation)} {x2} {y2} "
        path += "'\r        fill = 'url(#img1)'/>" if link != '' else "'/>\r"
        svg.write(path + "\r\n")

    svg.write("\r</svg>")
    svg.close()
    print("Image saved as '" + fname + "'!")


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


def write_csv(fname, nbrs):
    """
        Saves the neighbour list into a CSV table file

        Arguments:
        -----------
        fname : str
            Output file name including directory
        nbrs : List[List[int]]:
            Neighbors list
    """
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(nbrs)
