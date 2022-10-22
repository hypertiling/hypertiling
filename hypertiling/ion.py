import csv
import os
import numpy as np
from .geodesics import geodesic_arc
import matplotlib.lines as mlines


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




def write_svg(fname, tiling, edgecolor="black", facecolor="transparent", lw=.5,  link=''):
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

    
    pi2 = 2 * np.pi
    vs = [_ for _ in range(1, tiling.p)] + [0]
    for pgon in tiling.polygons:
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
