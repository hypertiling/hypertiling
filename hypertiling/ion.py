import xlsxwriter
import csv
import os
import numpy as np
from .geodesics import geodesic_arc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def to_px(z):  # transforms complex number to px coordinates
    offset = 1  # makes all coords positive
    x = np.real(z) + offset
    x *= 100  # some large scaling factor to conform to px scale
    y = np.imag(z) + offset
    y *= 100
    return x, y




def write_svg_graph(t, filename="geodesicplot.svg",  color="transparent", lw=.5, edgecolor="black", link=''):
    """
        Saves a plot of the geodesic edges as a .svg-file.

        Arguments:
        -----------
        t : HyperbolicTiling object
            An object containing the tiling.
        filename : string
            The name the file is saved as.
        color : string
            The background color of each polygon
        lw : float
            The line width of each geodesic.
        edgecolor : string
            The color of each geodesic.
        link : string
            A hyperlink referencing an image to fill each polygon with.
    """

    os.remove(filename) if os.path.exists(filename) else None
    head = f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' " \
           f"width='500px' height='500px' viewBox='0 0 200 200'>" + "\r\n"
    svg = open(filename, 'w')
    svg.write(head)

    if link != '':  # if background image is provided
        pattern = f"<defs>\r <pattern id='img1' width='5' height='5'>\r" \
                  f"  <image href='{link}' " \
                  "x='0' y='0' width='45' height='45'/>\r </pattern>\r</defs>"
        svg.write(pattern + "\r\n")
        if color == 'transparent':
            color = ''

    # note to self: this is slow and the svg turns out to be huge -> improve
    pi2 = 2 * np.pi
    vs = [_ for _ in range(1, t.p)] + [0]
    for pgon in t.polygons:
        start = f"   <path style='stroke:{edgecolor}; stroke-width:{lw}px; fill:{color}' "
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
            if a2 < a1:  # if second point is left of first point: swap values
                orientation = np.invert(orientation)
            if np.imag(z1) * np.imag(z2) < 0 < np.real(z1):  # for edges that intersect the x-axis: swap values
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
    print("Image saved as '" + filename + "'!")




# write list of lists to csv
def write_csv(nn_list, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(nn_list)


def write_excel(data, path, t):
    workbook = xlsxwriter.Workbook(path)  # writing to .csv instead might be faster
    ws = workbook.add_worksheet()

    # Bring border sites to the top of the matrix and sort each part by polygon number
    # pos = 0
    # for ind, row in enumerate(data):  # border sites are sorted by construction
    #     if np.count_nonzero(row == 0) != 0:
    #         data[[pos, ind]] = data[[ind, pos]]  # swaps the rows at index ind and pos
    #         pos += 1
    #
    # dataslice = data[pos:, :]
    # dataslice = sorted(dataslice, key=lambda dataslice_entry: dataslice_entry[0])  # sorting not-border sites
    # data[pos:, :] = np.stack(dataslice, axis=0)  # sorts the non-border sites

    # Header/first line of the table
    if t == "nn":
        ws.write(0, 0, "Polygon #")  # first column
        ws.write(0, 1, f"x_center")
        ws.write(0, 2, f"y_center")
        for x in range(1, data.shape[1]-2):
            ws.write(0, x+2, f"NN #{x}")

    # write array to table
    #cblue = workbook.add_format({'bg_color': '#8EA9DB'})  # blue for border sites NN < p
    #corange = workbook.add_format({'bg_color': '#F4B084'})  # orange for inner sites NN = p
    for i in range(len(data)):
        for j in range(data.shape[1]):
            if t == "nn":
                ws.write(i + 1, j, data[i, j])
            elif t == "adj":
                ws.write(i, j, data[i, j])
            # if i <= pos - 1:
            #     ws.write(i + 1, j, data[i, j], cblue)
            # else:
            #     ws.write(i + 1, j, data[i, j], corange)

    workbook.close()  # necessary for saving the changes