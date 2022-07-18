import matplotlib.lines
import numpy as np
import os
import geodesics


def to_px(z):  # transforms complex number to px coordinates
    offset = 5
    x = np.real(z) + offset
    x *= 100
    y = np.imag(z) + offset
    y *= 100  # danach int()?
    return x, y


def addgeo_svg(svg, x1, y1, x2, y2, r):
    start = "<path style='stroke:#000000; stroke-width:.5px; fill:transparent' "
    path = f"d = ' M {x1} {y1} A {r} {r} 0 0 0 {x2} {y2}'"
    svg.write(start + path + "/>\r\n")


def save_as_svg(t, sz=500, filename=f"geodesicplot.svg"):
    os.remove(filename) if os.path.exists(filename) else None
    head = f"<svg width='{sz}px' height='{sz}px' viewBox='0 0 800 800' xmlns='http://www.w3.org/2000/svg'>" + "\r\n"
    svg = open(filename, 'w')
    svg.write(head)

    pi2 = 2 * np.pi
    c = 0

    # note to self: this is slow and the svg turns out to be huge -> improve
    vs = [_ for _ in range(1, t.p)] + [0]
    for pgon in t:
        for v1, v2 in enumerate(vs):
            z1 = pgon.verticesP[v1]
            z2 = pgon.verticesP[v2]
            a1 = np.angle(z1) + pi2 if np.angle(z1) < 0 else np.angle(z1)
            a2 = np.angle(z2) + pi2 if np.angle(z2) < 0 else np.angle(z2)
            if a2 < a1:  # if second point is left of first point: swap values
                z1, z2 = z2, z1
            if np.imag(z1) * np.imag(z2) < 0 < np.real(z1):  # for edges that intersect the x-axis: swap values
                z1, z2 = z2, z1

            # calculate svg data
            arc = geodesics.geodesic_arc(z1, z2)
            if type(arc) == matplotlib.lines.Line2D:  # if r -> \infty
                c += 1
                q = 1e9  # some large number
            else:
                r = arc.get_width() / 2  # = height
                q = r / abs(z2 - z1)  # scale factor between coordinates and pixels

            x1, y1 = to_px(z1)
            x2, y2 = to_px(z2)
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            d_px = q * d
            addgeo_svg(svg, x1, y1, x2, y2, d_px)

    svg.write("</svg>")
    svg.close()
    print("Image saved as '" + filename + "'!")
