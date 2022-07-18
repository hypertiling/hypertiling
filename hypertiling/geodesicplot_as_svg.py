import numpy as np
import os
import geodesics
from core import HyperbolicTiling
from plot import quick_plot

p, q, n = 4, 8, 3
t = HyperbolicTiling(p, q, n, kernel="flo")
t.generate()
# quick_plot(t)


def init_svg(dim, file="myimg.svg"):
    os.remove(file) if os.path.exists(file) else None
    head = f"<svg width='{dim}px' height='{dim}px' viewBox='0 0 800 800' xmlns='http://www.w3.org/2000/svg'>" + "\r\n"
    svg = open(file, 'w')
    svg.write(head)
    return svg


def addgeo_svg(svg, x1, y1, x2, y2, r):
    start = "<path style='stroke:#000000; stroke-width:.5px; fill:transparent' "
    path = f"d = ' M {x1} {y1} A {r} {r} 0 0 0 {x2} {y2}'"
    end = "/>\r\n"
    svg.write(start + path + end)


def finalize_svg(svg):
    end = "</svg>"
    svg.write(end)
    svg.close()


def to_px(z):  # transforms complex number to px coordinates
    offset = 5
    x = np.real(z) + offset
    x *= 100
    y = np.imag(z) + offset
    y *= 100  # danach int()?
    return x, y


file = f"myimg {p, q}.svg"
mysvg = init_svg(800, file)
PI2 = 2*np.pi
c = 0

# note to self: this is slow and the svg turns out to be huge -> improve
vs = [_ for _ in range(1, t.p)] + [0]
for pgon in t:
    for v1, v2 in enumerate(vs):
        z1 = pgon.verticesP[v1]
        z2 = pgon.verticesP[v2]
        a1 = np.angle(z1) + PI2 if np.angle(z1) < 0 else np.angle(z1)
        a2 = np.angle(z2) + PI2 if np.angle(z2) < 0 else np.angle(z2)
        if a2 < a1:  # if second point is left of first point: flip values
            dummy = z1
            z1 = z2
            z2 = dummy
        if np.imag(z1)*np.imag(z2) < 0 < np.real(z1):  # for edges that intersect the x-axis: flip values
            dummy = z1
            z1 = z2
            z2 = dummy

        # calculate svg data
        arc = geodesics.geodesic_arc(z1, z2)

        try:  # this is rather impractical
            r = arc.get_width()/2  # = height
            q = r / abs(z2 - z1)  # scale factor between coordinates and pixels
        except AttributeError:  # happens when geodesic is a straight, radial line (r->\infty)
            c += 1
            q = 1e9  # some large number

        x1, y1 = to_px(z1)
        x2, y2 = to_px(z2)
        d = np.sqrt((x2-x1)**2+(y2-y1)**2)
        d_px = q*d
        addgeo_svg(mysvg, x1, y1, x2, y2, d_px)

finalize_svg(mysvg)
print(f"There were {c} geodesics with infinite curve radius.") if c != 0 else None
print("Image saved as '" + file + "'!")
