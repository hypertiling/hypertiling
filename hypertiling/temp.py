import numpy as np

import geodesics
from core import HyperbolicTiling
from plot import quick_plot
import matplotlib.pyplot as plt

t = HyperbolicTiling(4, 8, 2, kernel="manu")
t.generate()
# quick_plot(t)

z1 = t.polygons[0].verticesP[1]  # first Bezier point
z2 = t.polygons[0].verticesP[2]  # second
zm_geo = geodesics.geodesic_midpoint(z1, z2)
zm_avg = (z1 + z2)/2  # un-geodesic midpoint
n = zm_geo - zm_avg
z3 = zm_geo + n  # third Bezier point
arc = geodesics.geodesic_arc(z1, z2)
fg, ax = plt.subplots()
ax.add_patch(arc)
ax.plot(np.real(z1), np.imag(z1), 'ro')
ax.plot(np.real(z2), np.imag(z2), 'ro')
ax.plot(np.real(z3), np.imag(z3), 'ro')
ax.axis([-1, 1, -1, 1])
ax.set_aspect("equal")
plt.show()
unity = 1+1j
print(z1+unity, z2+unity, z3+unity)

# calculate svg data
x, y = arc.get_center()
x += 1
x = int(400*x)
y += 1
y = int(400*y)
r = arc.get_width()/2  # = height
th1, th2 = arc.theta1, arc.theta2
q = r/abs(z2-z1)
print(q)
# print(x, y, w)