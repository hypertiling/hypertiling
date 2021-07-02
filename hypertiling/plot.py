import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, PolyCollection


# taken from http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-polygons-in.html
# plots even very large samples of polygons in less than a second
def quick_plot(tiling, c='b', show_label=False, fs=5, save_img=False, path="", dpi=1200, refs=0):
    x, y = [], []
    for pgon in tiling.polygons:
        v = pgon.verticesP
        v = np.append(v, v[0])  # appending first vertex to close the circle to overcome missing edges in plot
        x.extend(v.real)
        x.append(None)  # this is some kind of trick that makes it that fast
        y.extend(v.imag)
        y.append(None)
        plt.text(pgon.centerP.real-0.015, pgon.centerP.imag-0.015, pgon.number, fontsize=fs) if show_label else None
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.axis('equal')
    plt.axis('off')
    plt.fill(x, y, facecolor='None', edgecolor=c, linewidth=.1)
    label = f"{{{tiling.p},{tiling.q}}}-{tiling.nlayers} tessellation," \
            f" {len(tiling.polygons)} polygons, {refs} refinement"
    label += "s" if refs != 1 else ""  # grammar
    plt.title(label)
    plt.savefig(path, dpi=1200) if save_img else None  # max dpi ca. 4000
    plt.show()


# simple plot function for hyperbolic tiling with colors
def plot_tiling(polygon_list, colors, symmetric_colors=False, plot_colorbar=True, xcrange=(-1,1), ycrange=(-1,1), **kwargs):   
    fig, ax = plt.subplots(figsize=(8,8), dpi=120)
    patches = []

    # loop over polygons
    for poly in polygon_list:
        # extract vertex coordinates
        u = poly.verticesP
        # transform to matplotlib Polygon format
        stack = np.column_stack((u.real,u.imag))
        polygon = Polygon(stack, True) 
        patches.append(polygon)

    # the polygon list has now become a PatchCollection
    pgons = PatchCollection(patches, **kwargs)

    # set colors and draw patches
    pgons.set_array(np.array(colors))
    ax.add_collection(pgons)
    
    # symmetric colorbar    
    if symmetric_colors:
        cmin = np.min(colors)
        cmax = np.max(colors)
        clim = np.maximum(-cmin,cmax)
        pgons.set_clim([-clim,clim])

    if plot_colorbar:
        plt.colorbar(pgons)

    plt.xlim(xcrange)
    plt.ylim(ycrange)
    plt.axis("off") 
    plt.gca().set_aspect('equal')

    return ax
