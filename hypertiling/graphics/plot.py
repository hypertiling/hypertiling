import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, PolyCollection
from ..geodesics import geodesic_arc


# taken from http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-polygons-in.html
# plots even very large samples of polygons in less than a second
def quick_plot(tiling, edgecolor='k', show_title=False, lw=0.3, save_img=False, path="", dpi=150):
    fig, ax = plt.subplots(figsize=(8,7), dpi=dpi)
    x, y = [], []
    for i, pgon in enumerate(tiling):
        v = tiling.get_vertices(i)
        v = np.append(v, v[0])  # appending first vertex to close the circle to overcome missing edges in plot
        x.extend(v.real)
        x.append(None)  # this is some kind of trick that makes it that fast
        y.extend(v.imag)
        y.append(None)
        w = tiling.get_center(i)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.axis('equal')
    plt.axis('off')
    plt.fill(x, y, facecolor='None', edgecolor=edgecolor, linewidth=lw)
    if show_title:
        label = f"{{{tiling.p},{tiling.q}}}-{tiling.nlayers} tessellation," \
            f" {len(tiling)} polygons"
        plt.title(label)
    plt.savefig(path, dpi=dpi) if save_img else None  # max dpi ca. 4000
    plt.show()


# convert Hyperbolic Tiling cells into matplotlib PatchCollection
def convert_polygons_to_patches(tiling, colors=None, lazy=False, cutoff=0.001, **kwargs):
    """
    Returns a PatchCollection, containing all polygons that are to be drawn.

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    colors: array-like
        Used for colormapping the PatchCollection. Must have same length as polygons.

    lazy: Bool, default: False
        If True, only polygons whose edges are all longer than the parameter cutoff will be added to the PatchCollection.

    cutoff: float, default: 0.001
        Only active, if lazy is True. Sets the minimal edge length for lazy plotting.

    Returns
    -------

    pgonpatches: PatchCollection
        Contains all the polygon patches.

    Other Parameters:
    -----------------

    **kwargs
        Patch properties.

    """
    patches = []
    accepted_polys = []

    # loop over polygons
    for idx in range(len(tiling)):
        # extract vertex coordinates
        u = tiling.get_vertices(idx)
        # lazy plotting
        if lazy and np.any(abs(np.diff(u)) < cutoff):
            continue
        # transform to matplotlib Polygon format
        stack = np.column_stack((u.real, u.imag))
        polygon = Polygon(stack, True)
        patches.append(polygon)
        accepted_polys.append(idx)

    # the polygon list has now become a PatchCollection
    pgonpatches = PatchCollection(patches, **kwargs)
    # add colors
    if colors is not None:
        pgonpatches.set_array(np.array(colors)[accepted_polys])

    return pgonpatches


# transform all edges in the tiling to either matplotlib Arc or Line2D
# depending on whether they came out straight or curved
# the respective type is encoded in the array "types"
def convert_edges_to_arcs(tiling, **kwargs):
    edges = []
    types = []

    for j, poly in enumerate(tiling):  # loop over polygons
        for i in range(tiling.p):  # loop over vertices
            z = tiling.get_vertices(j)
            z1 = z[i]  # extract edges
            z2 = z[(i + 1) % tiling.p]
            edge = geodesic_arc(z1, z2, **kwargs)  # compute arc
            edges.append(edge)

            edgetype = type(edge).__name__

            if edgetype == "Line2D":
                types.append(1)
            elif edgetype == "Arc":
                types.append(0)
            else:
                types.append(-1)

    return edges, types


# simple plot function for hyperbolic tiling with colors
def plot_tiling(tiling, colors, symmetric_colors=False, plot_colorbar=False, lazy=False, cutoff=0.001, xcrange=(-1, 1),
                ycrange=(-1, 1), **kwargs):
    """
    Plots a hyperbolic tiling.

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    colors: array-like
        Used for colormapping the PatchCollection. Must have same length as polygons.

    symmetric_colors: Bool, default: False
        If True, sets the colormap so that the center of the colormap corresponds to the center of colors.

    plot_colorbar: Bool, default: False
        If True, plots a colorbar.

    lazy: Bool, default: False
        If True, only polygons whose edges are all longer than the parameter cutoff will be added to the PatchCollection.

    cutoff: float, default: 0.001
        Only active, if lazy is True. Sets the minimal edge length for lazy plotting.

    xcrange: (2,) array-like, default: (-1,1)
        Sets the x limits of the plot.

    ycrange: (2,) array-like, default: (-1,1)
        Sets the y limits of the plot.

    Returns
    -------

    out: Axes
        Axes object containing the hyperbolic tiling plot.

    Other Parameters:
    -----------------

    **kwargs
        Patch properties.

    """

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

    # convert to matplotlib format
    pgons = convert_polygons_to_patches(tiling, colors, lazy, cutoff, **kwargs)

    # draw patches
    ax.add_collection(pgons)

    # symmetric colorbar    
    if symmetric_colors:
        cmin = np.min(colors)
        cmax = np.max(colors)
        clim = np.maximum(-cmin, cmax)
        pgons.set_clim([-clim, clim])

    if plot_colorbar:
        plt.colorbar(pgons)

    plt.xlim(xcrange)
    plt.ylim(ycrange)
    plt.axis("off")

    return ax


# simple plot function for hyperbolic tiling with geodesic edges
def plot_geodesic(tiling, color="k", xcrange=(-1, 1), ycrange=(-1, 1), **kwargs):
    """
    Plots a hyperbolic tiling.

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    color: color
        Sets the color of edges.

    xcrange: (2,) array-like, default: (-1,1)
        Sets the x limits of the plot.

    ycrange: (2,) array-like, default: (-1,1)
        Sets the y limits of the plot.

    Returns
    -------

    out: Axes
        Axes object containing the hyperbolic tiling plot.

    Other Parameters:
    -----------------

    **kwargs
        Patch properties.

    """

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

    edges, types = convert_edges_to_arcs(tiling, **kwargs)
    for edge in edges:
        ax.add_artist(edge)
        edge.set_color(color)

    plt.xlim(xcrange)
    plt.ylim(ycrange)
    plt.axis("off")

    return ax
