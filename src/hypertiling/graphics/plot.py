import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from ..geodesics import geodesic_arc
from matplotlib.colors import is_color_like


def quick_plot(tiling, unitcircle=False, dpi=150, **kwargs):
    """
    Fast plot function
    inspired by http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-polygons-in.html

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    unitcircle: boolean
        Determines whether the unit circle is plotted or not

    **kwargs
        Further properties (such as linewidth, alpha, facecolor, edgecolor, ...)

    Returns
    -------

    pgonpatches: PatchCollection
        Contains all the polygon patches.
    """

    # default kwargs
    if "fc" not in kwargs and "facecolor" not in kwargs:
        kwargs["fc"] = (1, 1, 1, 1)
    if "ec" not in kwargs and "edgecolor" not in kwargs:
        kwargs["ec"] = "k"

    # actual plot
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)

    # add bounding circle
    if unitcircle:
        circle = plt.Circle((0, 0), 1, **kwargs)
        ax.add_patch(circle)

    # draw polygons
    x, y = [], []
    for i in range(len(tiling)):
        v = tiling.get_vertices(i)
        v = np.append(v, v[0])  # appending first vertex to close the path
        x.extend(v.real)
        x.append(None)  # this is some kind of trick that makes it that fast
        y.extend(v.imag)
        y.append(None)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.axis('equal')
    plt.axis('off')
    plt.fill(x, y, **kwargs)
    plt.show()


def convert_polygons_to_patches(tiling, colors=None, cutoff=None, **kwargs):
    """
    Convert Hyperbolic Tiling cells into matplotlib PatchCollection
    Returns a PatchCollection, containing all polygons that are to be drawn.

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    colors: None or colorvalue or array-like
        Used for colormapping the PatchCollection. If None, all polygons are mapped with transparent faces;
        Other valid options are matplotlib color strings (e.g. "k", "white" or RGBA (0,0,1,1))
        or an array with the same length as the tiling, containing floats

    cutoff: float, default: None
        Actives lazy plotting; If set, only polygons which are located entirely inside the cutoff radius are being drawn

    **kwargs
        Further properties (such as linewidth, alpha, facecolor, edgecolor, ...)

    Returns
    -------

    pgonpatches: PatchCollection
        Contains all the polygon patches.
    """

    patches = []
    accepted_polys = []

    lazy = False
    if cutoff is not None:
        lazy = True
        cutoff = 1 - cutoff

    # iterate over tiling
    for poly in tiling:
        # extract vertex coordinates
        u = poly[1:]
        # lazy plotting
        if lazy:
            if np.all(np.abs(u) > cutoff):
                continue

        # transform to matplotlib Polygon format
        stack = np.column_stack((u.real, u.imag))
        polygon = Polygon(stack, True)
        patches.append(polygon)
        # accepted_polys.append(idx)

    # default values
    if colors is None:
        if "fc" not in kwargs and "facecolor" not in kwargs:
            kwargs["fc"] = (1, 1, 1, 1)
        if "ec" not in kwargs and "edgecolor" not in kwargs:
            kwargs["ec"] = "k"
        pgonpatches = PatchCollection(patches, **kwargs)

    # individual colors
    elif len(colors) == len(tiling):
        if "fc" in kwargs or "facecolor" in kwargs:
            print(
                "[hypertiling] Warning: Since an array of colors is provided, the facecolor argument (fc) is ignored.")

        # the polygon list has now become a PatchCollection
        pgonpatches = PatchCollection(patches, **kwargs)
        # add colors
        pgonpatches.set_array(np.array(colors))

    # identical colors
    elif is_color_like(colors):
        kwargs["fc"] = colors
        if "ec" not in kwargs and "edgecolor" not in kwargs:
            kwargs["ec"] = "k"
        if "cmap" in kwargs:
            print(
                "[hypertiling] Warning: Colormap argument (cmap) is being ignored, since only one static color is given.")

        # the polygon list has now become a PatchCollection
        pgonpatches = PatchCollection(patches, **kwargs)

    else:
        raise ValueError(
            "[hypertiling] Error: Argument 'colors' has no valid format. Must be matplotlib color type or array-like!")

    return pgonpatches


def convert_edges_to_arcs(tiling, cutoff=None, **kwargs):
    """
    Transform all edges of a tiling to either matplotlib Arc or Line2D object
    depending on whether they are straight lines or horocycles

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    cutoff: float, default: None
        Actives lazy plotting; If set, only polygons which are located entirely inside the cutoff radius are being drawn

    **kwargs
        Further properties (such as linewidth, alpha, edgecolor, ...)

    Returns
    -------

    edges: list of Arc and Line2D objects
        contains all edges in the lattice in a matplotlib-friendly format
    """

    edges = []
    types = []

    lazy = False
    if cutoff is not None:
        lazy = True
        cutoff = 1 - cutoff

    # iterate over cells
    for poly in tiling:
        # extract vertex coordinates
        u = poly[1:]
        nv = len(u)
        # lazy plotting
        if lazy:
            if np.all(np.abs(u) > cutoff):
                continue

        # loop over vertices/edges
        for i in range(nv):
            # extract edges
            z1 = u[i]
            z2 = u[(i + 1) % nv]
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


def plot_tiling(tiling, colors=None, unitcircle=False, symmetric_colors=False, plot_colorbar=False, cutoff=None,
                xcrange=(-1, 1),
                ycrange=(-1, 1), dpi=120, **kwargs):
    """
    Plots a hyperbolic tiling

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    colors: None or colorvalue or array-like
        Used for colormapping the PatchCollection. If None, all polygons are mapped with transparent faces;
        Other valid options are matplotlib color strings (e.g. "k", "white" or RGBA (0,0,1,1))
        or an array with the same length as the tiling, containing floats

    unitcircle: Bool, default: False
        If True, the unit circle (boundary of the Poincare disk) is added to the plot

    symmetric_colors: Bool, default: False
        If True, sets the colormap so that the center of the colormap corresponds to the center of colors.

    plot_colorbar: Bool, default: False
        If True, plots a colorbar.

    cutoff: float, default: None
        Actives lazy plotting; If set, only polygons which are located entirely inside the cutoff radius are being drawn

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
        Further properties (such as linewidth, alpha, facecolor, edgecolor, ...)

    """

    # create figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    # draw unit circle
    if unitcircle:
        circle = plt.Circle((0, 0), 1, lw=0.7, fc=(0, 0, 0, 0), ec="k")
        ax.add_patch(circle)

    # convert to matplotlib format
    pgons = convert_polygons_to_patches(tiling, colors, cutoff, **kwargs)

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


def plot_geodesic(tiling, color=None, unitcircle=False, cutoff=None, xcrange=(-1, 1), ycrange=(-1, 1), **kwargs):
    """
    Plots a hyperbolic tiling with geodesic edges
    Cells can not be filled!

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    color: color
        Sets the color of edges. This internally sets "fc" in kwargs.

    unitcircle: Bool, default: False
        If True, the unit circle (boundary of the Poincare disk) is added to the plot

    cutoff: float, default: None
        Actives lazy plotting; If set, only polygons which are located entirely inside the cutoff radius are being drawn

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
        Further properties (such as linewidth, alpha, facecolor, edgecolor, ...)

    """

    # create figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)

    # default values
    if color is not None:
        kwargs["ec"] = color
        kwargs.pop("edgecolor", None)
    else:
        if "ec" not in kwargs and "edgecolor" not in kwargs:
            kwargs["ec"] = "k"

    if "fc" in kwargs:
        del kwargs["fc"]
        print("[hypertiling] Warning: Setting a facecolor argument has no effect!")
    if "facecolor" in kwargs:
        del kwargs["fc"]
        print("[hypertiling] Warning: Setting a facecolor argument has no effect!")

    # draw unit circle
    if unitcircle:
        circle = plt.Circle((0, 0), 1, fc=(1, 1, 1, 0), **kwargs)
        ax.add_patch(circle)

    # transform
    edges, types = convert_edges_to_arcs(tiling, cutoff, **kwargs)

    # draw
    for edge in edges:
        ax.add_artist(edge)

    plt.xlim(xcrange)
    plt.ylim(ycrange)
    plt.axis("off")

    return ax
