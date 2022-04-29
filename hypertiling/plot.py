import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, PolyCollection
from .geodesics import geodesic_arc


# taken from http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-polygons-in.html
# plots even very large samples of polygons in less than a second
def quick_plot(tiling, c='b', show_label=False, fs=5, save_img=False, path="", dpi=1200, refs=0):
    x, y = [], []
    for pgon in tiling.polygons:
        v = pgon.verticesP[0:-1]
        v = np.append(v, v[0])  # appending first vertex to close the circle to overcome missing edges in plot
        x.extend(v.real)
        x.append(None)  # this is some kind of trick that makes it that fast
        y.extend(v.imag)
        y.append(None)
        plt.text(pgon.centerP().real-0.015, pgon.centerP().imag-0.015, pgon.number, fontsize=fs) if show_label else None
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



# convert list of HyperPolygon objects to matplotlib PatchCollection
def poly2patch(polygons, colors=None, **kwargs):
    patches = []

    # loop over polygons
    for poly in polygons:
        # extract vertex coordinates
        u = poly.verticesP[0:-1]
        # transform to matplotlib Polygon format
        stack = np.column_stack((u.real,u.imag))
        polygon = Polygon(stack, True) 
        patches.append(polygon)

    # the polygon list has now become a PatchCollection
    pgonpatches = PatchCollection(patches, **kwargs)
    # add colors
    if colors is not None:
        pgonpatches.set_array(np.array(colors))

    return pgonpatches    


# transform all edges in the tiling to either matplotlib Arc or Line2D
# depending on whether they came out straight or curved
# the respective type is encoded in the array "types"
def edges2matplotlib(T, **kwargs):
    
    edges = []
    types = []

    for poly in T: # loop over polygons
        for i in range(T.p): # loop over vertices
            z1 = poly.verticesP[i] # extract edges
            z2 = poly.verticesP[(i+1)%T.p]
            edge = geodesic_arc(z1,z2,**kwargs) # compute arc
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
def plot_tiling(polygons, colors, symmetric_colors=False, plot_colorbar=False, xcrange=(-1,1), ycrange=(-1,1), **kwargs):   
    fig, ax = plt.subplots(figsize=(10,7), dpi=120)

    # convert to matplotlib format
    pgons = poly2patch(polygons, colors, **kwargs)

    # draw patches
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
