from matplotlib import animation
from hypertiling.plot import poly2patch
from hypertiling.transformation import mymoeb, moeb_rotate_trafo
import numpy as np

"""
    Wrapper which specializes matplotlibs FuncAnimation for hyperbolic tilings

    use this if you want to calculated new states live
    
    Arguments
    ---------
    state : array-like
        initial polygon state (must have same length as number of polygons in "pgons")
    fig : matplotlib.Figure
        the figure to be animated
    pgons : matplotlib.collections.PatchCollection
        the polygon patches to be animated
    step : callable
        a function which calculates the next state from the current
    stepargs : dict, optional
        additional kwargs of function "step"
    animargs : dict, optional
        additional kwargs to be passed to the FuncAnimator

"""


class hyperanimator_live:

    def __init__(self, state, fig, pgons, step, stepargs={}, animargs={}):
        self.initstate = state
        self.stepargs = stepargs
        self.anim = animation.FuncAnimation(fig, self._update, init_func=self._init, **animargs)
        self.nextstate = step
        self.pgons = pgons

    def _init(self):
        self.state = self.initstate
        self.pgons.set_array(self.state)
        return self.pgons,

    def _update(self, i):
        self.state = self.nextstate(self.state, **self.stepargs)
        self.pgons.set_array(self.state)
        return self.pgons,

    def save(self, path, fps=5, codec=None):
        writer = animation.FFMpegWriter(fps, codec)
        self.anim.save(path, writer)


"""
    Wrapper which specializes matplotlibs FuncAnimation for hyperbolic tilings

    use this if you have a pre-computed array of polygon states
    
    Arguments
    ---------
    data : 2d array-like
        list of polygon states to be traversed through during the animation
    fig : matplotlib.Figure
        the figure to be animated
    pgons : matplotlib.collections.PatchCollection
        the polygon patches to be animated
    animargs : dict, optional
        additional kwargs to be passed to the FuncAnimator

"""


class hyperanimator_list:

    def __init__(self, data, fig, pgons, animargs={}):
        if "frames" in animargs:
            if animargs["frames"] > len(data):
                animargs["frames"] = len(data)
        else:
            animargs["frames"] = len(data)

        self.anim = animation.FuncAnimation(fig, self._update, init_func=self._init, **animargs)
        self.data = data
        self.pgons = pgons

    def _init(self):
        self.pgons.set_array(self.data[0])
        return self.pgons,

    def _update(self, i):
        self.pgons.set_array(self.data[i])
        return self.pgons,

    def save(self, path, fps=5, codec=None):
        writer = animation.FFMpegWriter(fps, codec)
        self.anim.save(path, writer)


""""
    doc to follow
"""

class HyperanimatorPath:

    def __init__(self, data, fig, ax, tiling, path, path_frames=32, data_frames=None, kwargs={}, animargs={}):
        self.tiling = tiling
        self.ax = ax

        ### Check whether path has entries of type int or complex/2d float
        ### If int: entries correspond to polygon IDs
        ### If complex or 2d float: entries correspond to coordinates
        if isinstance(path, list):
            path = np.array(path)
        if path.ndim == 2 and isinstance(path[0].item(), float):
            self.coords = path + 1j * path
        elif path.ndim == 1 and isinstance(path[0].item(), complex):
            self.coords = path
        elif path.ndim == 1 and isinstance(path[0].item(), int):
            self.coords = self._poly_id_to_coords(path)
        else:
            print(" some kind of error message ")
            return

        self.path_frames = path_frames
        if not data_frames:
            self.data_frames = self.path_frames
        self.data_frames = data_frames
        self.s_coords = self._stretch_coords_geodesic(self.coords, self.path_frames)
        self.s_data = self._stretch_data(data, self.data_frames)
        self.frames = np.min([len(self.s_coords), len(self.s_data)])

        self.anim = animation.FuncAnimation(fig, self._update, frames=self.frames, **animargs)
        self.kwargs = kwargs

    def _update(self, i):
        self.ax.clear()
        self.tiling.translate(self.s_coords[i])
        self.s_coords = mymoeb(-self.s_coords[i], self.s_coords)
        pgons = poly2patch(self.tiling, self.s_data[i], **self.kwargs)
        self.ax.add_collection(pgons)

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis("off")
        self.ax.set_aspect('equal')

        return self.ax

    def _poly_id_to_coords(self, path):
        coords = np.zeros(len(path), dtype=np.complex128)
        for i in range(len(path)):
            coords[i] = self.tiling[path[i]].verticesP[-1]
        return coords

    def _stretch_pair_geodesic(self, pair, factor):
        # first, translate first entry to the origin
        t_pair = mymoeb(-pair[0], pair)

        # then, rotate second entry on to the real axis
        angle = np.angle(t_pair[1])
        r_t_pair = moeb_rotate_trafo(t_pair, -angle)

        # we go to geodesic length to calculate equal path slices of length diff
        g_r_t_stretched = np.zeros(factor, dtype=np.complex128)
        diff = np.arctanh(r_t_pair[1]) / factor

        # diff gets added 'factor'-times to the first entry
        for i in range(1, factor):
            g_r_t_stretched[i] = g_r_t_stretched[i - 1] + diff

        # go back to poincare
        r_t_stretched = np.tanh(g_r_t_stretched)
        # rotate back
        t_stretched = moeb_rotate_trafo(r_t_stretched, angle)
        # lastly, translate everything back
        stretched = mymoeb(pair[0], t_stretched)

        return stretched

    def _stretch_coords_geodesic(self, coords, factor):
        stretched_path = []

        for i in range((factor + 1) // 2):
            stretched_path.append(coords[0])
        for i in range(coords[:-1].size):
            stretched_path.extend(self._stretch_pair_geodesic(coords[i:i + 2], factor))
        for i in range((factor + 1) // 2):
            stretched_path.append(coords[-1])

        return np.array(stretched_path)

    def _stretch_data(self, data, data_frames):
        return np.repeat(data, (data_frames + 1), axis=0).reshape(len(data) * (data_frames + 1), len(data[0]))

    def save(self, path, fps=5, codec=None):
        writer = animation.FFMpegWriter(fps, codec)
        self.anim.save(path, writer)
