from matplotlib import animation
from hypertiling.graphics.plot import convert_polygons_to_patches
from hypertiling.transformation import mymoebint
import numpy as np


def moeb_rotate_trafo(phi, z):
    return z * (np.cos(phi) + 1j * np.sin(phi))
    
mymoebint = np.vectorize(mymoebint)


class AnimatorLive:

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

    def __init__(self, state, fig, pgons, step, stepargs={}, animargs={}):
        self.initstate = state
        self.stepargs = stepargs
        self.pgons = pgons
        self.nextstate = step
        self.anim = animation.FuncAnimation(fig, self._update, init_func=self._init, **animargs)

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


class AnimatorList:

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


class AnimatorPath:
    """
    Wrapper which specializes matplotlibs FuncAnimation for hyperbolic tilings

    use this if you have a pre-computed array of polygon states and want to animate a moving tiling
    
    Arguments
    ---------
    data : 2d array-like
        list of polygon states to be traversed through during the animation
    fig : matplotlib.Figure
        the figure to be animated
    ax : matplotlib.axes
        the axes in which the animation is shown
    tiling : hypertiling.HyperbolicTiling
        the tiling to be animated
    path : 1d or 2d array-like
        points in the Poincare disk which will be moved to the center throughout the animation.
        if path is 1d and the elements are integers, the points are expected to be polygon id's.
        if path is 1d and the elements are complex, the points are expected to be coordinates on the Poincare disk
        if path is 2d and the elements are floats, the points are expected to be coordinates in the poincare disk,
        where the 1st dimension hold the real part and the 2nd the complex
    path_frames : int, default: 32
        how many frames it takes to go from one point to the next in path
    data_frames : int, default: None
        how many frames it takes to go from one state to the next in data 
        if no value is given, data_frames takes the same value as path_frames
    kwargs : dict, optional
        additional matplotlib.Patch kwargs
    animargs : dict, optional
        additional kwargs to be passed to the FuncAnimator

    """

    def __init__(self, data, fig, ax, tiling, path, path_frames=32, data_frames=None, kwargs={}, animargs={}):
        self.tiling = tiling
        self.ax = ax


        # Check whether path has entries of type int or complex/2d float
        # If int: entries correspond to polygon IDs
        # If complex or 2d float: entries correspond to coordinates
        if isinstance(path, list):
            path = np.array(path)
        if path.ndim == 2 and isinstance(path[0].item(), float):
            self.coords = path + 1j * path
        elif path.ndim == 1 and isinstance(path[0].item(), complex):
            self.coords = path
        elif path.ndim == 1 and isinstance(path[0].item(), int):
            self.coords = self._poly_id_to_coords(path)
        else:
            raise ValueError("[hypertiling] Error: Invalid input format for path")
        

        self.path_frames = path_frames
        if not data_frames:
            self.data_frames = self.path_frames
        else:
            self.data_frames = data_frames
        self.s_coords = self._stretch_coords_geodesic(self.coords, self.path_frames)
        self.s_data = self._stretch_data(data, self.data_frames, len(self.s_coords))
        self.frames = np.min([len(self.s_coords), len(self.s_data)])

        self.anim = animation.FuncAnimation(fig, self._update, frames=self.frames, **animargs)
        self.kwargs = kwargs



    def _update(self, i):
        self.ax.clear()
        
        self.tiling.translate(self.s_coords[i])
        self.s_coords = mymoebint(-self.s_coords[i], self.s_coords)[0]
        pgons = convert_polygons_to_patches(self.tiling, self.s_data[i], **self.kwargs)
        self.ax.add_collection(pgons)

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis("off")
        self.ax.set_aspect('equal')

        return self.ax


    def _poly_id_to_coords(self, path):
        """ Takes list of polygon id's and returns list containing the respective coordinates """
        coords = np.zeros(len(path), dtype=np.complex128)
        for i in range(len(path)):
            coords[i] = self.tiling.get_center(path[i])
        return coords


    def _stretch_pair_geodesic(self, pair, factor):
        """ Takes a list containing two coordinates and divides the path between them into "factor" geodesic parts """

        # t = translated
        # r = rotated
        # g = geodesic

        # Translate first entry to the origin
        t_pair = mymoebint(-pair[0], pair)[0]

        # Rotate second entry on to the real axis
        angle = np.angle(t_pair[1])
        r_t_pair = moeb_rotate_trafo(-angle, t_pair)

        # Go to geodesic length to calculate equal path slices of length diff
        g_r_t_stretched = np.zeros(factor, dtype=np.complex128)
        diff = np.arctanh(r_t_pair[1]) / factor

        # Diff gets added 'factor'-times to the first entry
        for i in range(1, factor):
            g_r_t_stretched[i] = g_r_t_stretched[i - 1] + diff

        # Go back to poincare
        r_t_stretched = np.tanh(g_r_t_stretched)
        # Rotate back
        t_stretched = moeb_rotate_trafo(angle, r_t_stretched)
        # Translate everything back
        stretched = mymoebint(pair[0], t_stretched)[0]

        return stretched


    def _stretch_coords_geodesic(self, coords, factor):
        """ Takes list of all coordinates to be visited and stretches it "factor" times """

        stretched_path = []

        # Replicate the first position
        for i in range((factor + 1) // 2):
            stretched_path.append(coords[0])
        # Stretch the in between positions
        for i in range(coords[:-1].size):
            stretched_path.extend(self._stretch_pair_geodesic(coords[i:i + 2], factor))
        # Replicate the last position
        for i in range((factor + 1) // 2):
            stretched_path.append(coords[-1])

        return np.array(stretched_path)


    def _stretch_data(self, data, data_frames, len_coords):
        try:
            len(data[0])
            return np.repeat(data, (data_frames + 1), axis=0)
        except:
            return np.tile(data, len_coords).reshape(len_coords, len(data))


    def save(self, path, fps=5, codec=None):
        writer = animation.FFMpegWriter(fps, codec)
        self.anim.save(path, writer)
