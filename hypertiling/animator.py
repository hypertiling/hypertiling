from matplotlib import animation
from hypertiling.plot import poly2patch
from hypertiling.transformation import mymoeb


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

    def _update(self,i):
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

    def _update(self,i):
        self.pgons.set_array(self.data[i])
        return self.pgons,
    
    def save(self, path, fps=5, codec=None):
        writer = animation.FFMpegWriter(fps, codec)
        self.anim.save(path, writer)

""""
    doc to follow
"""
class hyperanimator_path:
    def __init__(self, data, fig, ax, tiling, path, smoothness=32):
        # if "frames" in animargs:
        #     if animargs["frames"] > len(data):
        #         animargs["frames"] = len(data)
        # else:
        #     animargs["frames"] = len(data)

        self.anim = animation.FuncAnimation(fig, self._update, frames=len(data), blit=True)
        self.data = data
        self.tiling = tiling
        self.path = path
        self.smoothness = smoothness
        self.ax = ax


        ### check whether path has entries of type int, or complex/2d float
        ### if int: entries correspond to polygon id's
        ### if complex or 2d float: entries correspond to coordinates

    def _update(self, i):
        self.ax.clear()
        self.tiling.translate(self.path[i])
        self.path = mymoeb(-self.path[i], self.path)
        #pgons = poly2patch(self.tiling, fancy_colors(stretched_colors[i], layer_color), lazy=True, cutoff=0.003, **kwargs)
        pgons = poly2patch(self.tiling, self.data[i])
        self.ax.add_collection(pgons)

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis("off")
        self.ax.set_aspect('equal')

        return self.ax