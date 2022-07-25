from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot(tiling: Iterable, fig_ax: Optional[Tuple] = None, numerate: bool = False, fontsize: int = 6,
         **kwargs) -> Tuple:
    """
    Plots the tiling given by tiling.
    :param tiling: Iterable = Iterable of the tiling
    :param fig_ax: Tuple[fig, ax] = figure and axis the plot should be ploted in
    :param numerate: bool = Describes if the index of the polygon should be shown
    :param fontsize: int =  fontsize in which the index will be written
    :return: Tuple[fig, ax] = Figure and axis with plotted grid
    """
    if fig_ax is None:
        fig_ax = plt.subplots()
        fig_ax[1].set_xlim(-1, 1)
        fig_ax[1].set_ylim(-1, 1)

    for i, pgon in enumerate(tiling):
        p = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgon[1:]]), **kwargs)
        fig_ax[1].add_patch(p)
        if numerate:
            fig_ax[1].text(np.real(pgon[0]), np.imag(pgon[0]), i, fontsize=fontsize, horizontalalignment='center',
                           verticalalignment='center')
    return fig_ax
