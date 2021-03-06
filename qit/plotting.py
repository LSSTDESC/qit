"""Functions to plot PDFs"""


import numpy as np

def plot_2d_like(post_grid, **kwargs):
    """Utility function to plot the posteriors for a grid"""

    axes, _, kw = get_axes_and_xlims(**kwargs)
    ylabel = kw.get('ylabel', None)
    ylim = kw.get('ylim', None)

    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if ylim is not None:
        axes.set_ylim(ylim[0], ylim[1])

    im = axes.imshow(post_grid, origin='lower',
                        extent=(axes.get_xlim()[0], axes.get_xlim()[1], axes.get_ylim()[0], axes.get_ylim()[1]),
                        interpolation='none')
    return axes.figure, axes, im
