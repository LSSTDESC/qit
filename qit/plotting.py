"""Functions to plot PDFs"""

import matplotlib.pyplot as plt

def plot_2d_like(post_grid, **kwds):  #pragma: no cover
    """Utility function to plot the posteriors for a grid"""
    kwcopy = kwds.copy()
    fig = plt.figure()
    axes = fig.add_subplot(111)
    xlabel = kwcopy.pop('xlabel', r'$z$')
    ylabel = kwcopy.pop('ylabel', r'$z$')
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)

    _ = axes.imshow(post_grid, origin='lower', interpolation='none',
                    extent=kwcopy.get('extent', None))
    return axes
