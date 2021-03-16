""" A multi-dimensional estimator
"""

import numpy as np

from qp.ensemble import Ensemble

from .axis import Axis
from .like_funcs import get_posterior_grid

class BinnedEnsemble(Ensemble):
    """ A multi-dimensional Ensemble

    Sub-class of qp.Ensemble object with axes that allow you to make input values to
    specific PDFs in the ensemble.
    """
    def __init__(self, gen_func, data, axes):
        """ C'tor """
        self._axes = [ Axis(ax) for ax in axes ]
        self._axes_shape = [ ax.nbins for ax in self._axes ]
        self._naxes = len(self._axes)
        super(BinnedEnsemble, self).__init__(gen_func, data)
        #if self._axes_shape != list(self.shape):
        #    raise ValueError("Axes shape does not match Ensemble shape %s != %s" % (self._axes_shape, self.shape))

    @property
    def axes(self):
        """ Return the list of Axis associated to this Ensemble """
        return self._axes

    @property
    def axes_shape(self):
        """ Return the shape of the Axes associated to this Ensemble """
        return self._axes_shape

    @property
    def naxes(self):
        """ Return the number of Axes associated to this Ensemble """
        return self._naxes

    def get_indices(self, x_loc):
        """ Get the indices for a vector of input values """
        if len(np.shape(x_loc)) == 1:
            if self._naxes != 1:
                raise ValueError("Number of input vectors must equal number of axes 1 != %i" % (self._naxes))
            return self._axes[0].get_indices(x_loc)
        if len(x_loc) != self._naxes:
            raise ValueError("Number of input vectors must equal number of axes %i != %i" % (len(x_loc), self._naxes))
        indices = []
        masks = []
        for ax_, x_loc_ in zip(self._axes, x_loc):
            idx_, mask_ = ax_.get_indices(x_loc_)
            indices.append(idx_)
            masks.append(mask_)
        return np.array(indices), np.array(masks)

    def get_posterior_grid(self, x_loc, prior=None, axis=0):
        """ Build a grid of posterior distribution using a set of locations and optionally a set of priors
        """
        y_bins = self._axes[axis].bins
        y_vals = 0.5*(y_bins[1:] + y_bins[:-1])
        return get_posterior_grid(self, x_loc, prior, y_vals)

    def get_sampler(self, x_loc):
        """ Return a qp.Ensemble that will sample the appropriate PDFs from the wrapped qp.Ensemble """
        idx, mask = self.get_indices(x_loc)
        idx = idx[mask]
        return self[idx]
