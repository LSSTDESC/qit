""" A multi-dimensional estimator
"""

import numpy as np

from qp.ensemble import Ensemble

from .axis import Axis
from .like_funcs import get_posterior_grid

class NDEnsemble(Ensemble):
    """ A multi-dimensional Ensemble

    Sub-class of qp.Ensemble object with axes that allow you to make input values to
    specific PDFs in the ensemble.
    """
    def __init__(self, parameterization, parameters, index_vars):
        """ C'tor """
        self._index_vars = [ Axis(ax) for ax in index_vars ]
        self._index_vars_shape = [ ax.nbins for ax in self._index_vars ]
        self._nindex_vars = len(self._index_vars)
        super(NDEnsemble, self).__init__(parameterization, parameters)
        if self._index_vars_shape != list(self.shape):
            raise ValueError("Index_vars shape does not match Ensemble shape %s != %s" % (self._index_vars_shape, self.shape))

    @property
    def index_vars(self):
        """ Return the list of Axis associated to this Ensemble """
        return self._index_vars

    @property
    def index_vars_shape(self):
        """ Return the shape of the Index_vars associated to this Ensemble """
        return self._index_vars_shape

    @property
    def nindex_vars(self):
        """ Return the number of Index_vars associated to this Ensemble """
        return self._nindex_vars

    def get_indices(self, x_loc):
        """ Get the indices for a vector of input values """
        if len(np.shape(x_loc)) == 1:
            if self._nindex_vars != 1:
                raise ValueError("Number of input vectors must equal number of index_vars 1 != %i" % (self._nindex_vars))
            return self._index_vars[0].get_indices(x_loc)
        if len(x_loc) != self._nindex_vars:
            raise ValueError("Number of input vectors must equal number of index_vars %i != %i" % (len(x_loc), self._nindex_vars))
        indices = []
        masks = []
        for ax_, x_loc_ in zip(self._index_vars, x_loc):
            idx_, mask_ = ax_.get_indices(x_loc_)
            indices.append(idx_)
            masks.append(mask_)
        return np.array(indices), np.array(masks)

    def get_posterior_grid(self, x_loc, prior=None, idx=0):
        """ Build a grid of posterior distribution using a set of locations and optionally a set of priors
        """
        y_bins = self._index_vars[idx].bins
        y_vals = 0.5*(y_bins[1:] + y_bins[:-1])
        return get_posterior_grid(self, x_loc, prior, y_vals)

    def get_sampler(self, x_loc):
        """ Return a qp.Ensemble that will sample the appropriate PDFs from the wrapped qp.Ensemble """
        idx, mask = self.get_indices(x_loc)
        idx = idx[mask]
        return self[idx]
