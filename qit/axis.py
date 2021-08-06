"""This contains a simple implementation of an axis,
which can be used give context to qp.Ensemble objects.
"""

import numpy as np

from qp.utils import get_bin_indices #, get_indices_and_interpolation_factors

class Axis:
    """ Simple implementation of an axis

    This is used to give context to qp.Ensemble objects,
    by defining what the individual PDFs represent
    """

    def __init__(self, bins):
        """ C'tor, provide the bins """
        if len(np.shape(bins)) != 1:  #pragma: no cover
            raise ValueError("Only 1D inputs can be used to define axes, got %s" % np.shape(bins))
        self._bins = bins
        self._nbins = self._bins.size - 1

    @property
    def bins(self):
        """ Return the axis bins """
        return self._bins

    @property
    def nbins(self):
        """ Return the number of bins on the axis"""
        return self._nbins

    def get_indices(self, vals):
        """ Extract bin indices for vals """
        return get_bin_indices(self.bins, vals)

    def get_indices_and_interpolation_factors(self, vals):  #pragma: no cover
        """ Extract bin indices and interpolation factors for vals """
        raise NotImplementedError()
        #return get_indices_and_interpolation_factors(self.bins, vals)
