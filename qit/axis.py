"""This module implements an axis of a multi-dimensional distribution
"""

import numpy as np

from qp.utils import get_bin_indices, get_indices_and_interpolation_factors

class Axis:
    """An axis of a multi-dimensional distribution"""

    def __init__(self, bins):
        """ C'tor, define bin edges """
        if len(np.shape(bins)) != 1:
            raise ValueError("Only 1D inputs can be used to define axes, got %s" % str(np.shape(bins)))
        self._bins = bins
        self._nbins = self._bins.size - 1

    @property
    def bins(self):
        """ Return the bin edges """
        return self._bins

    @property
    def nbins(self):
        """ Return the number of bins """
        return self._nbins

    def get_indices(self, locs):
        """ Get the indices for a set of locations

        Returns
        -------
        idx : Array[int]
            The bin indices
        mask : Array[bool]
            Mask of values inside the axis limits
        """
        return get_bin_indices(self.bins, locs)

    def get_indices_and_interpolation_factors(self, locs):
        """ Get the indices and interpolation for a set of locations """
        return get_indices_and_interpolation_factors(self._bins, locs)
