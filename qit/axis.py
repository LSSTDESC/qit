"""This module implements a PDT distribution sub-class using interpolated quantiles
"""

import sys
import numpy as np

from qp.utils import get_bin_indices, get_indices_and_interpolation_factors

class Axis:

    def __init__(self, bins):
        if len(np.shape(bins)) != 1:
            raise ValueError("Only 1D inputs can be used to define axes, got %s" % np.shape(self._bins))
        self._bins = bins
        self._nbins = self._bins.size - 1

    @property
    def bins(self):
        return self._bins

    @property
    def nbins(self):
        return self._nbins
           
    def get_indices(self, vals):
        return get_bin_indices(self.bins, vals)        

    def get_indices_and_interpolation_factors(self, vals):
        return get_indices_and_interpolation_factors(self.bins, vals)

