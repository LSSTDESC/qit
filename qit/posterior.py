""" A multi-dimensional estimator
"""

import numpy as np

from qp.ensemble import Ensemble
from qp.hist_pdf import hist
from qp.interp_pdf import interp

from .axis import Axis
from .like_funcs import get_posterior_grid

class Posterior:
    """ A multi-dimensional estimator

    Wraps a qp.Ensemble object with axes that allow you to make input values to
    specific PDFs in the ensemble.
    """
    def __init__(self, ensemble, priors=None):
        """ C'tor """
        self._ens_shape = ensemble.shape
        self._ensemble = ensemble
        self._priors = priors
