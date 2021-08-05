"""
"""

import sys
import numpy as np

from qp.utils import get_bin_indices #, get_indices_and_interpolation_factors
from qp.ensemble import Ensemble
from qp.hist_pdf import hist
from qp.interp_pdf import interp

from .axis import Axis
from .like_funcs import get_posterior_grid

class Estimator:

    def __init__(self, axes, ensemble):
        self._axes = [ Axis(ax) for ax in axes ]
        self._axes_shape = [ ax.nbins for ax in self._axes ]
        self._naxes = len(self._axes)
        self._ens_shape = ensemble.shape
        self._ensemble = ensemble
        if self._axes_shape != list(self._ens_shape):  #pragma: no cover
            raise ValueError("Axes shape does not match Ensemble shape %s != %s" % (self._axes_shape, self._ens_shape))

    def get_indices(self, x_loc):        
        if len(x_loc.shape) == 1:
            if self._naxes != 1:  #pragma: no cover
                raise ValueError("Number of input vectors must equal number of axes 1 != %i" % (self._naxes))
            return self._axes[0].get_indices(x_loc)
        if len(x_loc) != self._naxes:  #pragma: no cove
            raise ValueError("Number of input vectors must equal number of axes %i != %i" % (len(x_loc), self._naxes))
        indices = []  #pragma: no cover
        masks = []  #pragma: no cover
        for x_loc_ in x_loc:  #pragma: no cover
            idx_, mask_ = ax.get_indices(x_loc_)
            indices.append(idx_)
            masks.append(mask_)
        return np.vstack(indices), np.vstack(masks)  #pragma: no cover
    

    def flat_posterior(self, x_loc, axis=0):
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = self._ensemble.pdf(x_eval).T
        bins = self._axes[axis]._bins
        return Estimator([x_loc], Ensemble(hist, data=dict(bins=bins, pdfs=post_grid_flat)))


    def get_posterior_grid(self, x_loc, prior=None):
        return get_posterior_grid(self._ensemble, x_loc, prior)

    def get_sampler(self, x_loc):
        idx, mask = self.get_indices(x_loc)
        idx = idx[mask]
        return self._ensemble[idx]

    def make_posterior_ensemble(self, x_loc, samples, prior=None, axis=0):
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = self._ensemble.pdf(x_eval).T
        bins = self._axes[axis]._bins
        flat_post = Estimator([x_loc], Ensemble(hist, data=dict(bins=bins, pdfs=post_grid_flat)))
        post_grid = flat_post.get_posterior_grid(x_eval, prior)
        idx, mask = flat_post.get_indices(samples)
        idx = idx[mask]
        return Ensemble(interp, data=dict(xvals=x_eval, yvals=post_grid[idx]))
