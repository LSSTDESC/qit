""" A multi-dimensional estimator
"""

import numpy as np

from qp.ensemble import Ensemble
from qp.hist_pdf import hist
from qp.interp_pdf import interp

from .axis import Axis
from .like_funcs import get_posterior_grid
from .posterior import Posterior

class Estimator:
    """ A multi-dimensional estimator

    Wraps a qp.Ensemble object with axes that allow you to make input values to
    specific PDFs in the ensemble.
    """
    def __init__(self, axes, ensemble, priors=None):
        """ C'tor """
        self._axes = [ Axis(ax) for ax in axes ]
        self._axes_shape = [ ax.nbins for ax in self._axes ]
        self._naxes = len(self._axes)
        self._ens_shape = ensemble.shape
        self._ensemble = ensemble
        self._priors = priors
        if self._axes_shape != list(self._ens_shape):
            raise ValueError("Axes shape does not match Ensemble shape %s != %s" % (self._axes_shape, self._ens_shape))

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


    def pdf_grid(self, x_loc, y_loc):
        idx, mask = self.get_indices(x_loc)
        return self._ensemble.pdf(y_loc)[idx[mask]]

    
    def flat_posterior(self, x_loc, axis=0):
        """ Build a flat posterior distribution using a set of locations
        """
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = self._ensemble.pdf(x_eval).T
        return Estimator([x_loc], Ensemble(hist, data=dict(bins=self._axes[axis].bins, pdfs=post_grid_flat)))


    def get_posterior_grid(self, x_loc, prior=None):
        """ Build a grid of posterior distribution using a set of locations and optionally a set of priors
        """
        return get_posterior_grid(self._ensemble, x_loc, prior)

    def get_sampler(self, x_loc):
        """ Return a qp.Ensemble that will sample the appropriate PDFs from the wrapped qp.Ensemble """
        idx, mask = self.get_indices(x_loc)
        idx = idx[mask]
        return self._ensemble[idx]

    def make_posterior(self, x_loc, samples, prior=None, axis=0):
        """ Make an ensemble that represents the PDFs from a collection of samples """
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = self._ensemble.pdf(x_eval).T
        bins = self._axes[axis].bins
        flat_post = Estimator([x_loc], Ensemble(hist, data=dict(bins=bins, pdfs=post_grid_flat)))
        post_grid = flat_post.get_posterior_grid(x_eval, prior)
        idx, mask = flat_post.get_indices(samples)
        idx = idx[mask]
        if prior is not None:
            priors = [prior]
        else:
            priors = None
        return Posterior(Ensemble(interp, data=dict(xvals=x_eval, yvals=post_grid[idx])), priors)
