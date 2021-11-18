"""Define an object that estimate PDF values using a look-table
"""

import numpy as np

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
        """ Get the indices and masks across the estimator axes

        Parameters
        ----------
        x_loc : array_like
            The locations along each of the axes

        Returns
        -------
        indices : array_like
            The indices into the qp.Ensemble
        masks : array_like
            Masks indicating which elements are in the supported ranges
        """
        if len(x_loc.shape) == 1:
            if self._naxes != 1:  #pragma: no cover
                raise ValueError("Number of input vectors must equal number of axes 1 != %i" % (self._naxes))
            return self._axes[0].get_indices(x_loc)
        if len(x_loc) != self._naxes:  #pragma: no cover
            raise ValueError("Number of input vectors must equal number of axes %i != %i" % (len(x_loc), self._naxes))
        indices = []  #pragma: no cover
        masks = []  #pragma: no cover
        for ax, x_loc_ in zip(self._axes, x_loc):  #pragma: no cover
            idx_, mask_ = ax.get_indices(x_loc_)
            indices.append(idx_)
            masks.append(mask_)
        return np.vstack(indices), np.vstack(masks)  #pragma: no cover


    def flat_posterior(self, x_loc, axis=0):
        """ Return an estimator constructed assuming a flat prior
        Parameters
        ----------
        x_loc : array_like
            The locations along each of the axes

        Returns
        -------
        post = `qp.Ensemble`
            The posterior assuming a flat prior
        """
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = self._ensemble.pdf(x_eval).T
        bins = self._axes[axis].bins
        return Estimator([x_loc], Ensemble(hist, data=dict(bins=bins, pdfs=post_grid_flat)))


    def get_posterior_grid(self, x_loc, prior=None):
        """Evaluate all of the PFDs in an ensemble at a set of values, and optionally multiply them over by a prior

        Parameters
        ----------
        x_loc : array_like (n)
            The values at which to evaluate the ensemble PDFs and the prior
        prior : `qp.Ensemble` or `None`
            The prior, using None will result in no multiplication, equivalent to a flat prior

        Returns
        -------
        post_grid : array_like (npdf, n)
           The grid of Posterior values
        """
        pdf_grid = self._ensemble.pdf(x_loc)
        pdf_grid = np.where(np.isnan(pdf_grid), 0, pdf_grid)
        flat_post = Ensemble(hist, data=dict(bins=self._axes[0].bins, pdfs=pdf_grid.T))
        out_grid = 0.5*(self._axes[0].bins[1:] + self._axes[0].bins[0:-1])
        return get_posterior_grid(flat_post, out_grid, prior, x_loc)

    def get_sampler(self, x_loc):
        """ Construct a PDF sampler as an ensemble
        by selecting the PDFs correspond to the locations x_loc

        ----------
        x_loc : array_like
            The locations along each of the axes

        Returns
        -------
        sample = `qp.Ensemble`
            The output ensemble
        """
        idx, mask = self.get_indices(x_loc)
        idx = idx[mask]
        return self._ensemble[idx]

    def make_posterior_ensemble(self, x_loc, samples, prior=None, axis=0):
        """ Return an estimator constructed assuming a flat prior

        Parameters
        ----------
        x_loc : array_like
            The locations along the relevant axis
        samples : array_like
            Sample locations
        prior : `qp.Ensemble` or `None`
            Single PDF ensemble representing the prior.  None -> flat prior
        axis : `int`
            The axis used to define the bins

        Returns
        -------
        post = `qp.Ensemble`
            The posterior assuming a flat prior
        """
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        flat_post = self.flat_posterior(self._axes[axis].bins)
        post_grid = flat_post.get_posterior_grid(x_eval, prior)
        idx, mask = self._axes[axis].get_indices(samples)
        idx = idx[mask]
        return Ensemble(interp, data=dict(xvals=x_eval, yvals=post_grid[idx]))
