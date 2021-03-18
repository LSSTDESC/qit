""" Implemenation of a set of Poseterior distributions
"""

import numpy as np

from qp.interp_pdf import interp

from .axis import Axis
from .nd_ensemble import NDEnsemble

class Posterior(NDEnsemble):
    """ A set of Posterior distributions
    """
    def __init__(self, parameterization, parameters, index_vars, **kwargs):
        """ C'tor """
        self._likelihood = kwargs.get('likelihood', None)
        self._priors = kwargs.get('priors', None)
        super(Posterior, self).__init__(parameterization, parameters, index_vars)


    @classmethod
    def create_from_grid(cls, likelihood, x_loc, **kwargs):
        """ Build a posterior from a set of grid values """
        prior = kwargs.get('prior', None)
        idx = kwargs.get('idx', 0)
        x_eval = 0.5*(x_loc[:-1] + x_loc[1:])
        p_grid = likelihood.get_posterior_grid(x_eval, prior, idx)
        bins = likelihood.index_vars[idx].bins
        bin_cents = 0.5*(bins[:-1] + bins[1:])
        if prior is not None:
            priors = [prior]
        else:
            priors = None
        return cls(interp, parameters=dict(xvals=bin_cents, yvals=p_grid.T), index_vars=[x_loc], likelihood=likelihood, priors=priors)

    @classmethod
    def create_from_samples(cls, likelihood, x_loc, samples, **kwargs):
        """ Make an ensemble that represents the PDFs from a collection of samples """
        prior = kwargs.get('prior', None)
        idx = kwargs.get('idx', 0)
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = likelihood.pdf(x_eval)
        bins = likelihood.index_vars[idx].bins
        bin_cents = 0.5*(bins[:-1] + bins[1:])
        if prior is not None:
            post_grid = post_grid_flat.T * prior.pdf(bin_cents)
            priors = [prior]
        else:
            post_grid = post_grid_flat.T
            priors = None

        new_axis = Axis(x_loc)
        idxs, mask = new_axis.get_indices(samples)

        return cls(interp, parameters=dict(xvals=bin_cents, yvals=post_grid[idxs[mask]]), index_vars=[np.arange(len(idxs)+1)], likelihood=likelihood, priors=priors)
