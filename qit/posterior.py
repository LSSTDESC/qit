""" Implemenation of a set of Poseterior distributions
"""

from qp.interp_pdf import interp

from .axis import Axis
from .binned_ensemble import BinnedEnsemble

class Posterior(BinnedEnsemble):
    """ A set of Posterior distributions
    """
    def __init__(self, gen_func, data, axes, **kwargs):
        """ C'tor """
        self._likelihood = kwargs.get('likelihood', None)
        self._priors = kwargs.get('priors', None)
        super(Posterior, self).__init__(gen_func, data, axes)


    @classmethod
    def create_from_grid(cls, likelihood, x_loc, **kwargs):
        """ Build a posterior from a set of grid values """
        prior = kwargs.get('prior', None)
        axis = kwargs.get('axis', 0)
        p_grid = likelihood.get_posterior_grid(x_loc, prior, axis)
        bins = likelihood.axes[axis].bins
        bin_cents = 0.5*(bins[:-1] + bins[1:])
        if prior is not None:
            priors = [prior]
        else:
            priors = None
        print(x_loc.shape, likelihood.shape, p_grid.shape, bins.shape)
        return cls(interp, data=dict(xvals=bin_cents, yvals=p_grid.T), axes=[x_loc], likelihood=likelihood, priors=priors)

    @classmethod
    def create_from_samples(cls, likelihood, x_loc, samples, **kwargs):
        """ Make an ensemble that represents the PDFs from a collection of samples """
        prior = kwargs.get('prior', None)
        axis = kwargs.get('axis', 0)
        x_eval = 0.5*(x_loc[1:] + x_loc[:-1])
        post_grid_flat = likelihood.pdf(x_eval)
        bins = likelihood.axes[axis].bins
        bin_cents = 0.5*(bins[:-1] + bins[1:])
        if prior is not None:
            post_grid = post_grid_flat.T * prior.pdf(bin_cents)
            priors = [prior]
        else:
            post_grid = post_grid_flat.T
            priors = None

        new_axis = Axis(x_loc)
        idx, mask = new_axis.get_indices(samples)

        return cls(interp, data=dict(xvals=bin_cents, yvals=post_grid[idx[mask]]), axes=[idx], likelihood=likelihood, priors=priors)
