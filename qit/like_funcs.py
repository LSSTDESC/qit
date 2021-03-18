"""High-level functions for the qp package"""

import numpy as np


def get_posterior_grid(ens, x_vals, prior=None, y_vals=None):
    """Evaluate all of the PFDs in an ensemble at a set of values, and optionally multiply them over by a prior

    Parameters
    ----------
    ens : `qp.Ensemble`
        The ensemble
    x_vals : array_like (n)
        The values at which to evaluate the ensemble PDFs
    y_vals : array_like (n)
        The values at which to evaluate the priors
    priors : `qp.Ensemble` or `list` or `None`
        The prior, using None will result in no multiplication, equivalent to a flat prior

    Returns
    -------
    post_grid : array_like (npdf, n)
        The grid of Posterior values
    """
    post_grid = ens.pdf(x_vals)
    if prior is None:
        return post_grid
    return (post_grid.T * prior.pdf(y_vals)).T


def loglike_poisson(data_cts, model_cts):
    """ Return the Poisson likelihood given data and a model """
    return np.sum(model_cts) - np.sum(data_cts*np.log(model_cts))
