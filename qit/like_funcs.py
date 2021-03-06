"""High-level functions for the qp package"""

import numpy as np

from functools import partial

from qp.ensemble import Ensemble
from qp.hist_pdf import hist
from qp.interp_pdf import interp


def get_posterior_grid(ens, vals, prior=None):
    """Evaluate all of the PFDs in an ensemble at a set of values, and optionally multiply them over by a prior

    Parameters
    ----------
    ens : `qp.Ensemble`
        The ensemble
    vals : array_like (n)
        The values at which to evaluate the ensemble PDFs and the prior
    prior : `qp.Ensemble` or `None`
        The prior, using None will result in no multiplication, equivalent to a flat prior

    Returns
    -------
    post_grid : array_like (npdf, n)
        The grid of Posterior values
    """
    post_grid = ens.pdf(vals)
    if prior is not None:
        prior_vals = np.squeeze(prior.pdf(vals))
        post_grid = post_grid * prior_vals
    return post_grid


def make_ensemble_for_posterior_hist(post_grid, z_grid, z_meas_bin):
    cdfs = post_grid[z_meas_bin].cumsum(axis=1)
    pdfs = cdfs[:,1:] - cdfs[:,:-1]
    ens = Ensemble(hist, data=dict(bins=z_grid, pdfs=pdfs))
    vals = ens.pdf(z_grid)
    stack = vals.mean(axis=0)
    return dict(ens=ens, vals=vals, stack=stack)


def make_ensemble_for_posterior_interp(post_grid, z_grid, z_meas_bin):
    ens = Ensemble(interp, data=dict(xvals=z_grid, yvals=post_grid[z_meas_bin]))
    vals = ens.pdf(z_grid)
    stack = vals.mean(axis=0)
    return dict(ens=ens, vals=vals, stack=stack)


def log_hyper_like(params, ensemble, model, implicit_prior, grid):
    npdf = ensemble.npdf
    widths = grid[1:] - grid[:-1]
    cents = 0.5*(grid[1:] + grid[:-1])
    post_vals = ensemble.pdf(cents)
    model.update_objdata(dict(pdfs=np.exp(np.expand_dims(np.array(params), 0))))
    model_vals = npdf*model.pdf(cents)
    prior_vals = implicit_prior.pdf(cents)
    prior_term = model_vals / prior_vals
    integrand = post_vals * prior_term * widths
    lnlvals = np.log(np.sum(integrand, axis=1))
    return -1*np.sum(lnlvals)


def make_log_hyper_obj_func(ensemble, model, implicit_prior, grid):
    obj_func = partial(log_hyper_like, ensemble=ensemble, model=model, implicit_prior=implicit_prior, grid=grid)
    return obj_func

def model_counts(params, model, like_eval, like_grid, model_grid, cts_grid):
    pdfs = np.exp(np.expand_dims(np.array(params), 0))
    norm = pdfs.sum()
    model.update_objdata(dict(pdfs=pdfs))
    model_wts = np.squeeze(model.pdf(model_grid))
    like_hist = np.histogram(like_grid, bins=cts_grid, weights=np.matmul(model_wts, like_eval))[0]
    like_hist *= norm/like_hist.sum()
    return like_hist

def loglike_poisson(data_cts, model_cts):
    return np.sum(model_cts) - np.sum(data_cts*np.log(model_cts))

def binned_loglike(params, model, data_cts, like_eval, like_grid, model_grid, cts_grid):
    model_cts = model_counts(params, model, like_eval, like_grid, model_grid, cts_grid)
    return loglike_poisson(data_cts, model_cts)

def make_binnned_loglike_obj_func(model, data_cts, like_eval, like_grid, model_grid, cts_grid):
    obj_func = partial(binned_loglike, model=model, data_cts=data_cts, like_eval=like_eval,
                           like_grid=like_grid, model_grid=model_grid, cts_grid=cts_grid)
    return obj_func
