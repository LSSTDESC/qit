"""This module implements a binned Poisson likelihood
"""

import numpy as np

from .like_funcs import loglike_poisson

class BinnedLike:
    """Implementation of a BinnedLikelihood"""
    
    def __init__(self, model, data_hist, estimator, like_grid, model_grid):
        """ C'tor define the binned likelihood 

        Parameters
        ----------
        model : `qp.ensemble`
            The model of the data
        data_hist : `np.histogram`
            The binned counts data
        estimator : `qit.estimator`
            The estimator 
        like_grid : `np.array` 
            The grid on which to evaluate the likelihood
        model_grid : `np.array`
            The grid on which to evaluate the model
        """
        self._model = model
        self._data_hist = data_hist
        self._estimator = estimator
        self._like_grid = like_grid
        self._model_grid = model_grid
        self._like_bins, self._like_mask = self._estimator.get_indices(self._model_grid)
        self._like_eval = self._estimator._ensemble.pdf(self._like_grid)[self._like_bins[self._like_mask]]

    def model_counts(self, params):
        """ Evaluate the model counts 
        
        Parameters
        ----------
        params : `np.array`
            The input parameters

        Returns
        -------
        counts : `np.array`
            The model counts        
        """
        pdfs = np.exp(np.expand_dims(np.array(params), 0))
        norm = pdfs.sum()
        self._model.update_objdata(dict(pdfs=pdfs))
        model_wts = np.squeeze(self._model.pdf(self._model_grid))[self._like_mask]
        like_hist = np.histogram(self._like_grid, bins=self._data_hist[1], weights=np.matmul(model_wts, self._like_eval))[0]
        like_hist *= norm/like_hist.sum()
        return like_hist

    def binned_loglike(self, params):
        """ Return the Poisson log-likelihood given data, a model, model parameters

        Parameters
        ----------
        params : `np.array`
            The input parameters

        Returns
        -------
        loglike : `float`
            The log-likelhood
        """
        model_cts = self.model_counts(params)
        return loglike_poisson(self._data_hist[0], model_cts)

    def __call__(self, params):
        """ Return the Poisson log-likelihood given data, a model, model parameters """
        return self.binned_loglike(params)




class Likelihood:
    """Implementation of a Likelihood estimation"""
    
    def __init__(self, model, posterior, grid):
        """
        """
        self._model = model
        self._posterior = posterior
        self._npdf = self._posterior._ensemble.npdf
        self._grid = grid
        self._post_vals = self._posterior._ensemble.pdf(self._grid)
        if self._posterior._priors is not None:
            self._prior_vals = self._posterior._priors[0].pdf(self._grid)
        else:
            self._prior_vals = 1.
            
    def model_vals(self, params):
        self._model.update_objdata(dict(pdfs=np.exp(np.expand_dims(np.array(params), 0))))
        return self._model.pdf(self._grid)

    def loglike(self, params):
        mv = self.model_vals(params)
        prior_term = mv / self._prior_vals
        integrand = self._post_vals * self.model_vals(params) / self._prior_vals
        lnlvals = np.log(np.trapz(integrand, self._grid))
        return -1*np.sum(lnlvals)

    def __call__(self, params):
        """ Return the Poisson log-likelihood given data, a model, model parameters """
        return self.loglike(params)
