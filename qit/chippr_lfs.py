"""This module implements tools to estimate the Prior used in the chippr example
"""

import numpy as np

from .like_funcs import loglike_poisson

class CHIPPR_count_LF:
    """Implementation of prior estimation by forward folding the number of counts per bin in the observed data space """

    def __init__(self, redshift_distribution, data_hist, joint_probspace, discretized_data, discretized_redshift): #pylint: disable=too-many-arguments
        """ C'tor define the binned likelihood

        Parameters
        ----------
        redshift_distribution : `qp.ensemble`
            The model of the data
        data_hist : `np.histogram`
            The binned counts data
        joint_probspace : `qit.BinnedEnsemble`
            The likelihood estimator
        discretized_data : `np.array`
            The grid on which to evaluate the likelihood
        discretized_redshift : `np.array`
            The grid on which to evaluate the model
        """
        self._redshift_distribution = redshift_distribution
        self._data_hist = data_hist
        self._joint_probspace = joint_probspace
        self._discretized_data = discretized_data
        self._discretized_redshift = discretized_redshift
        self._like_bins, self._like_mask = self._joint_probspace.get_indices(self._discretized_redshift)
        self._like_eval = self._joint_probspace.pdf(self._discretized_data)[self._like_bins[self._like_mask]]

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
        self._redshift_distribution.update_objdata(dict(pdfs=pdfs))
        model_wts = np.squeeze(self._redshift_distribution.pdf(self._discretized_redshift))[self._like_mask]
        like_hist = np.histogram(self._discretized_data, bins=self._data_hist[1], weights=np.matmul(model_wts, self._like_eval))[0]
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




class CHIPPR_LF:
    """Implementation of a Likelihood estimation"""

    def __init__(self, redshift_distribution, posterior_ensemble, discretized_redshift):
        """
        """
        self._redshift_distribution = redshift_distribution
        self._posterior_ensemble = posterior_ensemble
        self._npdf = self._posterior_ensemble.npdf
        self._discretized_redshift = discretized_redshift
        self._nmodel = 0.
        self._post_vals = self._posterior_ensemble.pdf(self._discretized_redshift)
        if self._posterior_ensemble._priors is not None:
            self._prior_vals = self._posterior_ensemble._priors[0].pdf(self._discretized_redshift)
        else:
            self._prior_vals = 1.

    def model_vals(self, params):
        """ Evaluate the model PDF values for the data

        Parameters
        ----------
        params : `np.array`
            The input parameters

        Returns
        -------
        values : `np.array`
            The model pdf values
        """
        self._nmodel = np.sum(np.exp(params))/(self._discretized_redshift[-1] - self._discretized_redshift[0])
        self._redshift_distribution.update_objdata(dict(pdfs=np.exp(np.expand_dims(np.array(params), 0))))
        return self._redshift_distribution.pdf(self._discretized_redshift)

    def loglike(self, params):
        """ Return the extended log-likelihood given data, a model, model parameters

        Parameters
        ----------
        params : `np.array`
            The input parameters

        Returns
        -------
        loglike : `float`
            The log-likelhood

        Notes
        -----
        To keep overall sum of the parameters constrained, we add an "extended likelihood" term
        that consists of the Poisson likelihood term to over the actual number of counts observed,
        given the model parameters
        """
        integrand = self._post_vals * self.model_vals(params) / self._prior_vals
        lnlvals = np.log(np.trapz(integrand, self._discretized_redshift))
        # "Extended" term to constain the normalizaiton of the model
        ext_term = ((self._nmodel - self._npdf)**2)/(self._npdf)
        return -1*np.sum(lnlvals) + ext_term

    def __call__(self, params):
        """ Return the Poisson log-likelihood given data, a model, model parameters """
        return self.loglike(params)
