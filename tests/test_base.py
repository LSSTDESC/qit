"""
Unit tests for PDF class
"""
import numpy as np
import unittest
import qp
import qit
from scipy.optimize import minimize

# true distribution of redshifts
Z_TRUE_MIN, Z_TRUE_MAX = 0., 2.
LOC_TRUE = 0.60
SCALE_TRUE = 0.30
LOC_PRIOR = 0.65
SCALE_PRIOR = 0.35
N_EST_BINS = 50
N_OBS_BINS = 300
Z_OBS_MIN, Z_OBS_MAX = -0.5, 2.5
N_SAMPLES = 10000
N_HIST_BINS = 50
N_OBS_HIST_BINS = 75
N_EVAL_PTS = 201
N_LIKE_PTS = 301
N_TRUE_BINS = 10
N_PROF_BINS = 10
N_FIT_BINS = 10

class BaseTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """

    def tearDown(self):
        "Clean up any mock data files created by the tests."
        
    def test_qit_example(self):
        
        true_dist = qp.Ensemble(qp.stats.norm, data=dict(loc=LOC_TRUE, scale=SCALE_TRUE))
        implicit_prior = qp.Ensemble(qp.stats.norm, data=dict(loc=LOC_PRIOR, scale=SCALE_PRIOR))
        z_bins = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_EST_BINS+1)
        z_centers = qp.utils.edge_to_center(z_bins)
        z_widths = 0.2 * np.ones(N_EST_BINS)
        likelihood = qp.Ensemble(qp.stats.norm, data=dict(loc=np.expand_dims(z_centers, -1), scale=np.expand_dims(z_widths, -1)))
        grid_edge = np.linspace(Z_OBS_MIN, Z_OBS_MAX, N_OBS_BINS+1)
        grid_cent = qp.utils.edge_to_center(grid_edge)
        p_grid = likelihood.pdf(grid_cent)
        z_grid = z_centers
        flat_post = qp.Ensemble(qp.stats.hist, data=dict(bins=z_bins, pdfs=p_grid.T))
        
        post_grid = qit.like_funcs.get_posterior_grid(flat_post, z_grid)
        est_grid = qit.like_funcs.get_posterior_grid(flat_post, z_grid, implicit_prior)
        true_grid = qit.like_funcs.get_posterior_grid(flat_post, z_grid, true_dist)
        z_true_sample = np.squeeze(true_dist.rvs(size=N_SAMPLES))
        
        whichbin = np.searchsorted(z_bins, z_true_sample)-1
        mask = (z_true_sample > 0) * (z_true_sample <= 2.0)
        mask *= (whichbin < z_centers.size)
        whichbin = whichbin[mask]

        sampler = qp.Ensemble(qp.stats.norm, data=dict(loc=np.expand_dims(z_centers[whichbin], -1), scale=np.expand_dims(z_widths[whichbin], -1)))
        z_meas_sample = np.squeeze(sampler.rvs(1))
        x_prof = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_PROF_BINS+1)
        x_prof_cent = qp.utils.edge_to_center(x_prof)
        prof_vals, prof_errs = qp.utils.profile(z_true_sample[mask], z_meas_sample, x_prof)

        z_meas_bin = np.searchsorted(grid_edge, z_meas_sample)-1
        z_meas_mask = (z_meas_bin >= 0) * (z_meas_bin < grid_cent.size)
        z_meas_bin = z_meas_bin[z_meas_mask]
        
        post_dict = qit.like_funcs.make_ensemble_for_posterior_interp(post_grid, z_grid, z_meas_bin)
        est_dict = qit.like_funcs.make_ensemble_for_posterior_interp(est_grid, z_grid, z_meas_bin)
        true_dict = qit.like_funcs.make_ensemble_for_posterior_interp(true_grid, z_grid, z_meas_bin)

        eval_grid = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_EVAL_PTS)

        N_FIT_BINS = 4
        hist_bins = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_FIT_BINS+1)

        model_params = np.log(np.ones(N_FIT_BINS))
        model = qp.Ensemble(qp.stats.hist, data=dict(bins=hist_bins, pdfs=model_params))

        hist_cents = qp.utils.edge_to_center(hist_bins)
        true_vals = np.histogram(z_true_sample, bins=np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_FIT_BINS+1))[0]
        v_flat = qit.like_funcs.log_hyper_like(model_params, est_dict['ens'], model, implicit_prior, eval_grid)
        v_true = qit.like_funcs.log_hyper_like(np.log(true_vals), est_dict['ens'], model, implicit_prior, eval_grid)

        obj_func = qit.like_funcs.make_log_hyper_obj_func(ensemble=est_dict['ens'],\
                   model=model, implicit_prior=implicit_prior, grid=eval_grid)

        result = minimize(obj_func, model_params)
        fitted_vals = np.exp(result['x'])
        fitted_errs = np.sqrt(np.array([result['hess_inv'][i,i] for i in range(4)]))
        norm_factor = 2 / fitted_vals.sum()
        normed_fit = norm_factor * fitted_vals
        jac = fitted_vals
        # Convert to PDF, for plotting
        normed_errs = norm_factor * jac * fitted_errs
        model.update_objdata(dict(pdfs=np.expand_dims(normed_fit, 0)))
        model_vals = np.squeeze(model.pdf(z_grid))

        like_grid = np.linspace(Z_OBS_MIN, Z_OBS_MAX, N_LIKE_PTS)
        eval_bins = np.searchsorted(z_bins, eval_grid, side='left')-1
        eval_mask = (eval_bins >= 0) * (eval_bins < z_bins.size-1)
        eval_grid = eval_grid[eval_mask]
        eval_bins = eval_bins[eval_mask]
        like_eval = likelihood.pdf(like_grid)[eval_bins]
        obs_cts_grid = np.linspace(Z_OBS_MIN, Z_OBS_MAX, 7)
        data_cts = np.histogram(z_meas_sample, bins=obs_cts_grid)[0]
        
        obj_func_binned = qit.like_funcs.make_binnned_loglike_obj_func(model=model, data_cts=data_cts,
                                                                       like_eval=like_eval, like_grid=like_grid,
                                                                       model_grid=eval_grid, cts_grid=obs_cts_grid)

        flat = 0.5*data_cts.sum()*np.ones(4)
        model_flat = qit.like_funcs.model_counts(np.log(flat), model, like_eval, like_grid, eval_grid, obs_cts_grid)
        model_true = qit.like_funcs.model_counts(np.log(true_vals), model, like_eval, like_grid, eval_grid, obs_cts_grid)
        ll_flat = obj_func_binned(np.log(flat))
        ll_true = obj_func_binned(np.log(true_vals))

        result = minimize(obj_func_binned, np.ones(4))
        model_cts = qit.like_funcs.model_counts(result['x'], model, like_eval, like_grid, eval_grid, obs_cts_grid)
        cts_cent = 0.5 * (obs_cts_grid[1:] + obs_cts_grid[:-1])

        fit_cts = np.exp(result['x'])
        fit_cts *= 2/fit_cts.sum()
        pdf_true = true_vals * 2 / true_vals.sum()


    def test_qit_binned(self):
        true_bins = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_TRUE_BINS+1)
        
        true_dist_norm = qp.Ensemble(qp.stats.norm, data=dict(loc=[[LOC_TRUE]], scale=[[SCALE_TRUE]]))
        true_dist = qp.convert(true_dist_norm, 'hist', bins=true_bins)

        implicit_prior_norm = qp.Ensemble(qp.stats.norm, data=dict(loc=[[LOC_PRIOR]], scale=[[SCALE_PRIOR]]))
        implicit_prior = qp.convert(implicit_prior_norm, 'hist', bins=true_bins)

        z_bins = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_TRUE_BINS+1)
        z_centers = qp.utils.edge_to_center(z_bins)
        z_widths = 0.2 * np.ones(N_TRUE_BINS)
        likelihood = qp.Ensemble(qp.stats.norm, data=dict(loc=np.expand_dims(z_centers, -1), scale=np.expand_dims(z_widths, -1)))

        grid_edge = np.linspace(Z_OBS_MIN, Z_OBS_MAX, N_OBS_BINS+1)
        grid_cent = qp.utils.edge_to_center(grid_edge)
        p_grid = likelihood.pdf(grid_cent)

        like_estim = qit.Estimator([z_bins], likelihood)
        flat_post = like_estim.flat_posterior(grid_edge)

        z_grid = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, 101)
        post_grid = flat_post.get_posterior_grid(z_grid)
        est_grid = flat_post.get_posterior_grid(z_grid, implicit_prior)
        true_grid = flat_post.get_posterior_grid(z_grid, true_dist)

        z_true_sample = np.squeeze(true_dist.rvs(size=N_SAMPLES))
        sampler = like_estim.get_sampler(z_true_sample)
        mask = (z_true_sample > 0) * (z_true_sample <= 2.0)
        z_meas_sample = np.squeeze(sampler.rvs(1))

        x_prof = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_PROF_BINS+1)
        x_prof_cent = qp.utils.edge_to_center(x_prof)
        prof_vals, prof_errs = qp.utils.profile(z_true_sample[mask], z_meas_sample, x_prof)

        z_meas_bin = np.searchsorted(grid_edge, z_meas_sample)
        z_meas_mask = (z_meas_bin >= 0) * (z_meas_bin < grid_cent.size)
        z_meas_bin = z_meas_bin[z_meas_mask]

        post_ens = like_estim.make_posterior_ensemble(z_grid, z_true_sample)
        est_ens = like_estim.make_posterior_ensemble(z_grid, z_true_sample, implicit_prior)
        true_ens = like_estim.make_posterior_ensemble(z_grid, z_true_sample, true_dist)

        def make_dict(ens, z_grid):
            vals = ens.pdf(z_grid)
            return dict(ens=ens, vals=vals, stack=vals.mean(axis=0))

        post_dict_o = qit.like_funcs.make_ensemble_for_posterior_interp(post_grid, z_grid, z_meas_bin)
        est_dict_o = qit.like_funcs.make_ensemble_for_posterior_interp(est_grid, z_grid, z_meas_bin)
        true_dict_o = qit.like_funcs.make_ensemble_for_posterior_interp(true_grid, z_grid, z_meas_bin)
        post_dict_i = make_dict(post_ens, z_grid)
        est_dict_i = make_dict(est_ens, z_grid)
        true_dict_i = make_dict(true_ens, z_grid)

        post_dict = post_dict_i
        est_dict = est_dict_i
        true_dict = true_dict_i

        model_params = np.ones((1, N_TRUE_BINS))
        model = qp.Ensemble(qp.stats.hist, data=dict(bins=true_bins, pdfs=model_params))
        
        eval_grid = np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_EVAL_PTS)
        model_params = np.log(np.ones(N_TRUE_BINS))
        hist_cents = qp.utils.edge_to_center(z_grid)
        true_vals = np.histogram(z_true_sample, bins=np.linspace(Z_TRUE_MIN, Z_TRUE_MAX, N_TRUE_BINS+1))[0]
        v_flat = qit.like_funcs.log_hyper_like(model_params, est_dict['ens'], model, implicit_prior, eval_grid)
        v_true = qit.like_funcs.log_hyper_like(np.log(true_vals), est_dict['ens'], model, implicit_prior, eval_grid)

        obj_func = qit.like_funcs.make_log_hyper_obj_func(ensemble=est_dict['ens'],\
                                                          model=model, implicit_prior=implicit_prior, grid=eval_grid)

        result = minimize(obj_func, model_params)
        fitted_vals = np.exp(result['x'])
        fitted_errs = np.sqrt(np.array([result['hess_inv'][i,i] for i in range(N_FIT_BINS)]))
        norm_factor = 2 / fitted_vals.sum()
        normed_fit = norm_factor * fitted_vals
        jac = fitted_vals
        normed_errs = norm_factor * jac * fitted_errs
        model.update_objdata(dict(pdfs=np.expand_dims(normed_fit, 0)))
        model_vals = np.squeeze(model.pdf(z_grid))

        like_grid = np.linspace(Z_OBS_MIN, Z_OBS_MAX, N_LIKE_PTS)
        eval_bins = np.searchsorted(z_bins, eval_grid, side='left')
        eval_mask = (eval_bins >= 0) * (eval_bins < z_bins.size-1)
        eval_grid = eval_grid[eval_mask]
        eval_bins = eval_bins[eval_mask]
        like_eval = likelihood.pdf(like_grid)[eval_bins]
        obs_cts_grid = np.linspace(Z_OBS_MIN, Z_OBS_MAX, N_OBS_BINS+1)
        data_cts = np.histogram(z_meas_sample, bins=obs_cts_grid)[0]
        
        obj_func_binned = qit.like_funcs.make_binnned_loglike_obj_func(model=model, data_cts=data_cts,
                                                                       like_eval=like_eval, like_grid=like_grid,
                                                                       model_grid=eval_grid, cts_grid=obs_cts_grid)

        flat = 0.5*data_cts.sum()*np.ones(N_TRUE_BINS)
        model_flat = qit.like_funcs.model_counts(np.log(flat), model, like_eval, like_grid, eval_grid, obs_cts_grid)
        model_true = qit.like_funcs.model_counts(np.log(true_vals), model, like_eval, like_grid, eval_grid, obs_cts_grid)
        ll_flat = obj_func_binned(np.log(flat))
        ll_true = obj_func_binned(np.log(true_vals))
        result = minimize(obj_func_binned, np.ones(N_TRUE_BINS))
        model_cts = qit.like_funcs.model_counts(result['x'], model, like_eval, like_grid, eval_grid, obs_cts_grid)
        cts_cent = 0.5 * (obs_cts_grid[1:] + obs_cts_grid[:-1])
        fit_cts = np.exp(result['x'])
        true_cents = qp.utils.edge_to_center(true_bins)
        pdf_true = true_vals * 2 / true_vals.sum()

if __name__ == '__main__':
    unittest.main()
