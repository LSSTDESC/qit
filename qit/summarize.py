"""Ensemble summarization functions"""

from collections import OrderedDict
import numpy as np
import qp


def sum_gridded(ens, out_grid):    
    summed = np.sum(ens.pdf(out_grid), axis=0) / ens.npdf
    return summed
    
def histogram_samples(ens, out_bins, n_samples=1):
    samples = ens.rvs(size=n_samples)
    hist = np.histogram(samples, out_bins)
    return hist[0]/samples.size

def histogram_modes(ens, out_bins):
    modes = ens.mode(grid=ens.dist.xvals)
    hist = np.histogram(modes, out_bins)
    return hist[0]/modes.size

def vstack_estimates(ens, mask_list, sum_func, *args):
    out_l = []
    n_pdf = []
    for mask in mask_list:
        ens_r = ens[mask]
        out_l.append(sum_func(ens_r, *args))
        n_pdf.append(mask.sum())
    return np.vstack(out_l), np.array(n_pdf)

def stack_ens(ens, mask_list, out_grid=None):    
    if out_grid is None:
        out_grid = ens.dist.xvals
    yvals, npdf = vstack_estimates(ens, mask_list, sum_gridded, out_grid)
    ens = qp.Ensemble(qp.interp, data=dict(xvals=out_grid, yvals=yvals))
    ens.set_ancil(dict(npdf=npdf))
    return ens

def sample_ens(ens, mask_list, out_bins=None, n_samples=1):
    if out_bins is None:
        out_bins = ens.dist.xvals
    pdfs, npdf = vstack_estimates(ens, mask_list, histogram_samples, out_bins, n_samples)
    ens = qp.Ensemble(qp.hist, data=dict(bins=out_bins, pdfs=pdfs))
    ens.set_ancil(dict(npdf=npdf))
    return ens
   
def modes_ens(ens, mask_list, out_bins=None):
    if out_bins is None:
        out_bins = ens.dist.xvals
    pdfs, npdf = vstack_estimates(ens, mask_list, histogram_modes, out_bins)
    ens = qp.Ensemble(qp.hist, data=dict(bins=out_bins, pdfs=pdfs))
    ens.set_ancil(dict(npdf=npdf))
    return ens
   
def weighted_sum_hist(ens):
    npdfs = ens.ancil['npdf']
    out_pdfs = np.expand_dims(np.sum(ens.hpdfs*np.expand_dims(npdfs, -1), axis=0), -1)
    return qp.Ensemble(qp.hist, data=dict(bins=ens.dist.hbins, pdfs=out_pdfs))

def weighted_sum_interp(ens):
    npdfs = ens.ancil['npdf']
    out_yvals = np.expand_dims(np.sum(ens.yvals*np.expand_dims(npdfs, -1), axis=0), -1)
    return qp.Ensemble(qp.interp, data=dict(xvals=ens.dist.xvals, yvals=ens.dist.yvals))

def simple_summarize(ens, mask_list, extrac_func, summing_func, **kwargs):
    binned_ensemble = extrac_func(ens, mask_list, **kwargs)
    summed_ensemble = summing_func(binned_ensemble)
    return OrderedDict([(binned, binned_ensemble),
                        (summed, summed_ensemble)])

def write_odict(base_name, odict):
    for k, v in odict.items():
        outfile = "%s_%s.hdf5" % (base_name, k)
        v.write_to(outfile)

def read_to_odict(base_name):
    keys = ["binned", "summed"]
    o_dict = OrderedDict([(key, qp.read("%s_%s.hdf5" % (base_name, key))) for key in keys])
    return o_dict




