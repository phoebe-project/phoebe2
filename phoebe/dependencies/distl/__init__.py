from . import distl as _distl
from .distl import DistributionCollection, from_dict, from_json, from_file, get_random_seed, _has_astropy, _units, __version__, version # , sample_from_dists, sample_ppf_from_dists, logp_from_dists, sample_func_from_dists, plot_func_from_dists,
import numpy as _np
from .distl import BaseDistribution # for isinstance checking
import json as _json

try:
    import dill as _dill
except ImportError:
    _has_dill = False
else:
    _has_dill = True

name = 'distl'

def delta(value=0.0, unit=None, label=None, wrap_at=None):
    """
    Create a <Delta> distribution.

    Arguments
    --------------
    * `value` (float or int, default=0.0): the value at which the delta function is True.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Delta.wrap_at> and <Delta.wrap> for more details.

    Returns
    --------
    * a <Delta> object
    """
    return _distl.Delta(value, unit=unit, label=label, wrap_at=wrap_at)


def uniform(low=0.0, high=1.0, unit=None, label=None, wrap_at=None):
    """
    Create a <Uniform> distribution.

    Arguments
    --------------
    * `low` (float or int, default=0.0): the lower limit of the uniform distribution.
    * `high` (float or int, default=1.0): the upper limits of the uniform distribution.
        Must be higher than `low` unless `wrap_at` is provided or `unit`
        is provided as angular (rad, deg, cycles).
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Uniform.wrap_at> and <Uniform.wrap> for more details.

    Returns
    --------
    * a <Uniform> object
    """
    return _distl.Uniform(low, high, unit=unit, label=label, wrap_at=wrap_at)

def boxcar(low=0.0, high=1.0, unit=None, label=None, wrap_at=None):
    """
    Shortcut to <distl.uniform>.
    """
    return _distl.Uniform(low, high, unit=unit, label=label, wrap_at=wrap_at)


def gaussian(loc=0.0, scale=1.0, unit=None, label=None, wrap_at=None):
    """
    Create a <Gaussian> distribution.

    Arguments
    --------------
    * `loc` (float or int, default=0.0): the central value of the gaussian distribution.
    * `scale` (float or int, default=1.0): the scale (sigma) of the gaussian distribution.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Gaussian.wrap_at> and <Gaussian.wrap> for more details.

    Returns
    --------
    * a <Gaussian> object
    """
    return _distl.Gaussian(loc, scale, unit=unit, label=label, wrap_at=wrap_at)

def normal(loc=0.0, scale=1.0, unit=None, label=None, wrap_at=None):
    """
    Shortcut to <distl.gaussian>.
    """
    return _distl.Gaussian(loc, scale, unit=unit, label=label, wrap_at=wrap_at)

def histogram_from_bins(bins, density, unit=None, label=None, wrap_at=None):
    """
    Create a <Histogram> distribution from binned data.

    See also:

    * <distl.histogram_from_data>

    Arguments
    --------------
    * `bins` (np.array object): the value of the bin-edges.  Must have one more
        entry than `density`.
    * `density` (np.array object): the value of the bin-densities.  Must have one
        less entry than `bins`.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Histogram.wrap_at> and <Histogram.wrap> for more details.

    Returns
    --------
    * a <Histogram> object
    """
    return _distl.Histogram(bins, density, unit=unit, label=label, wrap_at=wrap_at)

def histogram_from_data(data, bins=10, range=None, weights=None, unit=None, label=None, wrap_at=None):
    """
    Create a <Histogram> distribution from data.

    See also:

    * <distl.histogram_from_bins>

    Arguments
    --------------
    * `data` (np.array object): 1D array of values.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Histogram.wrap_at> and <Histogram.wrap> for more details.

    Returns
    --------
    * a <Histogram> object
    """

    return _distl.Histogram.from_data(data, bins=bins, range=range,
                                        weights=weights, unit=unit, label=label,
                                        wrap_at=wrap_at)


#### MULTIVARIATE DISTRIBUTJIONS ####

def mvgaussian(mean, cov, allow_singular=False,
               units=None, labels=None, wrap_ats=None):
    """
    Create a <MVGaussian> distribution.

    """
    return _distl.MVGaussian(mean, cov, allow_singular=allow_singular,
                             units=units, labels=labels, wrap_ats=wrap_ats)

def mvhistogram_from_data(data, bins=10, range=None, weights=None,
                          units=None, labels=None, wrap_ats=None):
    """
    """
    return _distl.MVHistogram.from_data(data, bins=bins, range=range, weights=weights,
                                        units=units, labels=labels, wrap_ats=wrap_ats)
