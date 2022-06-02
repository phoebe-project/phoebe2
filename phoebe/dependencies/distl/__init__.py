from . import distl as _distl
from .distl import DistributionCollection, from_dict, from_json, from_file, get_random_seed, _has_astropy, _units, __version__, version # , sample_from_dists, sample_ppf_from_dists, logp_from_dists, sample_func_from_dists, plot_func_from_dists,
import numpy as _np
from .distl import BaseDistlObject, BaseDistribution, BaseAroundGenerator # for isinstance checking
import json as _json

try:
    import dill as _dill
except ImportError:
    _has_dill = False
else:
    _has_dill = True

name = 'distl'

def delta(value=0.0, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Delta> distribution.

    Arguments
    --------------
    * `value` (float or int, default=0.0): the value at which the delta function is True.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Delta.wrap_at> and <Delta.wrap> for more details.

    Returns
    --------
    * a <Delta> object
    """
    return _distl.Delta(value, unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)


def uniform(low=0.0, high=1.0, unit=None, label=None, label_latex=None, wrap_at=None):
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
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Uniform.wrap_at> and <Uniform.wrap> for more details.

    Returns
    --------
    * a <Uniform> object
    """
    return _distl.Uniform(low, high, unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)

def boxcar(low=0.0, high=1.0, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Shortcut to <distl.uniform>.
    """
    return _distl.Uniform(low, high, unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)


def gaussian(loc=0.0, scale=1.0, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Gaussian> distribution.

    Arguments
    --------------
    * `loc` (float or int, default=0.0): the central value of the gaussian distribution.
    * `scale` (float or int, default=1.0): the scale (sigma) of the gaussian distribution.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Gaussian.wrap_at> and <Gaussian.wrap> for more details.

    Returns
    --------
    * a <Gaussian> object
    """
    return _distl.Gaussian(loc, scale, unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)

def normal(loc=0.0, scale=1.0, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Shortcut to <distl.gaussian>.
    """
    return _distl.Gaussian(loc, scale, unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)

def histogram_from_bins(bins, density, unit=None, label=None, label_latex=None, wrap_at=None):
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
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Histogram.wrap_at> and <Histogram.wrap> for more details.

    Returns
    --------
    * a <Histogram> object
    """
    return _distl.Histogram(bins, density, unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)

def histogram_from_data(data, bins=10, range=None, weights=None, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Histogram> distribution from data.

    See also:

    * <distl.histogram_from_bins>

    Arguments
    --------------
    * `data` (np.array object): 1D array of values.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float or False or None, optional, default=None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Histogram.wrap_at> and <Histogram.wrap> for more details.

    Returns
    --------
    * a <Histogram> object
    """

    return _distl.Histogram.from_data(data, bins=bins, range=range,
                                        weights=weights, unit=unit,
                                        label=label, label_latex=label_latex,
                                        wrap_at=wrap_at)

def samples(samples, weights=None, bw_method=None, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Samples> distribution.

    Arguments
    --------------
    * `samples` (np.array object): an array of samples.  Note that any Nans
        will be removed.
    * `weights` (np.array object with length nsamples or None, optional, default=None):
        weights for each entry in `samples`.  NOTE: only supported with
        scipy 1.2+.
    * `bw_method` (string, float, or None, optional, default=None): passed
        directly to scipy.stats.gaussian_kde.  Only used for methods that
        rely on the KDE.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float, None, or False, optional, default=None): value to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    --------
    * a <Samples> object
    """
    return _distl.Samples(samples, weights, bw_method,
                         unit=unit, label=label, label_latex=label_latex,
                         wrap_at=wrap_at)


def function(func, args=[], kwargs={}, vectorized=True, hist_samples=None, unit=None, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Function> distribution.


    Arguments
    ----------
    * `func`: callable function that accepts args and kwargs (as floats,
        after being sampled from any distribution objects)
    * `args` (list of distribution objects or floats): distribution objects
        or floats to pass as args to `func`.  Any items that are Distribution
        objects will be sampled and passed as floats.
    * `kwargs` (dictionary of distribution objects or floats): distribution
        objects or floats to pass as kwargs to `func`.  Any items that are
        Distribution objects will be sampled and passed as floats.
    * `vectorized` (bool, optional, default=True): whether `func` supports
        passing arrays to `args` and `kwargs`.
    * `hist_samples` (int, optional, default=None): number of samples to draw
        when generating the underlying <Histogram> distribution used for
        all probability calls.  If not provided or None, this will default
        to 1e6 if `vectorized` or 1e5 if not.  If `func` takes a long time
        or many samples in <Function.sample> are rejected and have to be
        re-drawn, it may be necessary to lower `hist_samples`.
    * `unit` (astropy.units object, optional): the units returned by `func`.
        Note that any Distribution objects in `args` and `kwargs` should be
        in the appropriate units (as the inputs and outputs to `func` are
        floats and not quantities)
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float, None, or False, optional, default=None): value to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    ---------
    * a <Function> object
    """
    return _distl.Function(func, args, kwargs, vectorized, hist_samples,
                           unit=unit, label=label, label_latex=label_latex, wrap_at=wrap_at)


#### MULTIVARIATE DISTRIBUTJIONS ####

def mvgaussian(mean, cov, allow_singular=False,
               units=None, labels=None, labels_latex=None, wrap_ats=None):
    """
    Create a <MVGaussian> distribution.

    Arguments
    --------------
    * `mean` (float or int, default=0.0): the central value of the
        multivariate gaussian distribution.
    * `cov` (float or int, default=1.0): the covariance matrix of the multivariate
        gaussian distribution.
    * `allow_singular` (bool, optional, default=False): passed directly to
        scipy (see link above).
    * `units` (list of astropy.units objects, optional): the units of the provided values.
    * `labels` (list of strings, optional): labels for each dimension in the
        distribution.  This is used
        for the x-labels while plotting the distribution when `labels_latex`
        is not provided, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `labels_latex` (list of strings, optional):  latex labels for each
        dimension in the distribution.  This is used for plotting the distribution.
    * `wrap_ats` (list of floats, None, or False, optional, default=None): values to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    --------
    * a <MVGaussian> object
    """
    return _distl.MVGaussian(mean, cov, allow_singular=allow_singular,
                             units=units, labels=labels, labels_latex=labels_latex, wrap_ats=wrap_ats)

def mvhistogram_from_data(data, bins=10, range=None, weights=None,
                          units=None, labels=None, labels_latex=None, wrap_ats=None):
    """
    Create a <MVHistogram> object from data.

    Arguments
    ------------
    * `data` (array): input array of samples.  Passed to
        [np.histogramdd](https://numpy.org/doc/1.18/reference/generated/numpy.histogramdd.html)
    * `bins` (integer or array, optional, default=10): number of bins or
        bin edges.  Passed to [np.histogramdd](https://numpy.org/doc/1.18/reference/generated/numpy.histogramdd.html)
    * `weights` (array, optional, default=None): weights for each entry
        in `data`.  Passed to [np.histogramdd](https://numpy.org/doc/1.18/reference/generated/numpy.histogramdd.html)
    * `units` (list of astropy.units objects, optional): the units of the provided values.
    * `labels` (list of strings, optional): labels for each dimension in the
        distribution.  This is used
        for the x-labels while plotting the distribution when `labels_latex`
        is not provided, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `labels_latex` (list of strings, optional):  latex labels for each
        dimension in the distribution.  This is used for plotting the distribution.
    * `wrap_ats` (list of floats, None, or False, optional, default=None): values to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    ----------
    * a <MVHistogram> object
    """
    return _distl.MVHistogram.from_data(data, bins=bins, range=range, weights=weights,
                                        units=units, labels=labels, labels_latex=labels_latex, wrap_ats=wrap_ats)


def mvsamples(samples, weights=None, bw_method=None,
              units=None, labels=None, labels_latex=None, wrap_ats=None):
    """
    Create a <MVSamples> distribution.

    Arguments
    --------------
    * `samples` (np.array object with shape (nsamples, <MVSamples.ndimensions>)):
        the samples.
    * `weights` (np.array object with shape (nsamples) or None, optional, default=None):
        weights for each entry in `samples`.  NOTE: only supported with scipy
        version 1.2+.
    * `bw_method` (string, float, or None, optional, default=None): passed
        directly to scipy.stats.gaussian_kde.  Only used for methods that
        rely on the KDE.
    * `units` (list of astropy.units objects, optional): the units of the provided values.
    * `labels` (list of strings, optional): labels for each dimension in the
        distribution.  This is used
        for the x-labels while plotting the distribution when `labels_latex`
        is not provided, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `labels_latex` (list of strings, optional):  latex labels for each
        dimension in the distribution.  This is used for plotting the distribution.
    * `wrap_ats` (list of floats, None, or False, optional, default=None): values to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    --------
    * an <MVSamples> object
    """
    return _distl.MVSamples(samples, weights, bw_method, units=units, labels=labels, labels_latex=labels_latex, wrap_ats=wrap_ats)

#### GENERATORS ####
def gaussian_around(scale, value=None,
                    unit=None, frac=False, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Gaussian_Around> object which, when called, will resolve
    to a <Gaussian> object around a given central value.

    Arguments
    --------------
    * `scale` (float or int, default=1.0): the scale (sigma) of the gaussian
        distribution.
    * `value` (float, optional, default=None): the current face-value.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `frac` (bool, optional, default=False): whether `scale` is provided as
        a fraction of `value` rather than in `unit`.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float, None, or False, optional, default=None): value to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.
    Returns
    --------
    * a <Gaussian_Around> object
    """

    return _distl.Gaussian_Around(scale, value, unit, frac, label, label_latex, wrap_at)

def uniform_around(width, value=None,
                   unit=None, frac=False, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Uniform_Around> object which, when called, will resolve
    to a <Uniform> object around a given central value.

    Arguments
    --------------
    * `width` (float): the width of the resulting <Uniform> object (<Uniform.low>
        and <Uniform.high> will be set based on the current value and `width`).
    * `value` (float, optional, default=None): the current face-value.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `frac` (bool, optional, default=False): whether `width` is provided as
        a fraction of `value` rather than in `unit`.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float, None, or False, optional, default=None): value to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    -----------
    * a <Uniform_Around> object.
    """

    return _distl.Uniform_Around(width, value, unit, frac, label, label_latex, wrap_at)

def delta_around(value=None, unit=None, frac=False, label=None, label_latex=None, wrap_at=None):
    """
    Create a <Delta_Around> object which, when called, will resolve
    to a <Delta> object around a given central value.

    Arguments
    --------------
    * `value` (float, optional, default=None): the current face-value.
    * `unit` (astropy.units object, optional): the units of the provided values.
    * `frac` (bool, optional, default=False): ignored as <Delta> has no width parameter.
    * `label` (string, optional): a label for the distribution.  This is used
        for the x-label while plotting the distribution if `label_latex` is not provided,
        as well as a shorthand notation when creating a <Composite> distribution.
    * `label_latex` (string, optional): a latex label for the distribution.  This is used
        for the x-label while plotting.
    * `wrap_at` (float, None, or False, optional, default=None): value to
        use for wrapping.  If None and `unit` are angles, will default to
        2*pi (or 360 degrees).  If None and `unit` are cycles, will default
        to 1.0.

    Returns
    --------
    * a <Delta> object
    """
    return _distl.Delta_Around(value, unit, frac, label, label_latex, wrap_at)
