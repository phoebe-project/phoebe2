from . import npdists as _npdists
from .npdists import get_random_seed, sample_from_dists, sample_ppf_from_dists, logp_from_dists, sample_func_from_dists, plot_func_from_dists, _has_astropy, _units
import numpy as _np
from .npdists import BaseDistribution # for isinstance checking
import json as _json

try:
    import dill as _dill
except ImportError:
    _has_dill = False
else:
    _has_dill = True

name = 'npdists'
__version__ = '0.1.0-dev'
version = __version__



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
    return _npdists.Delta(value, unit=unit, label=label, wrap_at=wrap_at)


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
    return _npdists.Uniform(low, high, unit=unit, label=label, wrap_at=wrap_at)

def boxcar(low=0.0, high=1.0, unit=None, label=None, wrap_at=None):
    """
    Shortcut to <npdists.uniform>.
    """
    return _npdists.Uniform(low, high, unit=unit, label=label, wrap_at=wrap_at)


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
    return _npdists.Gaussian(loc, scale, unit=unit, label=label, wrap_at=wrap_at)

def normal(loc=0.0, scale=1.0, unit=None, label=None, wrap_at=None):
    """
    Shortcut to <npdists.gaussian>.
    """
    return _npdists.Gaussian(loc, scale, unit=unit, label=label, wrap_at=wrap_at)

def histogram_from_bins(bins, density, unit=None, label=None, wrap_at=None):
    """
    Create a <Histogram> distribution from binned data.

    See also:

    * <npdists.histogram_from_data>

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
    return _npdists.Histogram(bins, density, unit=unit, label=label, wrap_at=wrap_at)

def histogram_from_data(data, bins=10, range=None, weights=None, unit=None, label=None, wrap_at=None):
    """
    Create a <Histogram> distribution from data.

    See also:

    * <npdists.histogram_from_bins>

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

    return _npdists.Histogram.from_data(data, bins=bins, range=range,
                                        weights=weights, unit=unit, label=label,
                                        wrap_at=wrap_at)


def function(func, unit, label, wrap_at, *args):
    """
    Create a <Function> distribution from some callable function and
    any number of arguments, including distribution objects.


    Arguments
    ----------
    * `func` (callable function): the callable function to be called to
        sample the distribution.
    * `unit` (astropy.units object or None): the units of the provided values.
    * `label` (string or None): a label for the distribution.  This is used
        for the x-label while plotting the distribution, as well as a shorthand
        notation when creating a <Composite> distribution.
    * `wrap_at` (float or False or None): value to wrap all
        sampled values.  If None, will default to 0-2pi if `unit` is angular
        (0-360 for degrees), or 0-1 if `unit` is cycles.  If False, will not wrap.
        See <Function.wrap_at> and <Function.wrap> for more details.
    * `*args`: all additional positional arguments will be passed on to
        `func` when sampling.  These can be, but are not limited to,
        other distribution objects.

    Returns
    ---------
    * a <Function> object.
    """
    return _npdists.Function(func, unit, label, wrap_at, *args)


#### MULTIVARIATE DISTRIBUTJIONS ####

def mvgaussian(locs, cov, unit=None, label=None, wrap_at=None):
    """
    Create a <MVGaussian> distribution.

    """
    return _npdists.MVGaussian(locs, cov, unit=unit, label=label, wrap_at=wrap_at)

def mvhistogram_from_data(data, bins=10, range=None, weights=None, unit=None, label=None, wrap_at=None):
    """
    """
    return _npdists.MVHistogram.from_data(data, bins=bins, range=range, weights=weights, unit=unit, label=label, wrap_at=wrap_at)



def from_dict(d):
    """
    Load an npdists object from a dictionary.

    See also:

    * <npdists.from_json>
    * <npdists.from_file>

    Arguments
    -------------
    * `d` (string or dict): dictionary (or json string of a dictionary)
        representing the npdists object.

    Returns
    ----------
    * The appropriate distribution object.
    """
    if isinstance(d, str):
        return from_json(d)

    if not isinstance(d, dict):
        raise TypeError("argument must be of type dict")
    if 'npdists' not in d.keys():
        raise ValueError("input dictionary missing 'nparray' entry")

    classname = d.get('npdists').title()
    unit = d.pop('unit', None)
    # instead of popping npdists (which would happen in memory and make that json
    # unloadable again), we'll do a dictionary comprehension.  If this causes
    # performance issues, we could instead accept and ignore npdists as
    # a keyword argument to __init__
    dist = getattr(_npdists, classname)(**{k:v for k,v in d.items() if k!='npdists'})
    if unit is not None and _has_astropy:
        dist *= _units.Unit(unit)
    return dist

def from_json(j):
    """
    Load an npdists object from a json-formatted string.

    See also:

    * <npdists.from_dict>
    * <npdists.from_file>

    Arguments
    -------------
    * `s` (string or dict): json formatted dictionary representing the npdists
        object.

    Returns
    ----------
    * The appropriate distribution object.
    """
    if isinstance(j, dict):
        return from_dict(j)

    if not (isinstance(j, str) or isinstance(j, unicode)):
        raise TypeError("argument must be of type str")

    return from_dict(_json.loads(j))

def from_file(filename):
    """
    Load an npdists object from a json filename.

    See also:

    * <npdists.from_dict>
    * <npdists.from_json>

    Arguments
    -------------
    * `s` (string): the filename pointing to a json formatted file representing
        an npdists object.

    Returns
    ----------
    * The appropriate distribution object.
    """
    f = open(filename, 'r')
    try:
        j = _json.load(f)
    except:
        f.close()
        if _has_dill:
            f = open(filename, 'rb')
            d = _dill.load(f)
            f.close()
            return d
        else:
            raise ImportError("file requires 'dill' package to load")
    else:
        f.close()
        return from_dict(j)
