import numpy as _np
from scipy import stats as _stats
from scipy import interpolate as _interpolate
from scipy import integrate as _integrate
import json as _json
import sys as _sys
from collections import OrderedDict

from . import stats_custom as _stats_custom

if _sys.version_info[0] > 2:
    unicode = str

try:
    import matplotlib.pyplot as _plt
except ImportError:
    _has_mpl = False
else:
    _has_mpl = True

try:
    import corner
except ImportError:
    _has_corner = False
else:
    _has_corner = True

try:
    from astropy import units as _units
except ImportError:
    _has_astropy = False
else:
    _has_astropy = True

__version__ = '0.1.0.dev1'
version = __version__

_math_symbols = {'__mul__': '*', '__add__': '+', '__sub__': '-', '__div__': '/', '__and__': '&', '__or__': '|'}

_builtin_attrs = ['unit', 'label', 'wrap_at', 'dimension', 'dist_constructor_argnames', 'dist_constructor_args', 'dist_constructor_func', 'dist_constructor_object']

_physical_types_to_si = {'length': 'solRad',
                         'mass': 'solMass',
                         'temperature': 'solTeff',
                         'power': 'solLum',
                         'time': 'd',
                         'speed': 'solRad/d',
                         'angle': 'rad',
                         'angular speed': 'rad/d',
                         'dimensionless': ''}

_physical_types_to_solar = {'length': 'm',
                            'mass': 'kg',
                            'temperature': 'K',
                            'power': 'W',
                            'time': 's',
                            'speed': 'm/s',
                            'angle': 'rad',
                            'angular speed': 'rad/s',
                            'dimensionless': ''}

########################## LOAD/SAVE FUNCTIONS #################################


def from_dict(d):
    """
    Load a distl object from a dictionary.

    See also:

    * <distl.from_json>
    * <distl.from_file>

    Arguments
    -------------
    * `d` (string or dict): dictionary (or json string of a dictionary)
        representing the distl object.

    Returns
    ----------
    * The appropriate distribution object.
    """
    if isinstance(d, str) or isinstance(d, unicode):
        return from_json(d)

    if not isinstance(d, dict):
        raise TypeError("argument must be of type dict")
    if 'distl' not in d.keys():
        raise ValueError("input dictionary missing 'distl' entry")

    classname = d.get('distl')
    # unit = d.pop('unit', None)
    # instead of popping distl (which would happen in memory and make that json
    # unloadable again), we'll do a dictionary comprehension.  If this causes
    # performance issues, we could instead accept and ignore distl as
    # a keyword argument to __init__
    args = d.get('args', None)
    kwargs = {k:v for k,v in d.items() if k not in ['distl', 'distl.version', 'args']}
    if args is not None:
        dist = getattr(_sys.modules[__name__], classname)(*args, **kwargs)
    else:
        dist = getattr(_sys.modules[__name__], classname)(**kwargs)
    # if unit is not None and _has_astropy:
        # dist *= _units.Unit(unit)
    return dist

def from_json(j):
    """
    Load a distl object from a json-formatted string.

    See also:

    * <distl.from_dict>
    * <distl.from_file>

    Arguments
    -------------
    * `s` (string or dict): json formatted dictionary representing the distl
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
    Load a distl object from a json filename.

    See also:

    * <distl.from_dict>
    * <distl.from_json>

    Arguments
    -------------
    * `s` (string): the filename pointing to a json formatted file representing
        a distl object.

    Returns
    ----------
    * The appropriate distribution object.
    """
    f = open(filename, 'r')
    j = _json.load(f)
    f.close()
    return from_dict(j)



############################# HELPER FUNCTIONS #################################

def get_random_seed():
    """
    Return a random seed which can be passed to <BaseDistribution.sample>.

    This allows for using a consistent/reproducible but still random seed instead
    of manually passing some arbitrary integer (like 1234).

    Returns
    ------------
    * (array): array of 624 32-bit integers which can be used as a seed to
        np.random.seed or <BaseDistribution.sample>.
    """
    return _np.random.get_state()[1]

def to_unit(unit):
    """
    Convert a string to an astropy unit object
    """
    if isinstance(unit, str) or isinstance(unit, unicode):
        unit = _units.Unit(unit)

    return unit

def _json_safe(v):
    if isinstance(v, _np.ndarray):
        return v.tolist()
    elif isinstance(v, list) or isinstance(v, tuple):
        return [_json_safe(li) for li in v]
    elif isinstance(v, BaseDistribution):
        return v.to_dict()
    # elif _is_unit(v):
    #     return v.to_string()
    else:
        return v

def _all_in_types(objects, types):
    return _np.all([_np.any([isinstance(o, t) for t in types]) for o in objects])

def _any_in_types(objects, types):
    return _np.any([_np.any([isinstance(o, t) for t in types]) for o in objects])

################## VALIDATORS ###################

# these all must accept a single value and return a boolean if it matches the condition as well as any alterations to the value
# NOTE: the docstring is used as the error message if the test fails

def is_distribution(value):
    if isinstance(value, dict) and 'distl' in value.keys():
        return from_dict(value)
    if isinstance(value, BaseDistribution):
        return value
    raise TypeError('must be a distl Distribution object')

def is_distribution_univariate_or_slice(value):
    value = is_distribution(value)
    if isinstance(value, BaseUnivariateDistribution) or isinstance(value, BaseMultivariateSliceDistribution):
        return value
    raise TypeError('must be a Univariate or MultivariateSlice distl Distribution object')

def is_distribution_or_none(value):
    if value is None:
        return value

    try:
        return is_distribution(value)
    except TypeError:
        raise TypeError('must be a distl Distribution object or None')

def is_math(value):
    valid_maths = ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__div__', '__rdiv__']
    valid_maths += ['sin', 'cos', 'tan']
    valid_maths += ['__and__', '__or__']
    if value in valid_maths:
        return value
    raise TypeError('must be a valid math operator (one of {})'.format(valid_maths))

def _is_unit(value):
    if not _has_astropy:
        raise ImportError("astropy must be installed for unit support")
    return (isinstance(value, _units.Unit) or isinstance(value, _units.IrreducibleUnit) or isinstance(value, _units.CompositeUnit))


def is_unit(value):
    if _is_unit(value):
        return value
    raise TypeError('must be an astropy unit')

def is_unit_or_unitstring(value):
    try:
        return to_unit(value)
    except:
        raise TypeError('must be an astropy unit or valid string')

def is_unit_or_unitstring_or_none(value):
    if value is None:
        return value
    return is_unit_or_unitstring(value)

def is_bool(value):
    if isinstance(value, bool):
        return value
    raise TypeError("must be boolean")

def is_float(value):
    try:
        value = float(value)
    except:
        raise TypeError("must be a float")
    else:
        return value

def is_int(value):
    if isinstance(value, int):
        return value
    raise TypeError('must be an integer')

def is_int_positive(value):
    if isinstance(value, int) and value > 0:
        return value
    raise TypeError("must be a positive integer")

def is_int_positive_or_none(value):
    if value is None:
        return value
    return is_int_positive(value)

def is_iterable(value):
    """must be an iterable (list, array, tuple)"""
    if _all_in_types((value,), (_np.ndarray, list, tuple)):
        return value
    raise TypeError("must be a numpy array, list, or tuple")

def is_1d_array(value):
    try:
        value = _np.asarray(value)
    except:
        raise TypeError("must be a 1d array")
    else:
        if len(value.shape)==1:
            return value
        raise TypeError("must be a 1d array")

def is_nd_array(value):
    try:
        value = _np.asarray(value)
    except:
        raise TypeError("must be an nd array")
    else:
        if len(value.shape) > 1:
            return value
        raise TypeError("must be a 1d array")

def is_square_matrix(value):
    if isinstance(value, list):
        value = _np.asarray(value)

    if isinstance(value, _np.ndarray) and len(value.shape)==2 and value.shape[0]==value.shape[1]:
        return value
    raise TypeError("must be a square matrix")


################################################################################

def _hist_pdf_cdf_ppf_callables(bins, density):
    bincenters = _np.mean(_np.vstack([bins[0:-1], bins[1:]]), axis=0)
    pdf_call = _stats_custom.interpolate_callable(bincenters, density)

    bincenters = _np.mean(_np.vstack([bins[0:-1], bins[1:]]), axis=0)
    cdf = _np.cumsum(density)
    cdf = cdf / float(cdf[-1])

    ppf_call = _stats_custom.interpolate_callable(cdf, bincenters)

    # make sure interpolation on the right always gives 1, not the fill_value of 0
    cdf_call = _stats_custom.interpolate_callable( _np.append(bincenters, _np.inf), _np.append(cdf, 1.0))

    return pdf_call, cdf_call, ppf_call

######################## DISTRIBUTION ABSTRACT CLASS ###########################

class BaseDistribution(object):
    """
    BaseDistribution is the parent class for all distributions and should
    not be used directly by the user.

    Any subclass distribution should override the following:

    * <BaseDistribution.__init__>
    """
    def __init__(self, dist_constructor_func, dist_constructor_argnames, **kwargs):
        """
        BaseDistribution is the parent class for all distributions and should
        not be used directly by the user.

        Any subclass distribution should override the init but call this via
        super.  See <Gaussian.__init__> for an example for subclassing.
        """
        self._cached_sample = None

        self._dist_constructor_func = dist_constructor_func
        self._dist_constructor_argnames = dist_constructor_argnames

        self._dist_constructor_object_cache = None
        self._parents_with_constructor_object_cache = []

        self._descriptors = kwargs.pop('descriptors', list(kwargs.keys()))

        for k,v in kwargs.items():
            setattr(self, k, v)

    ### REPRESENTATIONS

    def __float__(self):
        """
        by default, have the float representation come from sampling, but
        subclasses can/should override this to be the central/median/mode if
        possible
        """
        return self.median()

    ### COPYING

    def __copy__(self):
        return self.__class__(**{k:v for k,v in self.to_dict().items() if k not in ['distl', 'distl.version']})

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self):
        """
        Make a copy of the distribution object.

        Returns
        ---------
        * a copy of the distribution object
        """
        return self.__copy__()

    ### IO

    @property
    def hash(self):
        """
        """
        # return hash(frozenset({k:v for k,v in self.to_dict().items() if k not in ['dimension']}))
        return hash(str({k:v for k,v in self.to_dict().items()}))

    def to_json(self, **kwargs):
        """
        Return the json representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <distl.from_json>.

        See also:

        * <<class>.to_dict>
        * <<class>.to_file>

        Arguments
        ---------
        * `**kwargs`: all keyword arguments will be sent to json.dumps

        Returns
        --------
        * string
        """
        return _json.dumps(self.to_dict(), ensure_ascii=True, **kwargs)

    def to_file(self, filename, **kwargs):
        """
        Save the json representation of the distribution object to a file.

        The resulting file can be restored to the original object
        via <distl.from_file>.

        See also:

        * <<class>.to_dict>
        * <<class>.to_json>

        Arguments
        ----------
        * `filename` (string): path to the file to be created (will overwrite
            if already exists)
        * `**kwargs`: all keyword arguments will be sent to json.dumps

        Returns
        --------
        * string: the filename
        """
        f = open(filename, 'w')
        f.write(self.to_json(**kwargs))
        f.close()
        return filename

    ### MATH AND COMPARISON OPERATORS

    def __lt__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return self.__float__() < other.__float__()
        return self.__float__() < other

    def __le__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return self.__float__() <= other.__float__()
        return self.__float__() <= other

    def __gt__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return self.__float__() > other.__float__()
        return self.__float__() > other

    def __ge__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return self.__float__() >= other.__float__()
        return self.__float__() >= other

    def __mul__(self, other):
        if _has_astropy and _is_unit(other):
            copy = self.copy()
            copy.unit = other
            return copy

        elif _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return Composite("__mul__", (self, other))
        elif isinstance(other, float) or isinstance(other, int):
            return self.__mul__(Delta(other))
        else:
            raise TypeError("cannot multiply {} by type {}".format(self.__class__.__name__, type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return Composite("__div__", (self, other))
        elif isinstance(other, float) or isinstance(other, int):
            return self.__div__(Delta(other))
        else:
            raise TypeError("cannot divide {} by type {}".format(self.__class__.__name__, type(other)))

    def __rdiv__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return Composite("__rdiv__", (self, other))
        elif isinstance(other, float) or isinstance(other, int):
            return self.__rdiv__(Delta(other))
        else:
            raise TypeError("cannot divide {} by type {}".format(self.__class__.__name__, type(other)))

    def __add__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return Composite("__add__", (self, other))
        elif isinstance(other, float) or isinstance(other, int):
            return self.__add__(Delta(other))
        else:
            raise TypeError("cannot add {} by type {}".format(self.__class__.__name__, type(other)))

    # def __radd__(self, other):
    #     return self.__add__(other)

    def __sub__(self, other):
        if _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            return Composite("__sub__", (self, other))
        elif isinstance(other, float) or isinstance(other, int):
            return self.__sub__(Delta(other))
        else:
            raise TypeError("cannot subtract {} by type {}".format(self.__class__.__name__), type(other))

    def __and__(self, other):
        if _any_in_types((self, other), (BaseMultivariateSliceDistribution,)):
            raise TypeError("cannot use & (and) logic with MultivariateSlice distributions as covariances can not be maintained.  Use to_univariate() before applying & (and) logic")

        if not _all_in_types((self, other), (BaseUnivariateDistribution,)):
            raise TypeError("cannot use & (and) logic between types {} and {}".format(self.__class__.__name__, type(other)))

        return Composite("__and__", (self, other))

    def __or__(self, other):
        if not _all_in_types((self, other), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            raise TypeError("cannot use | (or) logic between types {} and {}".format(self.__class__.__name__, type(other)))

        return Composite("__or__", (self, other))

    def sin(self):
        if not _all_in_types((self,), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            raise TypeError("cannot use sin with type {}".format(self.__class__.__name__))

        if self.unit is not None:
            dist = self.to(_units.rad)
        else:
            dist = self

        return Composite("sin", (dist,))

    def cos(self):
        if not _all_in_types((self,), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            raise TypeError("cannot use cos with type {}".format(self.__class__.__name__))

        if self.unit is not None:
            dist = self.to(_units.rad)
        else:
            dist = self

        return Composite("cos", (dist,))

    def tan(self):
        if not _all_in_types((self,), (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            raise TypeError("cannot use tan with type {}".format(self.__class__.__name__))

        if self.unit is not None:
            dist = self.to(_units.rad)
        else:
            dist = self

        return Composite("tan", (dist,))

    ### SAMPLE CACHING
    @property
    def cached_sample(self):
        return self._cached_sample

    def clear_cached_sample(self):
        self._cached_sample = None

    def get_from_cache(self, x=None, unit=None):
        if x is not None:
            return _np.asarray(x)
        if self.cached_sample is None:
            raise ValueError("No cache exists: must provide value for x or call .sample() to create a cache first")

        if unit is not None:
            return (_np.asarray(self.cached_sample) * self.unit).to(unit).value

        return _np.asarray(self.cached_sample)


    ### PROPERTIES/METHODS THAT EXPOSE UNDERLYING SCIPY.STATS FUNCTIONALITY

    @property
    def dist_constructor_func(self):
        """
        Return the callable function to access the underlying distribution
        constructor (often the scipy.stats random variable generator function).

        See also:

        * <<class>.dist_constructor_args>
        * <<class>.dist_constructor_object>

        Returns
        -------
        * callable function
        """
        return self._dist_constructor_func

    @property
    def dist_constructor_argnames(self):
        """
        """
        return self._dist_constructor_argnames

    @property
    def dist_constructor_args(self):
        """
        Return the arguments to pass to the the underlying distribution
        constructor (often the scipy.stats random variable generator function)

        See also:

        * <<class>.dist_constructor_func>
        * <<class>.dist_constructor_object>

        Returns
        -------
        * tuple
        """
        return [getattr(self, a) for a in self.dist_constructor_argnames]

    def _dist_constructor_object_clear_cache(self):
        """
        """
        # print("*** clearing cache {}".format(self))
        self._dist_constructor_object_cache = None
        for parent in self._parents_with_constructor_object_cache:
            parent._dist_constructor_object_clear_cache()

    @property
    def dist_constructor_object(self):
        """
        Return the instantiated underlying distribution constructor (often the
        scipy.stats random variable object).

        See also:

        * <<class>.dist_constructor_func>
        * <<class>.dist_constructor_args>

        Returns
        -------
        * object
        """
        if self._dist_constructor_object_cache is None:
            self._dist_constructor_object_cache = self.dist_constructor_func(*self.dist_constructor_args)

        return self._dist_constructor_object_cache

    def pdf(self, x=None, unit=None):
        """
        Expose the probability density function (pdf) at values of `x`.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.pdf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.logpdf>
        * <<class>.cdf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the pdf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) pdf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.pdf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support pdf".format(self.__class__.__name__))

    def logpdf(self, x=None, unit=None):
        """
        Expose the log-probability density function (log of pdf) at values of `x`.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.logpdf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.pdf>
        * <<class>.cdf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the logpdf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) logpdf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.logpdf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support logpdf".format(self.__class__.__name__))

    def cdf(self, x=None, unit=None):
        """
        Expose the cummulative density function (cdf) at values of `x`.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.cdf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.logcdf>
        * <<class>.pdf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the cdf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) cdf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.cdf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support cdf".format(self.__class__.__name__))

    def logcdf(self, x=None, unit=None):
        """
        Expose the log-cummulative density function (log of cdf) at values of `x`.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.logcdf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.cdf>
        * <<class>.pdf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the logcdf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) logcdf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.logcdf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support logcdf".format(self.__class__.__name__))

    def sample(self, *args, **kwargs):
        # must be implemented by any subclasses
        raise NotImplementedError("sample not implemented for {}".format(self.__class__.__name__))

    ### PLOTTING

    def _xlabel(self, unit=None, label=None):
        label = label if label is not None else self.label
        l = 'value' if label is None else label
        if _has_astropy and self.unit is not None and self.unit not in [_units.dimensionless_unscaled]:
            l += ' ({})'.format(unit if unit is not None else self.unit)

        return l


    def plot(self, size=1e5, unit=None,
             wrap_at=None, seed=None,
             samples=None,
             plot_sample=True, plot_sample_kwargs={'color': 'gray'},
             plot_pdf=True, plot_pdf_kwargs={'color': 'red'},
             plot_cdf=False, plot_cdf_kwargs={'color': 'green'},
             plot_gaussian=False, plot_gaussian_kwargs={'color': 'blue'},
             label=None, xlabel=None, show=False, **kwargs):
        """
        Plot both the analytic distribution function as well as a sampled
        histogram from the distribution.  Requires matplotlib to be installed.

        See also:

        * <<class>.plot_sample>
        * <<class>.plot_pdf>
        * <<class>.plot_cdf>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `size` (int, optional, default=1e5): number of points to sample for
            the histogram.  See also <<class>.sample>.  Will be ignored
            if `samples` is provided.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.  If `samples` is provided,
            the passed values will be assumed to be in the correct units.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.  Will be ignored if
            `samples` is provided.
        * `seed` (int, optional): seed to use when sampling.  See also
            <<class>.sample>.  Will be ignored if `samples` is provided.
        * `samples` (array, optional, default=None): plot specific sampled
            values instead of calling <<class>.sample> internally.  Will override
            `size`.
        * `plot_sample` (bool, optional, default=True): whether to plot the
            histogram from sampling.  See also <<class>.plot_sample>.
        * `plot_sample_kwargs` (dict, optional, default={'color': 'gray'}):
            keyword arguments to send to <<class>.plot_sample>.
        * `plot_pdf` (bool, optional, default=True): whether to plot the
            analytic form of the underlying distribution, if applicable.
            See also <<class>.plot_pdf>.
        * `plot_pdf_kwargs` (dict, optional, default={'color': 'red'}):
            keyword arguments to send to <<class>.plot_pdf>.
        * `plot_cdf` (bool, optional, default=True): whether to plot the
            analytic form of the cdf, if applicable.
            See also <<class>.plot_cdf>.
        * `plot_cdf_kwargs` (dict, optional, default={'color': 'green'}):
            keyword arguments to send to <<class>.plot_cdf>.
        * `plot_gaussian` (bool, optional, default=False): whether to plot
            a guassian distribution fit to the sample.  Only supported for
            distributions that have <<class>.to_gaussian> methods.
        * `plot_gaussian_kwargs` (dict, optional, default={'color': 'blue'}):
            keyword arguments to send to <<class>.plot_gaussian>.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.  Unit will automatically be appended.
            Will be ignored if `xlabel` is provided.
        * `xlabel` (string, optional, default=None): override the label on the
            x-axis without appending the unit.  Will override `label`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: all keyword arguments (except for `bins`) will be passed
            on to <<class>.plot_pdf> and <<class>.plot_gaussian> and all
            keyword arguments will be passed on to <<class>.plot_sample>.
            Keyword arguments defined in `plot_sample_kwargs`,
            `plot_pdf_kwargs`, and `plot_gaussian_kwargs`
            will override the values sent in `kwargs`.

        Returns
        --------
        * tuple: the return values from <<class>.plot_sample> (or None if
            `plot_sample=False`), <<class>.plot_pdf> (or None if `plot_pdf=False`),
            <<class>.plot_cdf> (or None if `plot_cdf=False`),
            and <Gaussian.plot_pdf> (or None if `plot_gaussian=False`).

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        if not _has_mpl:
            raise ImportError("matplotlib required for plotting")

        ret = []

        if plot_sample:
            # we have to make a copy here, otherwise setdefault will change the
            # defaults in the function declaration for successive calls
            plot_sample_kwargs = plot_sample_kwargs.copy()
            for k,v in kwargs.items():
                plot_sample_kwargs.setdefault(k,v)
            ret_sample = self.plot_sample(size=int(size), samples=samples, unit=unit, wrap_at=wrap_at, seed=seed, show=False, **plot_sample_kwargs)
        else:
            ret_sample = None

        if plot_gaussian or plot_pdf or plot_cdf:
            # we need to know the original x-range, before wrapping
            # sample = self.sample(size=size, unit=unit, wrap_at=False, cache_sample=False)
            # xmin = _np.min(sample)
            # xmax = _np.max(sample)
            xmin, xmax = self.interval(0.999, wrap_at=False, unit=unit)
            x = _np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 1000)

        if plot_gaussian:
            if not hasattr(self, 'to_gaussian'):
                raise NotImplementedError("{} cannot plot with `plot_gaussian=True`".format(self.__class__.__name__))
            # we have to make a copy here, otherwise setdefault will change the
            # defaults in the function declaration for successive calls
            plot_gaussian_kwargs = plot_gaussian_kwargs.copy()
            for k,v in kwargs.items():
                if k in ['bins']:
                    continue
                plot_gaussian_kwargs.setdefault(k,v)
            ret_gauss = self.plot_gaussian(x, unit=unit, wrap_at=wrap_at, show=False, **plot_gaussian_kwargs)

        else:
            ret_gauss = None

        if plot_pdf:
            # we have to make a copy here, otherwise setdefault will change the
            # defaults in the function declaration for successive calls
            plot_pdf_kwargs = plot_pdf_kwargs.copy()
            for k,v in kwargs.items():
                if k in ['bins']:
                    continue
                plot_pdf_kwargs.setdefault(k,v)
            ret_pdf = self.plot_pdf(x, unit=unit, wrap_at=wrap_at, show=False, **plot_pdf_kwargs)
        else:
            ret_pdf = None

        if plot_cdf:
            # we have to make a copy here, otherwise setdefault will change the
            # defaults in the function declaration for successive calls
            plot_cdf_kwargs = plot_cdf_kwargs.copy()
            for k,v in kwargs.items():
                if k in ['bins']:
                    continue
                plot_cdf_kwargs.setdefault(k,v)
            ret_cdf = self.plot_cdf(x, unit=unit, wrap_at=wrap_at, show=False, **plot_cdf_kwargs)
        else:
            ret_cdf = None

        if show:
            _plt.xlabel(self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()

        return (ret_sample, ret_pdf, ret_cdf, ret_gauss)


    def plot_sample(self, size=100000, unit=None,
                    wrap_at=None, seed=None,
                    samples=None,
                    label=None, xlabel=None, show=False, **kwargs):
        """
        Plot both a sampled histogram from the distribution.  Requires
        matplotlib to be installed.

        See also:

        * <<class>.plot>
        * <<class>.plot_pdf>
        * <<class>.plot_cdf>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `size` (int, optional, default=1e5): number of points to sample for
            the histogram.  See also <<class>.sample>.  Will be ignored
            if `samples` is provided.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.  If `samples` is provided,
            the passed values will be assumed to be in the correct units.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.  Will be ignored
            if `samples` is provided.
        * `seed` (int, optional): seed to use when sampling.  See also
            <<class>.sample>.  Will be ignored if `samples` is provided.
        * `samples` (array, optional, default=None): plot specific sampled
            values instead of calling <<class>.sample> internally.  Will override
            `size`.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.  Unit will automatically be appended.
            Will be ignored if `xlabel` is provided.
        * `xlabel` (string, optional, default=None): override the label on the
            x-axis without appending the unit.  Will override `label`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: all keyword arguments will be passed on to plt.hist.  If
            not provided, `bins` will default to the stored bins for <Histogram>
            distributions, otherwise will default to 25.

        Returns
        --------
        * the return from plt.hist

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        if not _has_mpl:
            raise ImportError("matplotlib required for plotting")

        if hasattr(self, 'bins'):
            # let's default to the stored bins (probably only the case for
            # histogram distributions)
            if wrap_at or (hasattr(self, 'wrap_at') and self.wrap_at):
                kwargs.setdefault('bins', len(self.bins))
            else:
                kwargs.setdefault('bins', self.bins)
        else:
            kwargs.setdefault('bins', 25)

        # TODO: wrapping can sometimes cause annoying things with bins due to a large datagap.
        # Perhaps we should bin and then wrap?  Or bin before wrapping and get a guess at the
        # appropriate bins
        if samples is None:
            samples = self.sample(size, unit=unit, wrap_at=wrap_at, seed=seed, cache_sample=False)

        try:
            ret = _plt.hist(samples, density=True, **kwargs)
        except AttributeError:
            # TODO: determine which version of matplotlib
            # TODO: this still doesn't handle the same
            ret = _plt.hist(samples, normed=True, **kwargs)

        if show:
            _plt.xlabel(xlabel if xlabel is not None else self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()

        return ret

    def plot_pdf(self, x=None, unit=None, wrap_at=None,
                  label=None, xlabel=None, show=False, **kwargs):
        """
        Plot the pdf function.  Requires matplotlib to be installed.

        See also:

        * <<class>.plot>
        * <<class>.plot_cdf>
        * <<class>.plot_sample>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `x` (array, optional, default=None): the numpy array at which to
            sample the value on the x-axis.  If `unit` is not None, the value
            of `x` are assumed to be in the original units <<class>.unit>,
            not `unit`.  If not provided or None, `x` will be based to cover
            the 99.9% of all distributions (see <<class>.interval>) with 1000
            points and 10% padding.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.  Unit will automatically be appended.
            Will be ignored if `xlabel` is provided.
        * `xlabel` (string, optional, default=None): override the label on the
            x-axis without appending the unit.  Will override `label`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: all keyword arguments will be passed on to plt.plot.  Note:
            if wrapping is enabled, either via `wrap_at` or <<class>.wrap_at>,
            the resulting line will break when wrapping, resulting in using multiple
            colors.  Sending `color` as a keyword argument will prevent this
            matplotlib behavior.  Calling this through <<class>.plot> with
            `plot_gaussian=True` defaults to sending `color='blue'` through
            the `plot_gaussian_kwargs` argument.

        Returns
        --------
        * the return from plt.plot

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        if not _has_mpl:
            raise ImportError("matplotlib required for plotting")

        if x is None:
            # TODO: test how this plays with units
            xmin, xmax = self.interval(0.999, wrap_at=False, unit=unit)
            x = _np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 1000)

        # x is assumed to be in new units
        if hasattr(self, 'pdf'):
            y = self.pdf(x, unit=unit)
            x = self.wrap(x, wrap_at=wrap_at)

            # if unit is not None:
                # print "*** converting from {} to {}".format(self.unit, unit)
                # print "*** before convert", x.min(), x.max()
                # x = (x*self.unit).to(unit).value
                # print "*** after convert", x.min(), x.max()

            # handle wrapping by making multiple calls to plot whenever the sign
            # changes direction
            split_inds = _np.where(x[1:]-x[:-1] < 0)[0]
            xs, ys = _np.split(x, split_inds+1), _np.split(y, split_inds+1)
            for x,y in zip(xs, ys):
                ret = _plt.plot(x, y, **kwargs)
        else:
            return None


        if show:
            _plt.xlabel(xlabel if xlabel is not None else self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()

        return ret

    def plot_cdf(self, x=None, unit=None, wrap_at=None,
                  label=None, xlabel=None, show=False, **kwargs):
        """
        Plot the pdf function.  Requires matplotlib to be installed.

        See also:

        * <<class>.plot>
        * <<class>.plot_pdf>
        * <<class>.plot_sample>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `x` (array, optional, default=None): the numpy array at which to
            sample the value on the x-axis.  If `unit` is not None, the value
            of `x` are assumed to be in the original units <<class>.unit>,
            not `unit`.  If not provided or None, `x` will be based to cover
            the 99.9% of all distributions (see <<class>.interval>) with 1000
            points and 10% padding.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.  Unit will automatically be appended.
            Will be ignored if `xlabel` is provided.
        * `xlabel` (string, optional, default=None): override the label on the
            x-axis without appending the unit.  Will override `label`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: all keyword arguments will be passed on to plt.plot.  Note:
            if wrapping is enabled, either via `wrap_at` or <<class>.wrap_at>,
            the resulting line will break when wrapping, resulting in using multiple
            colors.  Sending `color` as a keyword argument will prevent this
            matplotlib behavior.  Calling this through <<class>.plot> with
            `plot_gaussian=True` defaults to sending `color='blue'` through
            the `plot_gaussian_kwargs` argument.

        Returns
        --------
        * the return from plt.plot

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        if not _has_mpl:
            raise ImportError("matplotlib required for plotting")

        if x is None:
            # TODO: test how this plays with units
            xmin, xmax = self.interval(0.999, wrap_at=False, unit=unit)
            x = _np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 1000)

        # x is assumed to be in new units
        if hasattr(self, 'cdf'):
            y = self.cdf(x, unit=unit)
            x = self.wrap(x, wrap_at=wrap_at)

            # if unit is not None:
                # print "*** converting from {} to {}".format(self.unit, unit)
                # print "*** before convert", x.min(), x.max()
                # x = (x*self.unit).to(unit).value
                # print "*** after convert", x.min(), x.max()

            # handle wrapping by making multiple calls to plot whenever the sign
            # changes direction
            split_inds = _np.where(x[1:]-x[:-1] < 0)[0]
            xs, ys = _np.split(x, split_inds+1), _np.split(y, split_inds+1)
            for x,y in zip(xs, ys):
                ret = _plt.plot(x, y, **kwargs)
        else:
            return None


        if show:
            _plt.xlabel(xlabel if xlabel is not None else self._xlabel(unit, label=label))
            _plt.ylabel('cummulative density')
            _plt.show()

        return ret

    def plot_gaussian(self, x=None, unit=None, wrap_at=None,
                      label=None, xlabel=None, show=False, **kwargs):
        """
        Plot the gaussian distribution that would result from calling
        <<class>.to_gaussian> with the same arguments.

        Note that for distributions in which <<class>.to_gaussian> calls
        <<class>.to_histogram> under-the-hood, this could result in slightly
        different distributions for each call.

        See also:

        * <<class>.plot>
        * <<class>.plot_sample>
        * <<class>.plot_pdf>
        * <<class>.plot_cdf>

        Arguments
        -----------
        * `x` (array, optional, default=None): the numpy array at which to
            sample the value on the x-axis.  If `unit` is not None, the value
            of `x` are assumed to be in the original units <<class>.unit>,
            not `unit`.  If not provided or None, `x` will be based to cover
            the 99.9% of all distributions (see <<class>.interval>) with 1000
            points and 10% padding.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.  Unit will automatically be appended.
            Will be ignored if `xlabel` is provided.
        * `xlabel` (string, optional, default=None): override the label on the
            x-axis without appending the unit.  Will override `label`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: keyword arguments for `sigma`, `N`, `bins`, `range` will
            be passed on to <<class>.to_gaussian> (must be accepted by the
            given distribution type).  All other keyword arguments will be passed
            on to <Gaussian.plot_pdf> on the resulting distribution.

        Returns
        --------
        * the return from plt.plot

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        if not _has_mpl:
            raise ImportError("matplotlib required for plotting")

        if x is None:
            # TODO: test how this plays with units
            xmin, xmax = self.interval(0.999, wrap_at=False, unit=unit)
            x = _np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 1000)

        to_gauss_keys = ['sigma', 'N', 'bins', 'range']
        g = self.to_gaussian(**{k:v for k,v in kwargs.items() if k in to_gauss_keys})

        if unit is not None:
            g = g.to(unit)
            if wrap_at is not None and wrap_at is not False:
                wrap_at = (wrap_at * self.unit).to(unit).value

        # TODO: this time wrap_at is assumed to be in the plotted units, not the original... do we need to convert?
        ret = g.plot_pdf(x, wrap_at=wrap_at, **{k:v for k,v in kwargs.items() if k not in to_gauss_keys})

        if show:
            _plt.xlabel(xlabel if xlabel is not None else self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()
        return ret



class BaseUnivariateDistribution(BaseDistribution):
    def __init__(self, unit, label, wrap_at, *args, **kwargs):
        super(BaseUnivariateDistribution, self).__init__(*args, **kwargs)
        self.unit = unit
        self.label = label
        self.wrap_at = wrap_at

    def __repr__(self):
        descriptors = " ".join(["{}={}".format(k,getattr(self,k)) for k in self._descriptors])
        if self.unit is not None:
            descriptors += " unit={}".format(self.unit)
        if self.wrap_at is not None:
            descriptors += " wrap_at={}".format(self.wrap_at)
        if self.label is not None:
            descriptors += " label={}".format(self.label)
        return "<distl.{} {}>".format(self.__class__.__name__.lower(), descriptors)

    def __str__(self):
        if self.label is not None:
            return "{"+self.label+"}"
        else:
            return self.__repr__()

    def to_dict(self):
        """
        Return the dictionary representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <distl.from_dict>.

        See also:

        * <<class>.to_json>
        * <<class>.to_file>

        Returns
        --------
        * dictionary
        """
        d = {k: _json_safe(getattr(self, k)) for k in self._descriptors}
        d['distl'] = self.__class__.__name__
        d['distl.version'] = __version__
        if self.unit is not None:
            d['unit'] = str(self.unit.to_string())
        if self.label is not None:
            d['label'] = self.label
        if self.wrap_at is not None:
            d['wrap_at'] = self.wrap_at
        return d

    ### LABEL

    @property
    def label(self):
        """
        The label of the distribution object.  When not None, this is used for
        the x-label when plotting (see <<class>.plot>) and for the
        string representation for any math in a <Composite>.
        """
        return self._label

    @label.setter
    def label(self, label):
        if label is not None:
            try:
                label = str(label)
            except:
                raise TypeError("label must be of type str")

        self._label = label

    ### UNITS AND UNIT CONVERSIONS

    @property
    def unit(self):
        """
        The units of the distribution.  Astropy is required in order to set
        and/or use distributions with units.

        See also:

        * <<class>.to>
        """
        return self._unit

    @unit.setter
    def unit(self, unit):
        if isinstance(unit, str) or isinstance(unit, unicode):
            unit = _units.Unit(unit)

        if not (unit is None or isinstance(unit, _units.Unit) or isinstance(unit, _units.CompositeUnit) or isinstance(unit, _units.IrreducibleUnit)):
            raise TypeError("unit must be of type astropy.units.Unit, got {} (type: {})".format(unit, type(unit)))

        self._unit = unit

    def _return_with_units(self, value, unit=None, as_quantity=False):
        if (as_quantity or unit) and not _has_astropy:
            raise ImportError("astropy required to return quantity objects")

        if unit is None and not as_quantity:
            return value

        if self.unit is not None:
            value *= self.unit

        if unit is not None:
            if self.unit is None:
                raise ValueError("can only return unit if unit is set for distribution")

            if not _has_astropy:
                raise ImportError("astropy must be installed for unit support")

            value = value.to(unit)


        if as_quantity:
            return value
        else:
            return value.value

    def to(self, unit):
        """
        Convert to different units.  This creates a copy and returns the
        new distribution with the new units.  Astropy is required in order to
        set and/or use units.

        See also:

        * <<class>.unit>

        Arguments
        ------------
        * `unit` (astropy.unit object): unit to use in the new distribution.
            The current units (see <<class>.unit>) must be able to
            convert to the requested units.

        Returns
        ------------
        * the new distribution object

        Raises
        -----------
        * ImportError: if astropy dependency is not met.
        """
        if not _has_astropy:
            raise ImportError("astropy required to handle units")

        if self.unit is None or self.unit in [_units.dimensionless_unscaled]:
            # then we'll just adopt the units without applying any scaling
            factor = 1.0
        else:
            factor = self.unit.to(unit)

        new_dist = self.copy()
        new_dist.unit = unit
        if new_dist.wrap_at is not None and new_dist.wrap_at is not False:
            new_dist.wrap_at *= factor
        new_dist *= factor
        return new_dist

    def to_si(self):
        """
        """
        physical_type = self.unit.physical_type

        if physical_type not in _physical_types_to_si.keys():
            raise NotImplementedError("cannot convert object with physical_type={} to SI units".format(physical_type))

        return self.to(_units.Unit(_physical_types_to_si.get(physical_type)))

    def to_solar(self):
        """
        """
        physical_type = self.unit.physical_type

        if physical_type not in _physical_types_to_solar.keys():
            raise NotImplementedError("cannot convert object with physical_type={} to solar units".format(physical_type))

        return self.to(_units.Unit(_physical_types_to_solar.get(physical_type)))

    ### CONVENIENCE METHODS FOR SAMPLING/WRAPPING/PLOTTING

    @property
    def wrap_at(self):
        """
        Value at which to wrap all sampled values.  If <<class>.unit> is not None,
        then the value of `wrap_at` is the same as the set units.

        If `False`: will not wrap
        If `None`: will wrap on range 0-2pi (0-360 deg) if <<class>.unit> are angular
            or 0-1 if <<class>.unit> are cycles.
        If float: will wrap on range 0-`wrap_at`.

        See also:

        * <<class>.get_wrap_at>
        * <<class>.wrap>

        Returns
        ---------
        * (float or None)
        """
        return self._wrap_at

    @wrap_at.setter
    def wrap_at(self, wrap_at):
        if wrap_at is None or wrap_at is False:
            self._wrap_at = wrap_at

        elif not (isinstance(wrap_at, float) or isinstance(wrap_at, int)):
            raise TypeError("wrap_at={} must be of type float, int, False, or None".format(wrap_at))

        else:
            self._wrap_at = wrap_at

    def get_wrap_at(self, wrap_at=None):
        """
        Get the computed value used for wrapping, given `wrap_at` as an optional
        override to the attribute <<class>.wrap_at>.

        See also:

        * <<class>.wrap_at>
        * <<class>.wrap>

        Arguments
        ------------
        * `wrap_at` (float or False or None, optional, default=None): override
            the value of <<class>.wrap_at>.

        Returns
        ----------
        * The computed wrapping value, accounting for <<class>.unit> if `wrap_at`
            is None.
        """

        if wrap_at is None:
            wrap_at = self.wrap_at

        if wrap_at is None:
            if _has_astropy and self.unit is not None:
                if self.unit.physical_type == 'angle':
                    if self.unit.to_string() == 'deg':
                        wrap_at = 360
                    elif self.unit.to_string() == 'cycle':
                        wrap_at = 1
                    elif self.unit.to_string() == 'rad':
                        wrap_at = 2 * _np.pi
                    else:
                        raise NotImplementedError("wrapping for angle unit {} not implemented.".format(self.unit.to_string()))
            else:
                wrap_at = False

        return wrap_at

    def wrap(self, value, wrap_at=None):
        """
        Wrap values via modulo:

        ```py
        value % wrap_at
        ```

        See also:

        * <<class>.wrap_at>
        * <<class>.get_wrap_at>

        Arguments
        ------------
        * `value` (float or array): values to wrap
        * `wrap_at` (float, optional, default=None): value to use for the upper-limit.
            If not provided or None, will use <<class>.wrap_at>.  If False,
            will return `value` unwrapped.

        Returns
        ----------
        * (float or array): same shape/type as input `value`.
        """
        wrap_at = self.get_wrap_at(wrap_at)

        if wrap_at is False or wrap_at is None:
            return value

        return value % wrap_at

    ### expose underlying scipy.stats functionality for rv_continuous (univariate only methods)

    def ppf(self, q, unit=None, as_quantity=False, wrap_at=None):
        """
        Expose the percent point function (ppf; iverse of cdf - percentiles) at
        values of `q`.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> with unit-conversions, support for
        quantity objects, and wrapping done on the returned result.

        See also:

        * <<class>.pdf>
        * <<class>.cdf>
        * <<class>.sample>

        Arguments
        ----------
        * `q` (float or array): percentiles at which to expose the ppf
        * `unit` (astropy.unit, optional, default=None): unit of the exposed
            values.  If None or not provided, will assume they're provided in
            <<class>.unit>.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.

        Returns
        ---------
        * (float or array) ppf values of the same type/shape as `x`
        """
        try:
            ppf = self.dist_constructor_object.ppf(q)
        except AttributeError:
            raise NotImplementedError("{} does not support ppf".format(self.__class__.__name__))

        return self._return_with_units(self.wrap(ppf, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def sf(self, x=None, unit=None):
        """
        Expose the survival function (sf; also defined as 1 - cdf, but sf is
        sometimes more accurate)

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.sf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.cdf>
        * <<class>.logsf>
        * <<class>.isf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the sf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) sf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.sf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support sf".format(self.__class__.__name__))

    def logsf(self, x=None, unit=None):
        """
        Expose the log of the survival function (logsf).

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.logsf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.sf>
        * <<class>.cdf>
        * <<class>.isf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the logsf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) logsf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.logsf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support logsf".format(self.__class__.__name__))

    def isf(self, x=None, unit=None):
        """
        Expose the inverse of the survival function (isf).

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.isf.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        See also:

        * <<class>.sf>
        * <<class>.cdf>
        * <<class>.logsf>

        Arguments
        ----------
        * `x` (float or array, optional, default=None): x-values at which to
            expose the isf.  If None or not provided, <<class>.cached_sample>
            will be used if available, or raise an error if no cached samples
            are available.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * (float or array) osf values of the same type/shape as `x`
        """
        x = self.get_from_cache(x, unit=unit)

        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        try:
            return self.dist_constructor_object.isf(x)
        except AttributeError:
            raise NotImplementedError("{} does not support isf".format(self.__class__.__name__))

    def moment(self, m):
        """
        Expose the non-central moment of order `m`.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.moment.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object>.

        Arguments
        ----------
        * `m` (int): order

        Returns
        ---------
        * (float) non-central moment
        """
        try:
            return self.dist_constructor_object.moment(m)
        except AttributeError:
            raise NotImplementedError("{} does not support moment".format(self.__class__.__name__))

    def entropy(self):
        """
        Expose the (differental) entropy.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.entropy.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        Returns
        ---------
        * (float) entropy
        """
        try:
            return self.dist_constructor_object.entropy()
        except AttributeError:
            raise NotImplementedError("{} does not support entropy".format(self.__class__.__name__))

    def expect(self, func, args=(), lb=None, ub=None, conditional=False, **kwargs):
        """
        Expose the expected value of a function (of one argument) with respect
        to the distribution.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.expect.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> after doing any requested unit-conversions.

        Arguments
        -----------
        * `func` (callable): passed directly to scipy (see link above)
        * `args` (tuple, optional): passed directly to scipy (see link above)
        * `lb` (float, optional): passed directly to scipy (see link above)
        * `ub` (float, optional): passed directly to scipy (see link above)
        * `conditional` (bool, optional, default=False): passed directly to scipy (see link above)
        * `**kwargs`: passed directly to scipy (see link above)

        Returns
        ---------
        * the output from scipy (see link above)
        """
        try:
            return self.dist_constructor_object.expect(func, args=args, lb=lb,
                                                        ub=ub,
                                                        conditional=conditional,
                                                        **kwargs)
        except AttributeError:
            raise NotImplementedError("{} does not support entropy".format(self.__class__.__name__))


    def median(self, unit=None, as_quantity=False, wrap_at=None):
        """
        Expose the median of the distribution.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.median.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> with unit-conversions, support for
        quantity objects, and wrapping done on the returned result.

        See also:

        * <<class>.mean>
        * <<class>.var>
        * <<class>.std>

        Arguments
        ----------
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x` to expose.  If None or not provided, will assume they're in
            <<class>.unit>.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.

        Returns
        ---------
        * (float) median of the distribution in units `unit`.
        """
        try:
            median = self.dist_constructor_object.median()
        except AttributeError as err:
            raise NotImplementedError("{} does not support median.  Original error: {}".format(self.__class__.__name__, err))

        return self._return_with_units(self.wrap(median, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def mean(self, unit=None, as_quantity=False, wrap_at=None):
        """
        Expose the mean of the distribution.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.mean.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> with unit-conversions, support for
        quantity objects, and wrapping done on the returned result.

        See also:

        * <<class>.median>
        * <<class>.var>
        * <<class>.std>

        Arguments
        ----------
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x` to expose.  If None or not provided, will assume they're in
            <<class>.unit>.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.

        Returns
        ---------
        * (float) mean of the distribution in units `unit`.
        """
        try:
            mean = self.dist_constructor_object.mean()
        except AttributeError:
            raise NotImplementedError("{} does not support mean".format(self.__class__.__name__))

        return self._return_with_units(self.wrap(mean, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def var(self, unit=None, as_quantity=False, wrap_at=None):
        """
        Expose the variance of the distribution.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.var.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> with unit-conversions, support for
        quantity objects, and wrapping done on the returned result.

        See also:

        * <<class>.median>
        * <<class>.mean>
        * <<class>.std>

        Arguments
        ----------
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x` to expose.  If None or not provided, will assume they're in
            <<class>.unit>.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.

        Returns
        ---------
        * (float) variance of the distribution in units `unit`.
        """
        try:
            var = self.dist_constructor_object.var()
        except AttributeError:
            raise NotImplementedError("{} does not support var".format(self.__class__.__name__))

        return self._return_with_units(self.wrap(var, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def std(self, unit=None, as_quantity=False, wrap_at=None):
        """
        Expose the standard deviation of the distribution.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.std.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> with unit-conversions, support for
        quantity objects, and wrapping done on the returned result.

        See also:

        * <<class>.median>
        * <<class>.mean>
        * <<class>.var>

        Arguments
        ----------
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x` to expose.  If None or not provided, will assume they're in
            <<class>.unit>.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.

        Returns
        ---------
        * (float) standard deviation of the distribution in units `unit`.
        """
        try:
            std = self.dist_constructor_object.std()
        except AttributeError:
            raise NotImplementedError("{} does not support std".format(self.__class__.__name__))

        return self._return_with_units(self.wrap(std, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def interval(self, alpha, unit=None, as_quantity=False, wrap_at=None):
        """
        Expose the range that contains alpha percent of the distribution.

        See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.interval.html)

        This method is just a wrapper around the scipy.stats method on
        <<class>.dist_constructor_object> with unit-conversions, support for
        quantity objects, and wrapping done on the returned result.

        Arguments
        ----------
        * `alpha` (float): passed directly to scipy (see link above)
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x` to expose.  If None or not provided, will assume they're in
            <<class>.unit>.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.

        Returns
        ---------
        * (array) endpoints in units `unit`.
        """
        try:
            interval = self.dist_constructor_object.interval(alpha)
        except AttributeError:
            raise NotImplementedError("{} does not support interval".format(self.__class__.__name__))

        # we call np.asarray so that wrapping and units works on an array object instead of a tuple
        return self._return_with_units(self.wrap(_np.asarray(interval), wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    ### SAMPLING

    def sample(self, size=None, unit=None, as_quantity=False, wrap_at=None, seed=None, cache_sample=True):
        """
        Sample from the distribution.

        See also:

        * <<class>.pdf>
        * <<class>.cdf>
        * <<class>.ppf>
        * <<class>.plot_sample>
        * <<class>.plot>

        Arguments
        -----------
        * `size` (int or tuple or None, optional, default=None): size/shape of the
            resulting array.
        * `unit` (astropy.unit, optional, default=None): unit to convert the
            resulting sample(s).  Astropy must be installed in order to convert
            units.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `seed` (int, optional): seed to pass to np.random.seed
            prior to sampling.
        * `cache_sample` (bool, optional, default=True): whether to override the
            existing <<class>.cached_sample>.

        Returns
        ---------
        * float or array: float if `size=None`, otherwise a numpy array with
            shape defined by `size`.
        """
        if isinstance(seed, dict):
            seed = seed.get(self.hash, None)

        if seed is not None:
            _np.random.seed(seed)

        qs = _np.random.random(size=size)
        sample = self.dist_constructor_object.ppf(qs)
        if cache_sample:
            self._cached_sample = sample

        return self._return_with_units(self.wrap(sample, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

        # this causes all sorts of issues as it casts the interpolators to arrays
        # return self._return_with_units(self.wrap(self.dist_constructor_object.rvs(size=size), wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)


    ### CONVERSION TO OTHER DISTRIBUTION TYPES
    def to_delta(self, loc='median'):
        """
        Convert the <<class>> distribution to a <Delta> distribution at the
        <<class>.median> (or <<class>.mean>).

        Arguments
        ------------
        * `loc` (string or float, optional, default='median'):  If a float,
            will create a delta function directly at that value.  If 'median' or
            'mean' will use <<class>.median> or <<class>.mean>, respectively.
            If 'sample', will draw a random sample from <<class>.sample>.
            All other strings will raise a ValueError.

        Returns
        -----------
        * a <Delta> object

        Raises
        ----------
        * ValueError: if the value of `loc` is not one of 'median', 'mean', 'sample'
        """
        if isinstance(loc, str):
            if loc not in ['median', 'mean', 'sample']:
                raise ValueError("loc must be a float or one of 'median', 'mean', 'sample'")

            loc = getattr(self, loc)()

        elif not (isinstance(loc, float) or isinstance(loc, int)):
            raise TypeError("loc must be a float or one of 'median', 'mean', 'sample'")

        return Delta(loc=loc,
                     unit=self.unit, label=self.label, wrap_at=self.wrap_at)

    def to_histogram(self, N=100000, bins=10, range=None, wrap_at=None):
        """
        Convert the <<class>> distribution to a <Histogram> distribution.

        Under-the-hood, this calls <<class>.sample> with `size=N` and `wrap_at=False`
        and passes the resulting array as well as the requested `bins` and `range`
        to <Histogram.from_data>.

        Arguments
        -----------
        * `N` (int, optional, default=100000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.
        * `wrap_at` (float or None, optional, default=None): value to set for
            `wrap_at` of the returned <Histogram>.  If None or not provided,
            will default to <<class>.wrap_at>.

        Returns
        --------
        * a <Histogram> object
        """
        return Histogram.from_data(self.sample(size=N, wrap_at=False, cache_sample=False),
                                   bins=bins, range=range,
                                   unit=self.unit, label=self.label, wrap_at=wrap_at if wrap_at is not None else self.wrap_at)


class BaseMultivariateDistribution(BaseDistribution):
    def __init__(self, units=None, labels=None, wrap_ats=None, *args, **kwargs):
        super(BaseMultivariateDistribution, self).__init__(*args, **kwargs)
        self.units = units
        self.labels = labels
        self.wrap_ats = wrap_ats

    def __repr__(self):
        descriptors = " ".join(["{}={}".format(k,getattr(self,k)) for k in self._descriptors])
        if self.units is not None:
            descriptors += " units={}".format(self.units)
        if self.wrap_ats is not None:
            descriptors += " wrap_ats={}".format(self.wrap_ats)
        if self.labels is not None:
            descriptors += " labels={}".format(self.labels)
        return "<distl.{} {}>".format(self.__class__.__name__.lower(), descriptors)

    def __str__(self):
        if self.labels is not None:
            return "{"+self.labels+"}"
        else:
            return self.__repr__()

    def to_dict(self):
        """
        Return the dictionary representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <distl.from_dict>.

        See also:

        * <<class>.to_json>
        * <<class>.to_file>

        Returns
        --------
        * dictionary
        """

        d = {k:_json_safe(getattr(self,k)) for k in self._descriptors}
        d['distl'] = self.__class__.__name__
        d['distl.version'] = __version__
        if self.units is not None:
            d['units'] = [u.to_string() if u is not None else None for u in self.units]
        if self.labels is not None:
            d['labels'] = self.labels
        if self.wrap_ats is not None:
            d['wrap_ats'] = self.wrap_ats
        return d

    def _get_dimension_index(self, dimension):
        if isinstance(dimension, str) or isinstance(dimension, unicode):
            if dimension not in self.labels:
                raise ValueError("dimension must be one of {}".format(self.labels))

            dimension = self.labels.index(dimension)

        if isinstance(dimension, int):
            ndimensions = self.ndimensions
            if dimension < 0 or dimension > ndimensions-1:
                raise ValueError("dimension must be between 0 and {}".format(ndimensions-1))
        else:
            raise TypeError("dimension must be of type int or str")

        return dimension

    @property
    def labels(self):
        """
        The labels of each dimension in the multivariate distribution.

        See also:

        * <<class>.all_labels>
        * <<class>.dimensions>

        Returns
        ---------
        * (list): list of labels
        """
        return self._labels #[self.dimension_indices]

    @labels.setter
    def labels(self, labels):
        if isinstance(labels, str) or isinstance(labels, unicode):
            raise NotImplementedError()
            # labels = [label for _ in range(self._ndimensions_available)]

        if not (labels is None or isinstance(labels, list)):
            raise TypeError("labels must be of type list")

        self._labels = labels

    ### UNITS AND UNIT CONVERSIONS

    # @property
    # def all_units(self):
    #     """
    #
    #     See also:
    #
    #     * <<class>.units>
    #     * <<class>.dimensions>
    #     """
    #     return self._units
    #
    # @all_units.setter
    # def all_units(self, units):
    #     if not isinstance(units, list):
    #         units = [units]
    #
    #     units = [to_unit(u) for u in units]

    @property
    def units(self):
        """
        The units of each dimension in the distribution.  Astropy is required in
        order to set and/or use distributions with units.

        See also:

        * <<class>.all_units>
        * <<class>.dimensions>

        """
        return self._units #[self.dimension_indices]

    @units.setter
    def units(self, units):
        if units is None:
            self._units = units
            return

        if not isinstance(units, list):
            units = [units]

        units = [to_unit(u) for u in units]


        if not _np.all([unit is None or isinstance(unit, _units.Unit) or isinstance(unit, _units.CompositeUnit) or isinstance(unit, _units.IrreducibleUnit) for unit in units]):
            raise TypeError("units must be a list of type astropy.units.Unit")

        self._units = units


    ### CONVENIENCE METHODS FOR SAMPLING/WRAPPING/PLOTTING

    # @property
    # def all_wrap_ats(self):
    #     """
    #
    #     See also:
    #
    #     * <<class>.wrap_ats>
    #     * <<class>.dimensions>
    #     """
    #     return self._wrap_ats
    #
    # @all_wrap_ats.setter
    # def all_wrap_ats(self, wrap_ats):

    @property
    def wrap_ats(self):
        """
        Value at which to wrap all sampled values.  If <<class>.unit> is not None,
        then the value of `wrap_at` is the same as the set units.

        If `False`: will not wrap
        If `None`: will wrap on range 0-2pi (0-360 deg) if <<class>.unit> are angular
            or 0-1 if <<class>.unit> are cycles.
        If float: will wrap on range 0-`wrap_at`.

        See also:

        * <<class>.all_wrap_ats>
        * <<class>.dimensions>
        * <<class>.get_wrap_at>
        * <<class>.wrap>

        Returns
        ---------
        * (float or None)
        """
        return self._wrap_ats

    @wrap_ats.setter
    def wrap_ats(self, wrap_ats):
        if wrap_ats is None or wrap_ats is False:
            self._wrap_ats = wrap_ats
            return

        if isinstance(wrap_ats, float) or isinstance(wrap_ats, int):
            raise NotImplementedError()
            # wrap_ats = [wrap_ats for _ in self.]

        if not isinstance(wrap_ats, list):
            raise TypeError("wrap_ats must be a list of floats")

        if not np.all([isinstance(w, int) or isinstance(w, float) for w in wrap_ats]):
            raise ValueError("wrap_ats must be a list of floats")

        self._wrap_ats = wrap_ats


    def get_wrap_at(self, wrap_at=None):
        """
        Get the computed value used for wrapping, given `wrap_at` as an optional
        override to the attribute <<class>.wrap_at>.

        See also:

        * <<class>.wrap_at>
        * <<class>.wrap>

        Arguments
        ------------
        * `wrap_at` (float or False or None, optional, default=None): override
            the value of <<class>.wrap_at>.

        Returns
        ----------
        * The computed wrapping value, accounting for <<class>.unit> if `wrap_at`
            is None.
        """

        if wrap_at is None:
            wrap_at = self.wrap_at

        if wrap_at is None:
            if _has_astropy and self.unit is not None:
                if self.unit.physical_type == 'angle':
                    if self.unit.to_string() == 'deg':
                        wrap_at = 360
                    elif self.unit.to_string() == 'cycle':
                        wrap_at = 1
                    elif self.unit.to_string() == 'rad':
                        wrap_at = 2 * _np.pi
                    else:
                        raise NotImplementedError("wrapping for angle unit {} not implemented.".format(self.unit.to_string()))
            else:
                wrap_at = False

        return wrap_at

    def pdf(self, x=None):
        x = self.get_from_cache(x)
        if not hasattr(x, '__iter__'):
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))
        if len(x.shape) > 2 or x.shape[-1] != self.ndimensions:
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))

        # TODO: unit support?
        return super(BaseMultivariateDistribution, self).pdf(x)

    def logpdf(self, x=None):
        x = self.get_from_cache(x)
        if not hasattr(x, '__iter__'):
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))
        if len(x.shape) > 2 or x.shape[-1] != self.ndimensions:
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))


        # TODO: unit support?
        return super(BaseMultivariateDistribution, self).logpdf(x)

    def cdf(self, x=None):
        x = self.get_from_cache(x)
        if not hasattr(x, '__iter__'):
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))
        if len(x.shape) > 2 or x.shape[-1] != self.ndimensions:
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))

        # TODO: unit support?
        return super(BaseMultivariateDistribution, self).cdf(x)

    def logcdf(self, x=None):
        x = self.get_from_cache(x)
        if not hasattr(x, '__iter__'):
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))
        if len(x.shape) > 2 or x.shape[-1] != self.ndimensions:
            raise TypeError('x must be an array with length {} or matrix with shape (S, {}) where S is number of samples'.format(self.ndimensions, self.ndimensions))

        # TODO: unit support?
        return super(BaseMultivariateDistribution, self).logcdf(x)



    def sample(self, size=None, dimension=None, seed=None, cache_sample=True):
        """
        Sample from the distribution.

        Arguments
        -----------
        * `size` (int or tuple or None, optional, default=None): size/shape of the
            resulting array.
        * `dimension`: (int or list of ints, optional, default=None): dimension(s)
            of the multivariate distribution to sample.  If not provided or
            None, will return all dimensions.
        * `seed` (int, optional): seed to pass to np.random.seed
            prior to sampling.
        * `cache_sample` (bool, optional, default=True): whether to override the
            existing <<class>.cached_sample>.

        Returns
        ---------
        * float or array: float if `size=None`, otherwise a numpy array with
            shape defined by `size`.
        """

        # TODO: add support for per-dimension unit, wrap_at, as_quantity (and pass in to_mvhistogram)
        # TODO: add support for seed
        if isinstance(seed, dict):
            seed = seed.get(self.hash, None)

        if seed is not None:
            _np.random.seed(seed)

        sample = self.dist_constructor_object.rvs(size=size)

        if cache_sample:
            self._cached_sample = sample

        if dimension is not None:
            if len(sample.shape) == 1:
                return sample[dimension]
            else:
                return sample[:, dimension]
        else:
            return sample

    def _xlabel(self, dimension, unit=None, label=None):
        label = label if label is not None else self.labels[dimension]
        l = 'value' if label is None else label
        if _has_astropy and self.units is not None and self.units[dimension] is not None and self.units[dimension] not in [_units.dimensionless_unscaled]:
            l += ' ({})'.format(unit if unit is not None else self.units[dimension])

        return l

    def plot_sample(self, **kwargs):
        dimension = kwargs.pop('dimension', None)

        if dimension is not None:
            # TODO: include all univariate stuffs

            # TODO: wrapping can sometimes cause annoying things with bins due to a large datagap.
            # Perhaps we should bin and then wrap?  Or bin before wrapping and get a guess at the
            # appropriate bins
            label = kwargs.pop('label', self.labels[dimension] if self.labels is not None else None)
            unit = kwargs.pop('unit', self.units[dimension] if self.units is not None else None)
            wrap_at = kwargs.pop('wrap_at', self.wrap_ats[dimension] if self.wrap_ats is not None else None)
            xlabel = kwargs.pop('xlabel', self._xlabel(dimension, unit=unit, label=label))

            samples = kwargs.pop('samples', None)
            if samples is None:
                samples = self.sample(size=int(1e5), dimension=dimension, cache_sample=False) #, unit=unit, wrap_at=wrap_at)
            return super(BaseMultivariateDistribution, self).plot_sample(samples=samples, label=label, unit=unit, wrap_at=wrap_at, xlabel=xlabel, **kwargs)
        else:
            # then we need to do a corner plot
            if not _has_corner:
                raise ImportError("corner must be installed to plot multivariate distributions.  Either install corner or pass a value to dimension to plot a 1D distribution.")


            return corner.corner(self.sample(size=int(1e5), cache_sample=False), labels=[self._xlabel(dim) for dim in range(self.ndimensions)], **kwargs)

    def plot(self, **kwargs):
        """
        """
        dimension = kwargs.pop('dimension', None)

        if dimension is not None:
            return self.slice(dimension).plot(**kwargs)
        else:
            return self.plot_sample(**kwargs)

    def to_univariate(self, *args, **kwargs):
        raise NotImplementedError("to_univariate not implemented by {}".format(self.__class__.__name__))

    def take_dimensions(self, dimensions):
        raise NotImplementedError("take_dimensions not implemented by {}".format(self.__class__.__name__))


class BaseMultivariateSliceDistribution(BaseUnivariateDistribution):
    def __init__(self, multivariate, dimension):
        self._dist_constructor_object_cache = None
        self._parents_with_constructor_object_cache = []

        if isinstance(multivariate, dict):
            multivariate = from_dict(multivariate)

        if not isinstance(multivariate, BaseMultivariateDistribution):
            raise TypeError("multivariate must be of type BaseMultivariateDistribution")

        self._multivariate = multivariate
        self.dimension = dimension



    def __repr__(self):
        descriptors = " ".join(["{}={}".format(k,getattr(self.multivariate,k)) for k in self.multivariate._descriptors])
        if self.unit is not None:
            descriptors += " unit={}".format(self.unit)
        if self.wrap_at is not None:
            descriptors += " wrap_at={}".format(self.wrap_at)
        if self.label is not None:
            descriptors += " label={}".format(self.label)
        return "<distl.{} dimension={} {})>".format(self.__class__.__name__.lower(), self.dimension, descriptors)

    def __str__(self):
        if self.label is not None:
            return "{"+self.label+"}"
        else:
            return self.__repr__()

    def to_dict(self):
        """
        Return the dictionary representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <distl.from_dict>.

        See also:

        * <<class>.to_json>
        * <<class>.to_file>

        Returns
        --------
        * dictionary
        """

        d = {}
        d['distl'] = self.__class__.__name__
        d['distl.version'] = __version__
        d['multivariate'] = self.multivariate.to_dict()
        d['dimension'] = self.dimension
        return d

    @property
    def multivariate(self):
        """
        Access the full multivariate distribution

        Returns
        ----------
        * <BaseMultivariateDistribution> object
        """
        return self._multivariate

    @property
    def hash(self):
        return self.multivariate.hash

    @property
    def hash_slice(self):
        """
        """
        # return hash(frozenset({k:v for k,v in self.to_dict().items() if k not in ['dimension']}))
        return hash(str({k:v for k,v in self.to_dict().items()}))

    @property
    def unit(self):
        """
        Access the unit of the multivariate distribution corresponsing to the
        sliced dimension

        See also:

        * <<class>.multivariate>
        * <<class>.dimension>

        Returns
        ---------
        * Unit or None
        """
        return self.multivariate.units[self.dimension] if self.multivariate.units is not None else None

    @property
    def label(self):
        """
        Access the label of the multivariate distribution corresponsing to the
        sliced dimension

        See also:

        * <<class>.multivariate>
        * <<class>.dimension>

        Returns
        -------------
        * string or None
        """
        return self.multivariate.labels[self.dimension] if self.multivariate.labels is not None else None

    @property
    def wrap_at(self):
        """
        Access the wrap_at of the multivariate distribution corresponsing to the
        sliced dimension

        See also:

        * <<class>.multivariate>
        * <<class>.dimension>

        Returns
        ---------
        * float or None
        """
        return self.multivariate.wrap_ats[self.dimension] if self.multivariate.wrap_ats is not None else None

    @property
    def dimension(self):
        """
        Access the index of the sliced dimension.

        See also:

        * <<class>.change_sliced_dimension>
        * <<class>.label>

        Returns
        ---------
        * integer
        """
        return self._dimension

    def change_slice_dimension(self, dimension):
        """
        Change the sliced dimension of the underlying <<class>.multivariate> distribution.

        See also:

        * <<class>.dimension>
        """
        self.dimension = dimension

    @dimension.setter
    def dimension(self, dimension):
        dimension = self.multivariate._get_dimension_index(dimension)
        self._dimension = dimension

    ### SAMPLE CACHING
    @property
    def cached_sample(self):
        if self.multivariate.cached_sample is None:
            return None
        return self.multivariate.cached_sample[self.dimension]

    def clear_cached_sample(self):
        self.multivariate.clear_cached_sample()

    ### OVERRIDE SCIPY.STATS FROM UNIVARIATE

    def ppf(self, q):
        raise NotImplementedError("ppf not supported for multivariate slices ({}).  Translate to a univariate via to_univariate() first.".format(self.__class__.__name__))

    ### SAMPLING & PLOTTING

    def sample(self, size=None, wrap_at=None, seed=None, cache_sample=True):
        """
        Sample the underlying <<class>.multivariate> distribution in the dimension
        defined in <<class>.dimension>.
        """

        # TODO: support unit, wrap_at, as_quantity
        return self.multivariate.sample(size=size, seed=seed, dimension=self.dimension, cache_sample=cache_sample)

    def plot_sample(self, *args, **kwargs):
        if hasattr(self, 'bins'):
            # for MVHistogramSlice, we want to take the correct bins to pass on
            # to plotting so we don't access the ndimensional MVHistogram.bins
            kwargs.setdefault('bins', self.bins)
            # if wrap_at or (hasattr(self, 'wrap_at') and self.wrap_at):
            #     kwargs.setdefault('bins', len(self.bins))
            # else:
            #     kwargs.setdefault('bins', self.bins)

        return self.multivariate.plot_sample(*args, dimension=self.dimension, **kwargs)

    def to_univariate(self):
        """

        """
        return self.multivariate.to_univariate(dimension=self.dimension)


class DistributionCollection(object):
    """
    <DistributionCollection> allows sampling from multiple distribution objects
    simultaneously, respecting all underlying covariances whenever possible.
    """
    def __init__(self, *dists):
        if isinstance(dists, BaseDistribution):
            dists = [dists]

        self.dists = dists

        self._cached_sample = None

    def to_dict(self):
        """
        Return the dictionary representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <distl.from_dict>.

        See also:

        * <<class>.to_json>
        * <<class>.to_file>

        Returns
        --------
        * dictionary
        """

        d = {}
        d['distl'] = self.__class__.__name__
        d['distl.version'] = __version__
        d['args'] = [distribution.to_dict() for distribution in self.dists]
        return d

    @property
    def dists(self):
        """
        """
        return self._dists

    @dists.setter
    def dists(self, distributions):
        # TODO: OPTIMIZE simplify this logic
        if isinstance(distributions, BaseDistribution):
            distributions = [distributions]

        if not (isinstance(distributions, list) or isinstance(distributions, tuple)):
            raise TypeError('distributions must be a list of distribution objects')

        # TODO: clear caches?
        self._dists = [is_distribution_univariate_or_slice(distribution) for distribution in distributions]

    @property
    def labels(self):
        """
        """
        return [d.label for d in self.dists]

    @property
    def dists_unpacked(self):
        """
        """
        # first well expand any Composite distributions to access the underlying
        # distributions
        def unpack_dists(dist):
            if isinstance(dist, Composite):
                dists = []
                for dist in dist.dists:
                    dists += unpack_dists(dist)
                return dists
            else:
                return [dist]

        dists_all = []
        for dist in self.dists:
            dists_all += unpack_dists(dist)

        return dists_all

    @property
    def labels_unpacked(self):
        """
        """
        return [d.label for d in self.dists_unpacked]

    @property
    def cached_sample(self):
        return self._cached_sample

    @property
    def cached_sample_unpacked(self):
        """
        """
        return _np.asarray([d.cached_sample for d in self.dists_unpacked])

    def _method_on_values(self, method, npmethod, values, as_univariates):
        # TODO: add support for units?
        dist_values_dict = self.get_distributions_with_values(values, as_univariates)
        return getattr(_np, npmethod)([getattr(dist, method)(value) for dist, value in dist_values_dict.items()])


    def _get_cached_values(self, values, as_univariates):
        if values is None:
            values = self.cached_sample if as_univariates else self.cached_sample_unpacked
            if values is None:
                raise ValueError("no cached values available.  Must past values or call .sample()")

        return _np.asarray(values)

    def get_distributions_with_values(self, values=None, as_univariates=False):
        """
        Expose the distributions and the values that will be applied when
        calling <DistributionCollection.pdf>, <DistributionCollection.logpdf>,
        <DistributionCollection.cdf>, or <DistributionCollection.logcdf>

        Arguments
        ------------
        * `values` (list, tuple, array or None, optional, default=None): list of
            values in same length and order as <DistributionCollection.distributions> or
            <DistributionCollection.distributions_unpacked> (see `as_univariates`).
            If not provided or None, the latest values from <DistributionCollection.sample>
            will be assumed (respecting the value of `as_univariates`).  If no cached
            samples are available, a ValueError will be raised.
        * `as_univariates` (bool, optional, default=False): whether `values` corresponds
            to the passed distributions (<DistributionCollection.distributions>)
            or the underlying unpacked distributions (<DistributionCollection.distributions_unpacked>).
            If the former (`as_univariates=False`), covariances will be respected
            from any underlying multivariate distributions.  If the latter
            (`as_univariates=True`) covariances will be ignored.

        Returns
        ----------
        * dictionary of distribution: value (list or float) pairs
        """
        # values_dict, dists_dict = self._get_unique_values_dists(self._get_cached_values(values, as_univariates), as_univariates)
        values = self._get_cached_values(values, as_univariates)
        dists = self.dists if as_univariates else self.dists_unpacked

        if len(values) != len(dists):
            raise ValueError("values must be same length as self.{} (length={}).  To use self.{} instead, pass as_univariates={}".format('distributions' if as_univariates else 'distributions_unpacked', len(dists), 'distributions' if not as_univariates else 'distributions_unpacked', not as_univariates))

        dists_dict = {}
        values_dict = {}
        dims_dict = {}
        for dist_orig, v in zip(dists, values):
            if not as_univariates and isinstance(dist_orig, BaseMultivariateSliceDistribution):
                d = dist_orig.multivariate
            else:
                d = dist_orig

            # if as_univariates then we want MVSlices with the same parent MV to be treated separately
            take_dimensions = not as_univariates and isinstance(dist_orig, BaseMultivariateSliceDistribution)


            hash = dist_orig.hash_slice if isinstance(dist_orig, BaseMultivariateSliceDistribution) and not take_dimensions else dist_orig.hash
            # print("***", dist_orig.label, hash)
            if hash not in dists_dict.keys():
                dists_dict[hash] = d
                values_dict[hash] = [v]
                if take_dimensions:
                    dims_dict[hash] = [dist_orig.dimension]
            elif not isinstance(dist_orig, BaseMultivariateSliceDistribution):
                # duplicate entry
                if values_dict[hash][0] != v:
                    raise ValueError("All passed values for {} must be identical".format(d if d.label is None else d.label))
            else:
                values_dict[hash].append(v)
                if take_dimensions:
                    dims_dict[hash].append(dist_orig.dimension)

        for hash, dims in dims_dict.items():
            dists_dict[hash] = dists_dict[hash].take_dimensions(dims)


        return {dists_dict.get(hash): values_dict.get(hash) if len(values_dict.get(hash)) > 1 else values_dict.get(hash)[0] for hash in values_dict.keys()}

    def pdf(self, values=None, as_univariates=False):
        """
        Compute the pdf of drawing `values` from the stored distributions.

        See also:

        * <DistributionCollection.logpdf>
        * <DistributionCollection.cdf>
        * <DistributionCollection.logcdf>
        * <DistributionCollection.get_distributions_with_values>

        Arguments
        ------------
        * `values` (list, tuple, array or None, optional, default=None): list of
            values in same length and order as <DistributionCollection.distributions> or
            <DistributionCollection.distributions_unpacked> (see `as_univariates`).
            If not provided or None, the latest values from <DistributionCollection.sample>
            will be assumed (respecting the value of `as_univariates`).  If no cached
            samples are available, a ValueError will be raised.
        * `as_univariates` (bool, optional, default=False): whether `values` corresponds
            to the passed distributions (<DistributionCollection.distributions>)
            or the underlying unpacked distributions (<DistributionCollection.distributions_unpacked>).
            If the former (`as_univariates=False`), covariances will be respected
            from any underlying multivariate distributions.  If the latter
            (`as_univariates=True`) covariances will be ignored.

        Returns
        ----------
        * float or array of floats

        Raises
        ----------
        * ValueError: if `values` is None, but no cached samples are available.
        """
        return self._method_on_values('pdf', 'product', values, as_univariates)


    def logpdf(self, values=None, as_univariates=False):
        """
        Compute the logpdf of drawing `values` from the stored distributions.

        See also:

        * <DistributionCollection.pdf>
        * <DistributionCollection.cdf>
        * <DistributionCollection.logcdf>
        * <DistributionCollection.get_distributions_with_values>

        Arguments
        ------------
        * `values` (list, tuple, array or None, optional, default=None): list of
            values in same length and order as <DistributionCollection.distributions> or
            <DistributionCollection.distributions_unpacked> (see `as_univariates`).
            If not provided or None, the latest values from <DistributionCollection.sample>
            will be assumed (respecting the value of `as_univariates`).  If no cached
            samples are available, a ValueError will be raised.
        * `as_univariates` (bool, optional, default=False): whether `values` corresponds
            to the passed distributions (<DistributionCollection.distributions>)
            or the underlying unpacked distributions (<DistributionCollection.distributions_unpacked>).
            If the former (`as_univariates=False`), covariances will be respected
            from any underlying multivariate distributions.  If the latter
            (`as_univariates=True`) covariances will be ignored.

        Returns
        ----------
        * float or array of floats

        Raises
        ----------
        * ValueError: if `values` is None, but no cached samples are available.
        """
        return self._method_on_values('logpdf', 'sum', values, as_univariates)

    def cdf(self, values=None, as_univariates=False):
        """
        Compute the cdf of drawing `values` from the stored distributions.

        See also:

        * <DistributionCollection.pdf>
        * <DistributionCollection.logpdf>
        * <DistributionCollection.logcdf>
        * <DistributionCollection.get_distributions_with_values>

        Arguments
        ------------
        * `values` (list, tuple, array or None, optional, default=None): list of
            values in same length and order as <DistributionCollection.distributions> or
            <DistributionCollection.distributions_unpacked> (see `as_univariates`).
            If not provided or None, the latest values from <DistributionCollection.sample>
            will be assumed (respecting the value of `as_univariates`).  If no cached
            samples are available, a ValueError will be raised.
        * `as_univariates` (bool, optional, default=False): whether `values` corresponds
            to the passed distributions (<DistributionCollection.distributions>)
            or the underlying unpacked distributions (<DistributionCollection.distributions_unpacked>).
            If the former (`as_univariates=False`), covariances will be respected
            from any underlying multivariate distributions.  If the latter
            (`as_univariates=True`) covariances will be ignored.

        Returns
        ----------
        * float or array of floats

        Raises
        ----------
        * ValueError: if `values` is None, but no cached samples are available.
        """
        return self._method_on_values('cdf', 'product', values, as_univariates)


    def logcdf(self, values=None, as_univariates=False):
        """
        Compute the logcdf of drawing `values` from the stored distributions.

        See also:

        * <DistributionCollection.pdf>
        * <DistributionCollection.logpdf>
        * <DistributionCollection.cdf>
        * <DistributionCollection.get_distributions_with_values>

        Arguments
        ------------
        * `values` (list, tuple, array or None, optional, default=None): list of
            values in same length and order as <DistributionCollection.distributions> or
            <DistributionCollection.distributions_unpacked> (see `as_univariates`).
            If not provided or None, the latest values from <DistributionCollection.sample>
            will be assumed (respecting the value of `as_univariates`).  If no cached
            samples are available, a ValueError will be raised.
        * `as_univariates` (bool, optional, default=False): whether `values` corresponds
            to the passed distributions (<DistributionCollection.distributions>)
            or the underlying unpacked distributions (<DistributionCollection.distributions_unpacked>).
            If the former (`as_univariates=False`), covariances will be respected
            from any underlying multivariate distributions.  If the latter
            (`as_univariates=True`) covariances will be ignored.

        Returns
        ----------
        * float or array of floats

        Raises
        ----------
        * ValueError: if `values` is None, but no cached samples are available.
        """
        return self._method_on_values('logcdf', 'sum', values, as_univariates)

    def sample(self, *args, **kwargs):
        """
        Sample from multiple distributions with random seeds automatically determined,
        but applied to distributions of the same underlying multivariate distribution
        automatically.

        For each unique <BaseDistribution.hash> in the distributions in `dists` a
        random seed will be generated and applied to <BaseDistribution.sample>
        for all distributionis in `dists` which share that same hash value.  By doing
        so, any <BaseMultivariateDistribution> which samples from the same underlying
        multivariate distribution (but for a different
        <BaseMultivariateDistribution.dimension>), will be correctly sampled to account
        for the covariance/correlation between parameters, but all other 1-D
        <BaseDistribution> objects will be sampled with their own independent
        random seeds.

        Arguments
        -------------
        * `*args`: all positional arguments are sent to <BaseDistribution.sample>
            for each item in `dists`.
        * `cache_sample` (bool, optional, default=True): whether to cache the
            sampled values for subsequent calls to <DistributionCollection.pdf>,
            <DistributionCollection.logpdf>, etc.
        * `**kwargs`: all keyword arguments are sent to <BaseDistribution.sample>
            for each item in `dists`.  Note: `seed` is forbidden and will raise
            a ValueError.

        Returns
        -------------
        * (list): list of samples, in same order as <<class>.distributions>.

        Raises
        ----------
        * ValueError: if `seed` is passed.
        """
        if 'seed' in kwargs.keys():
            raise ValueError("seeds are automatically determined: cannot pass seed")

        cache_sample = kwargs.pop('cache_sample', True)

        seeds = kwargs.pop('seeds', {})
        if seeds is None:
            seeds = {}

        for i,dist in enumerate(self.dists_unpacked):
            seeds.setdefault(dist.hash, get_random_seed()[i])

        sample_kwargs = {k:v for k,v in kwargs.items() if k not in ['seeds']}
        # print("*** seeds: {}, sample_kwargs: {}".format(seeds, sample_kwargs))
        samples = _np.asarray([dist.sample(*args, seed=seeds, cache_sample=cache_sample, **sample_kwargs) for dist in self.dists]).T

        if cache_sample:
            self._cached_sample = samples

        # TODO: units, quantity, wrap_at
        return samples

    def sample_func(self, func, x, N=1000, func_kwargs={}):
        """
        Draw samples from a callable function.

        See also:

        * <<class>.plot_func>

        Arguments
        -----------
        * `func` (callable): callable function
        * `x` (array like): x values to pass to `func`.
        * `N` (int, optional, default=1000): number of samples to draw.
        * `func_kwargs` (dict, optional): additional keyword arguments to pass to
            `func`.


        Returns
        -----------
        * an array of models with shape (N, len(x))

        Raises
        -----------
        """
        # TODO: allow passing args to sample_from_dists
        # TODO: optimize this by doing all sampling first?
        sample_args = [self.sample(cache_sample=False) for i in range(N)]
        models = _np.array([func(x, *sample_args[i], **func_kwargs) for i in range(N)])
        return models

    def plot_sample(self, **kwargs):
        # then we need to do a corner plot
        if not _has_corner:
            raise ImportError("corner must be installed to plot multivariate distributions.  Either install corner or pass a value to dimension to plot a 1D distribution.")


        return corner.corner(self.sample(size=int(1e5), cache_sample=False), labels=[dist._xlabel() for dist in self.dists], **kwargs)

    def plot(self, **kwargs):
        """
        """
        return self.plot_sample(**kwargs)

    def plot_func(self, func, x, N=1000, func_kwargs={}, show=False):
        """
        Draw samples from a callable function and plot.

        The passed callable `func` will be called with arguments `x` followed by
        the individually drawn values from each distribution in `dists` (in order
        provided) and then any additional `func_kwargs`.

        See also:

        * <<class>.sample_func>
        * <<class>.sample>

        Arguments
        -----------
        * `func` (callable): callable function
        * `x` (array like): x values to pass to `func`.
        * `N` (int, optional, default=1000): number of samples to draw.
        * `func_kwargs` (dict, optional): additional keyword arguments to pass to
            `func`.
        * `show` (bool, optional, default=False): whether to call plt.show()

        Returns
        -----------
        * list of created matplotlib artists

        Raises
        -----------
        * ImportError: if matplotlib is not imported
        """


        if not _has_mpl:
            raise ImportError("plot_func requires matplotlib.")

        models = self.sample_func(func, x, N=N, func_kwargs=func_kwargs)

        # TODO: allow options for sigma boundaries
        bounds = _np.percentile(models, 100 * _norm.cdf([-2, -1, 1, 2]), axis=0)

        ret1 = _plt.fill_between(x, bounds[0, :], bounds[-1, :],
                         label="95\% uncertainty", facecolor="#03A9F4", alpha=0.4)
        ret2 = _plt.fill_between(x, bounds[1, :], bounds[-2, :],
                         label="68\% uncertainty", facecolor="#0288D1", alpha=0.4)

        if show:
            _plt.show()

        return ret1, ret2



########################## UNIVARIATE DISTRIBUTIONS ############################


class Composite(BaseUnivariateDistribution):
    """
    A composite distribution consisting of some math operator between one or two
    other Distribution objects.

    For example:

    ```py
    g = distl.gaussian(10, 2)
    u = distl.gaussian(1, 5)

    c = g * u
    print(c)
    ```

    or:

    ```py
    import numpy as np
    g = distl.gaussian(0, 1)
    sin_g = np.sin(g)
    print(sin_g)
    ```

    Currently supported operators include:

    * multiplication, division, addition, subtraction
    * np.sin, np.cos, np.tan (but not math.sin, etc)
    * bitwise and (&), bitwise or (|)

    When doing math between a distribution and a float or integer, that float/int
    will be treated as a <Delta> distribution.  In some simple cases, the
    applicable distribution type will be returned, but in other cases,
    a <Composite> distribution will be returned.  For example, multiplying
    a <Uniform> or <Gaussian> distribution by a float will return another
    <Uniform> or <Gaussian> distribution, respectively.

    Limitations and treatment "under-the-hood":

    * &: the pdfs of the two underlying distributions are sampled over their
        99.99\% intervals and multiplied to create a new pdf.  A spline is then
        fit to the pdf and integrated to create the cdf (which is inverted to
        create the ppf function).  Each of these are then linearly interpolated
        to create the underlying scipy.stats object.  This object is then used
        for sampling as well as accessing the <Composite.pdf>, <Composite.cdf>,
        <Composite.ppf>, etc.  For this reason, the and operator does not support
        retaining covariances at all.

    * |: the pdfs and cdfs of the two underlying distributions are sampled over their
        99.9\% intervals and added to create the new pdfs and cdfs, respectively
        (and the cdf inverted to create the ppf function).  Each of these are then
        linearly interpolated to create the underlying scipy.stats object.  This
        object is then used for any call to the underlying call EXCEPT for sampling.
        Sampling is handled by randomly choosing which child distribution to sample
        from and then sampling from that distribution.  Or operators are therefore
        able to retain covariances for <Composite.sample>, but not for any calls
        to <Composite.pdf>, <Composite.cdf>, or <Composite.ppf>.

    * all others: sampling is handled by sampling the underyling children and
        therefore can retain covariances.  The pdfs, cdfs, and ppfs are
        created by taking 1 million samples, converting to a <Histogram>,
        and linearly interpolating between the bins, thereby losing all covariances.

    """
    def __init__(self, math, dists, unit=None, label=None, wrap_at=None):
        """
        Create a <Composite> distribution from two other distributions.

        Most likely, users will create Composite objects through math operators
        directly.  See examples on the <Composite> overview page.

        Arguments
        ----------
        * `math`: operator to be used between the `dists`.  Must
            be a valid and implemented operator.
        * `dists` (list of distribution objects): distribution objects
            to apply `math` operator.  Some operators (e.g. sin, cos, tan) only
            take one distribution as an argument, but most require 2 or more.
        * `unit` (astropy.units object, optional): the units of the provided values.
        * `label` (string, optional): a label for the distribution.  This is used
            for the x-label while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        ---------
        * a <Composite> object.
        """
        super(Composite, self).__init__(unit, label, wrap_at,
                                        _stats_custom.generic_pdf_cdf_ppf, ('_pdf_cdf_ppf_callables'),
                                        math=math, dists=dists)

        if label is None:
            if len(dists) == 1:
                if dists[0].label is not None:
                    self.label = '{}({})'.format(math, dists[0].label)
            else:
                if _np.all([d.label is not None for d in dists]):
                    self.label = ' {} '.format(_math_symbols.get(math, math)).join([d.label for d in dists])


        def recursive_math(math, items):
            if len(items) > 2:
                return getattr(items[0], math)(items[1])
            return getattr(items[0], math)(recursive_math(math, items[1:]))

        if _has_astropy:
            if len(dists) == 1:
                if dists[0].unit is None:
                    # trig always gives unitless
                    self.unit = _units.dimensionless_unscaled
            elif _np.all([d.unit is not None for d in dists]):
                if math in ['__add__', '__sub__', '__and__', '__or__']:
                    # all units must match
                    if _np.all([d.unit==dists[0].unit for d in dists]):
                        self.unit = dists[0].unit
                    else:
                        # TODO: if they're convertible, we should handle the scaling automatically?
                        raise ValueError("units do not match for {} operator".format(math))
                elif hasattr(dists[0].unit, math):
                    self.unit = recursive_math(math, [d.unit for d in dists])
                else:
                    raise ValueError("cannot determine new units from {}".format(" {} ".format(_math_symbols.get(math, math)).join([d.unit for d in dists])))

        # do some paperwork so changes to descriptors in the children bubble
        # up and will call self._dist_constructor_object_clear_cache()
        for dist in dists:
            dist._parents_with_constructor_object_cache.append(self)

    @property
    def math(self):
        """
        operator to be used between the <Composite.dists>.  Must be a valid and
        implemented operator.
        """
        return self._math

    @math.setter
    def math(self, value):
        self._math = is_math(value)

    @property
    def dists(self):
        """
        distribution objects to apply <Composite.math> operator.  Some operators (e.g. sin,
        cos, tan) only take one distribution as an argument, but most require
        2 or more.
        """
        return self._dists

    @dists.setter
    def dists(self, value):
        if isinstance(value, BaseDistribution):
            value = [value]

        if not (isinstance(value, tuple) or isinstance(value, list)):
            raise TypeError("dists must be a tuple or list of distributions, got {}".format(type(value)))

        for i,dist in enumerate(value):
            if isinstance(dist, dict):
                value[i] = from_dict(dist)

        if not _all_in_types(value, (BaseUnivariateDistribution, BaseMultivariateSliceDistribution)):
            raise TypeError("dists must be a tuple or list of distributions (univariate or multivariate slices)")

        if len(value)==1 and self.math not in ['sin', 'cos', 'tan']:
            raise ValueError("math with operator '{}' requires more than one distribution".format(self.math))

        self._dists = value

    def __repr__(self):
        return "<distl.{} {} unit={}>".format(self.__class__.__name__.lower(), self.__str__(), self.unit if self.unit is not None else "None")

    def __str__(self):
        if len(self.dists) == 1:
            return "{}({})".format(_math_symbols.get(self.math, self.math), self.dists[0].__str__())
        else:
            return " {} ".format(_math_symbols.get(self.math, self.math)).join(d.__str__() for d in self.dists)

    @property
    def hash(self):
        """
        """
        # NOTE (IMPORTANT): then we are going to "forget" these when
        # nesting CompositeDistributions
        # return super(CompositeDistribution, self).hash()
        return ",".join([str(d.hash) for d in self.dists])

    ### SAMPLE CACHING

    @property
    def cached_sample_children(self):
        return _np.asarray([d.cached_sample for d in self.dists])

    def clear_cached_sample(self):
        super(Composite, self).clear_cached_sample()
        for d in self.dists:
            d.clear_cached_sample()

    @property
    def _pdf_cdf_ppf_callables(self):
        # TODO: how do we need to handle units here... do we always need to
        # convert to SI before doing anything?  Or keep quantities?  Or do math
        # on units after?

        if self.math in ['__and__', '__or__']:
            ranges = [d.interval(0.9999, wrap_at=False) for d in self.dists]
            # dist1range = self.dist1.interval(0.9999, wrap_at=False)
            # dist2range = self.dist2.interval(0.9999, wrap_at=False)

            # we'll set the sampling so each distribution gets its range sample 1000 times, append, and then sort
            x = _np.concatenate([_np.linspace(r[0]-0.1*(r[1]-r[0]),
                                              r[1]+0.1*(r[1]-r[0]),
                                              int(1e4)) for r in ranges])

            # TODO: do we need to remove duplicates?
            x.sort()
            if self.math == '__and__':
                pdf = _np.product([d.pdf(x) for d in self.dists], axis=0)
                # unfortunately we'll need to integrate to get the cdf... we'll do that later
                cdf = None
            elif self.math == '__or__':
                pdf = _np.sum([d.pdf(x) for d in self.dists], axis=0)
                cdf = _np.sum([d.cdf(x) for d in self.dists], axis=0)
            else:
                raise NotImplementedError()

            # make sure pdf is normalized correctly
            pdf_integral = _np.sum(pdf[1:]*abs(x[1:]-x[:-1]))
            pdf /= pdf_integral

            pdf_call = _stats_custom.interpolate_callable(x, pdf)

            if cdf is None:
                # print("*** integrating to compute cdf over spline (this may be slow... we'll eventually cache this so it only needs to be done once until an attribute is changed)")
                spline = _interpolate.UnivariateSpline(x, pdf, k=1, s=0)
                def _spline_int_single(xi):
                    return spline.integral(x[0], xi)

                cdf_vec = _np.vectorize(_spline_int_single)
                cdf = cdf_vec(x)

            # make sure cdf is normalized correctly
            cdf /= cdf[-1]

            ppf_call = _stats_custom.interpolate_callable(cdf, x)

            # make sure interpolation on the right always gives 1, not the fill_value of 0
            cdf_call = _stats_custom.interpolate_callable(_np.append(x, _np.inf), _np.append(cdf, 1.0))

            return pdf_call, cdf_call, ppf_call

        else:
            # TODO: how do we send reasonable defaults here to know how to bin?
            # Should we look at the ranges of the children like we do for
            # and/or?
            return self.to_histogram(N=int(1e6), bins=100, wrap_at=False)._pdf_cdf_ppf_callables

    @property
    def dist_constructor_args(self):
        return self._pdf_cdf_ppf_callables

    def _sample_from_children(self, math, dists, seed={}, size=None, cache_sample=True):
        if math == '__and__':
            raise NotImplementedError("cannot sample from children with & logic")
        elif self.math == '__or__':
            # choose randomly between the two child Distributions
            choice = _np.random.randint(0,len(dists))
            if size is None:
                dist = dists[choice]
                return dist.sample(size=size, seed=seed, cache_sample=cache_sample, as_quantity=_has_astropy and self.unit not in [None, _units.dimensionless_unscaled])
            else:
                # NOTE: // for python2 and 3 will do floor division, returning an integer
                sizes = [size//len(dists)]*len(dists)
                remainder = (sizes[0] % len(dists))
                for i in range(remainder):
                    choice = _np.random.randint(0, len(dists))
                    sizes[choice] += 1

                return _np.concatenate([d.sample(size=s, seed=seed, cache_sample=cache_sample, as_quantity=_has_astropy and self.unit not in [None, _units.dimensionless_unscaled]) for d,s in zip(dists, sizes)])

        elif len(dists) > 1:
            # NOTE: this will account for multivariate, but only for THESE 2
            # if there are nested CompositeDistributions, then the seed will be lost

            # samples = sample_from_dists((dist1, dist2), seeds=seed, size=size)
            # TODO: OPTIMIZE: should we cache the collection? (in which case we should pass cache_sample)
            samples = DistributionCollection(*dists).sample(seeds=seed, size=size, cache_sample=cache_sample)
            if size is not None:
                return getattr(samples[:,0], math)(samples[:,1])
            else:
                return getattr(samples[0], math)(samples[1])
        else:
            # if math in ['sin', 'cos', 'tan'] and _has_astropy and dist1.unit is not None:
            #     unit = _units.rad
            # else:
            #     unit = None
            return getattr(_np, math)(dists[0].sample(size=size, seed=seed, cache_sample=cache_sample, as_quantity=_has_astropy and self.unit not in [None, _units.dimensionless_unscaled]))


    def sample(self, size=None, unit=None, as_quantity=False, wrap_at=None, seed={}, as_univariate=False, cache_sample=True):
        """
        Sample from the <Composite> distribution.

        Arguments
        -----------
        * `size` (int or tuple or None, optional, default=None): size/shape of the
            resulting array.
        * `unit` (astropy.unit, optional, default=None): unit to convert the
            resulting sample(s).  Astropy must be installed in order to convert
            units.
        * `as_quantity` (bool, optional, default=False): whether to return an
            astropy quantity object instead of just the value.  Astropy must
            be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `seed` (dict, optional, default={}): seeds (as hash: seed pairs) to
            pass to underlying distributions.
        * `as_univariate` (bool, optional, default=False): whether to draw from
            the flattend <<class>.pdf> rather than from the children distributions.
            If True, any underlying covariances from <BaseMultivariateSliceDistribution>
            objects will be ignored.  This may be slightly faster, especially
            with repeated calls.  Note that `as_univariate` is ignored for
            <Composite> distributions with 'and' logic as these are always
            sampled from the combined pdf.
        * `cache_sample` (bool, optional, default=True): whether to override the
            existing <<class>.cached_sample>.

        Returns
        ---------
        * float or array: float if `size=None`, otherwise a numpy array with
            shape defined by `size`.
        """
        if as_univariate or self.math in ['__and__']:
            # fallback on using the interpolated combined pdf/cdf/ppf

            # TODO: test if doing this will cause issues with caching expecting the children to have cache filled
            for d in self.dists:
                d.clear_cached_sample()

            return super(Composite, self).sample(size=size, unit=unit, as_quantity=as_quantity, wrap_at=wrap_at, seed=seed, cache_sample=cache_sample)
        else:
            # NOTE: even though in these cases we sample from the underlying children
            # (and therefore can account for covariances from multivariate children),
            # calls to pdf/cdf/ppf will still need to merge and interpolate
            # and will ignore these covariances.

            sample = self._sample_from_children(self.math, self.dists, size=size, seed=seed, cache_sample=cache_sample)

            if isinstance(sample, _units.Quantity):
                if sample.unit is None or sample.unit in [_units.dimensionless_unscaled]:
                    sample = sample.value
                else:
                    sample = sample.to(self.unit).value

            if cache_sample:
                self._cached_sample = sample

            return self._return_with_units(self.wrap(sample, wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)



    def to_gaussian(self, N=1000, bins=10, range=None):
        """
        Convert the <Composite> distribution to a <Gaussian> distribution via
        a <Histogram> distribution.

        Under-the-hood, this calls <Composite.to_histogram> with the requested
        values of `N`, `bins`, and `range` and then calls <Histogram.to_gaussian>.

        Arguments
        -----------
        * `N` (int, optional, default=1000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.

        Returns
        --------
        * a <Gaussian> object
        """
        return self.to_histogram(N=N, bins=bins, range=range).to_gaussian()

    def to_uniform(self, sigma=1.0, N=1000, bins=10, range=None):
        """
        Convert the <Composite> distribution to a <Uniform> distribution via
        a <Histogram> distribution.

        Under-the-hood, this calls <Composite.to_histogram> with the requested
        values of `N`, `bins`, and `range` and then calls <Histogram.to_uniform>
        with the requested value of `sigma`.

        Arguments
        -----------
        * `sigma` (float, optional, default=1.0): the number of standard deviations
            to adopt as the lower and upper bounds of the uniform distribution.
        * `N` (int, optional, default=1000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.

        Returns
        --------
        * a <Uniform> object
        """
        return self.to_histogram(N=N, bins=bins, range=range).to_uniform(sigma=sigma)


class Histogram(BaseUnivariateDistribution):
    """
    A Histogram distribution stores a discrete PDF and allows sampling from
    that binned density distribution.

    To create a Histogram distribution from already binned data, see
    <distl.histogram_from_bins> or <Histogram.__init__>.  To create a
    Histogram distribtuion from the data array itself, see
    <distl.histogram_from_data> or <Histogram.from_data>.

    Treatment under-the-hood:

    The densities at each bin-midpoint are linearly interpolated to create
    a pdf (which is normalized to an integral of 1).  A numerical integral
    of the bins is then performed to create the cdf (again, normalized to 1)
    and inverted to create the ppf.  Each of these are then interpolated
    whenever accessing <Histogram.pdf>, <Histogram.cdf>, <Histogram.ppf>, etc as
    well as used when calling <Histogram.sample>.
    """
    def __init__(self, bins, density, unit=None, label=None, wrap_at=None):
        """
        Create a <Histogram> distribution from bins and density.

        This can also be created from a function at the top-level as:

        * <distl.histogram_from_bins>

        See also:

        * <Histogram.from_data>
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
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <Histogram> object
        """
        super(Histogram, self).__init__(unit, label, wrap_at,
                                        _stats_custom.generic_pdf_cdf_ppf, ('_pdf_cdf_ppf_callables'),
                                        bins=bins, density=density)

    @property
    def bins(self):
        """
        the value of the bin-edges.  Must have one more entry than <Histogram.density>.
        """
        return self._bins

    @bins.setter
    def bins(self, value):
        self._bins = is_1d_array(value)

    @property
    def density(self):
        """
        the value of the bin-densities.  Must have one less entry than <Histogram.bins>.
        """
        return self._density

    @density.setter
    def density(self, value):
        self._density = is_1d_array(value)

    @classmethod
    def from_data(cls, data, bins=10, range=None, weights=None,
                  label=None, unit=None, wrap_at=None):
        """
        Create a <Histogram> distribution from data.  Note that under-the-hood
        a linear interpolator is used between the bins for the pdf, cdf, and ppf
        functions (and for sampling).

        This can also be created from a function at the top-level as:

        * <distl.histogram_from_data>

        See also:

        * <Histogram.__init__>
        * <distl.histogram_from_bins>

        Arguments
        --------------
        * `data` (np.array object): 1D array of values.
        * `bins` (int or array, optional, default=10): number of bins or value
            of bin edges.  Passed to np.histogram.
        * `range` (tuple, optional, default=None): passed to np.histogram.
        * `weights` (array, optional, default=None): passed to np.histogram.
        * `unit` (astropy.units object, optional): the units of the provided values.
        * `label` (string, optional): a label for the distribution.  This is used
            for the x-label while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <Histogram> object
        """
        hist, bin_edges = _np.histogram(data, bins=bins, range=range, weights=weights, density=True)

        return cls(bin_edges, hist, label=label, unit=unit, wrap_at=wrap_at)

    @property
    def _pdf_cdf_ppf_callables(self):
        return _hist_pdf_cdf_ppf_callables(self.bins, self.density)

    @property
    def dist_constructor_args(self):
        return self._pdf_cdf_ppf_callables

    def to_gaussian(self):
        """
        Convert the <Histogram> distribution to a <Gaussian> distribution by
        adopting the values of <Histogram.median> and <Histogram.std>.

        Returns
        --------
        * a <Gaussian> object
        """
        dco = self.dist_constructor_object
        return Gaussian(dco.median(), dco.std(), label=self.label, unit=self.unit, wrap_at=self.wrap_at)

    def to_uniform(self, sigma=1.0):
        """
        Convert the <Histogram> distribution to a <Uniform> distribution via
        a <Gaussian> distribution.

        Under-the-hood, this calls <Histogram.to_gaussian> and then calls
        <Gaussian.to_uniform> with the requested value of `sigma`.

        Arguments
        -----------
        * `sigma` (float, optional, default=1.0): the number of standard deviations
            to adopt as the lower and upper bounds of the uniform distribution.

        Returns
        --------
        * a <Uniform> object
        """
        return self.to_gaussian().to_uniform(sigma=sigma)

class Delta(BaseUnivariateDistribution):
    """
    A Delta distribution will _always_ return the central values.  In most cases,
    there is no need to manually create a Delta distribution.  But when doing
    math on other <BaseDistribution> objects, <Delta> distributions are often
    created for clarity.

    Can be created from the top-level via the <distl.delta> convenience function.
    """
    def __init__(self, loc=0.0, unit=None, label=None, wrap_at=None):
        """
        Create a <Delta> distribution.

        This can also be created from a function at the top-level as:

        * <distl.delta>

        Arguments
        --------------
        * `loc` (float or int, default=0.0): the loc at which the delta function is True.
        * `unit` (astropy.units object, optional): the units of the provided values.
        * `label` (string, optional): a label for the distribution.  This is used
            for the x-label while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <Delta> object
        """
        super(Delta, self).__init__(unit, label, wrap_at,
                                    _stats_custom.delta, ('loc',),
                                    loc=loc)

    @property
    def loc(self):
        """
        the loc at which the delta function is True.
        """
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = is_float(value)

    def __mul__(self, other):
        if isinstance(other, Delta):
            other = other.loc

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.loc *= other
            return dist

        return super(Delta, self).__mul__(other)

    def __div__(self, other):
        return self.__mul__(1./other)

    def __add__(self, other):
        if isinstance(other, Delta):
            other = other.loc

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.loc += other
            return dist

        return super(Delta, self).__add__(other)

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __float__(self):
        return self.loc

    def to_uniform(self):
        """
        Convert the <Delta> distribution to a <Uniform> distribution in which
        both the lower and upper bounds are the same as the value.

        Returns
        ----------
        * a <Uniform> object
        """
        low = self.loc
        high = self.loc
        return Uniform(low, high, label=self.label, unit=self.unit, wrap_at=self.wrap_at)

    def to_gaussian(self):
        """
        Convert the <Delta> distribution to a <Gaussian> distribution in which
        the central value is adopted with a scale/sigma of 0.0.

        See also:

        * <Delta.mean>
        * <Delta.std>

        Returns
        --------
        * a <Gaussian> object
        """
        return Gaussian(self.loc, 0.0, label=self.label, unit=self.unit, wrap_at=self.wrap_at)


class Gaussian(BaseUnivariateDistribution):
    """
    A Gaussian (or Normal) distribution uses [scipy.stats.norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
    to sample values from a gaussian function.

    Can be created from the top-level via the <distl.gaussian> or
    <distl.normal> convenience functions.
    """
    def __init__(self, loc=0.0, scale=1.0, unit=None, label=None, wrap_at=None):
        """
        Create a <Gaussian> distribution.

        This can also be created from a function at the top-level as:

        * <distl.gaussian>

        Arguments
        --------------
        * `loc` (float or int, default=0.0): the central value (mean) of the
            gaussian distribution.
        * `scale` (float or int, default=1.0): the scale (sigma) of the gaussian
            distribution.
        * `unit` (astropy.units object, optional): the units of the provided values.
        * `label` (string, optional): a label for the distribution.  This is used
            for the x-label while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <Gaussian> object
        """
        super(Gaussian, self).__init__(unit, label, wrap_at,
                                       _stats.norm, ('loc', 'scale'),
                                       loc=loc, scale=scale)


    @property
    def loc(self):
        """
        the central value (mean) of the gaussian distribution.
        """
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = is_float(value)

    @property
    def scale(self):
        """
        the scale (sigma) of the gaussian distribution.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = is_float(value)

    def __mul__(self, other):
        if isinstance(other, Delta):
            other = other.loc

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.loc *= other
            dist.scale *= other
            return dist

        return super(Gaussian, self).__mul__(other)

    def __div__(self, other):
        return self.__mul__(1./other)

    def __add__(self, other):
        if isinstance(other, Delta):
            other = other.loc

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.loc += other
            return dist

        return super(Gaussian, self).__add__(other)

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __float__(self):
        return self.loc

    def to_uniform(self, sigma=1.0):
        """
        Convert the <Gaussian> distribution to a <Uniform> distribution by
        adopting the lower and upper bounds as a certain value of `sigma`
        for the <Gaussian> distribution.

        Arguments
        ----------
        * `sigma` (float, optional, default=1.0): number of standard deviations
            which should be adopted as the lower and upper bounds of the
            <Uniform> distribution.

        Returns
        ---------
        * a <Uniform> distribution
        """
        low = self.loc - self.scale * sigma
        high = self.loc + self.scale * sigma
        return Uniform(low, high, label=self.label, unit=self.unit, wrap_at=self.wrap_at)


class Uniform(BaseUnivariateDistribution):
    """
    A Uniform (or Boxcar) distribution gives equal weights to all values within
    the defined range and uses [scipy.stats.uniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html)
    to sample values.

    Can be created from the top-level via the <distl.uniform> or
    <distl.boxcar> convenience functions.
    """
    def __init__(self, low=0.0, high=1.0, unit=None, label=None, wrap_at=None):
        """
        Create a <Uniform> distribution.

        This can also be created from a function at the top-level as:

        * <distl.uniform>

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
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <Uniform> object
        """
        super(Uniform, self).__init__(unit, label, wrap_at,
                                       _stats.uniform, ('low', 'width'),
                                       low=low, high=high)

    @property
    def low(self):
        """
        the lower limit of the uniform distribution.
        """
        return self._low

    @low.setter
    def low(self, value):
        self._low = is_float(value)

    @property
    def high(self):
        """
        the upper limits of the uniform distribution.

        Must be higher than <Uniform.low> unless <Uniform.wrap_at> is provided or
        <Uniform.unit> is provided as angular (rad, deg, cycles).
        """
        return self._high

    @high.setter
    def high(self, value):
        self._high = is_float(value)

    @property
    def width(self):
        """
        Access the width of the <Uniform> distribution, defined as
        <Uniform.high> - <Uniform.low>

        Returns
        ----------
        * float
        """
        return self.high - self.low

    def __mul__(self, other):
        if isinstance(other, Delta):
            other = other.loc

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.low *= other
            dist.high *= other
            return dist

        return super(Uniform, self).__mul__(other)


    def __div__(self, other):
        return self.__mul__(1./other)

    def __add__(self, other):
        if isinstance(other, Delta):
            other = other.loc

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.low += other
            dist.high += other
            return dist
        # elif isinstance(other, Uniform):
            ## NOTE: this does not seem to be true as we should get a trapezoid if sampling separately
            # dist = self.copy()
            # dist.low += other.low
            # dist.high += other.high
            # return dist

        return super(Uniform, self).__add__(other)


    def __sub__(self, other):
        return self.__add__(-1*other)

    def to_gaussian(self, sigma=1.0):
        """
        Convert the <Uniform> distribution to a <Gaussian> distribution by
        adopting a certain `sigma`: number of standard deviations which should
        be adopted as the lower and upper bounds of the <Uniform> distribution.

        Arguments
        ----------
        * `sigma` (float, optional, default=1.0): number of standard deviations
            which should be adopted as the lower and upper bounds of the
            <Uniform> distribution.

        Returns
        ---------
        * a <Gaussian> distribution
        """
        loc = self.median()
        scale = (self.high - self.low) / (2.0 * sigma)
        return Gaussian(loc, scale, unit=self.unit, label=self.label, wrap_at=self.wrap_at)

######################### MULTIVARIATE DISTRIBUTIONS ###########################

class MVGaussian(BaseMultivariateDistribution):
    def __init__(self, mean=0.0, cov=1.0, allow_singular=False,
                 units=None, labels=None, wrap_ats=None):
        """
        A Multivariate Gaussian distribution uses [scipy.stats.multivariate_normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html)
        to sample values from a multivariate gaussian/normal function.

        This can also be created from a function at the top-level as:

        * <distl.mvgaussian>

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
            for the x-labels while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_ats` (list of floats, None, or False, optional, default=None): values to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <MVGaussian> object
        """
        super(MVGaussian, self).__init__(units, labels, wrap_ats,
                                         _stats.multivariate_normal, ('mean', 'cov', 'allow_singular'),
                                         mean=mean, cov=cov, allow_singular=allow_singular)


    @property
    def mean(self):
        """
        the central value of the multivariate gaussian distribution.
        """
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = is_iterable(value)

    @property
    def cov(self):
        """
        the covariance matrix of the multivariate gaussian distribution
        """
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov = is_square_matrix(value)

    @property
    def allow_singular(self):
        """
        passed directly to scipy (see [scipy.stats.multivariate_normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html))
        """
        return self._allow_singular

    @allow_singular.setter
    def allow_singular(self, value):
        self._allow_singular = is_bool(value)

    @property
    def ndimensions(self):
        """
        Access the number of dimensions in the <MVGaussian> distribution.

        Returns
        --------
        * int
        """
        return len(self.mean)

    def slice(self, dimension):
        """
        Take a single dimension from the multivariate distribution while
        retaining the covariances.  The returned <MVGaussianSlice> object
        keeps the full multivariate distribution while acting somewhat
        like a univariate distribution.

        See also:

        * <MVGaussian.take_dimensions>
        * <<class>.to_histogram>
        * <<class>.to_gaussian>
        * <MVGaussianSlice.dimension>

        Arguments
        ----------
        * `dimension` (int or string): the label or index of the dimension to
            take.

        Returns
        ------------
        * <MVGaussianSlice> object
        """
        return MVGaussianSlice(self, dimension)

    def take_dimensions(self, dimensions):
        """
        Take multiple dimensions from the multivariate distribution (and remove
        all others), returning another <MVGaussian> object.

        See also:

        * <MVGaussian.slice>
        * <MVGaussian.to_univariate>

        Arguments
        ----------
        * `dimension` (list of strings or ints): the labels or indices of the
            dimensions to include in the new distribution.

        Returns
        ----------
        * <MVGaussian> object or <Gaussian> if only one dimension provided
        """
        if isinstance(dimensions, int) or isinstance(dimensions, str) or isinstance(dimensions, unicode):
            dimensions = [dimensions]

        dimensions = [self._get_dimension_index(d) for d in dimensions]

        if len(dimensions) == 1:
            return self.to_univariate(dimensions[0])

        mean = _np.asarray(self.mean)[dimensions]
        cov = _np.asarray(self.cov)[dimensions, :][:, dimensions]

        return MVGaussian(mean=mean, cov=cov,
                          units=[self.units[d] for d in dimensions] if self.units is not None else None,
                          labels=[self.labels[d] for d in dimensions] if self.labels is not None else None,
                          wrap_ats=[self.wrap_ats[d] for d in dimensions] if self.wrap_ats is not None else None)

    def to_mvhistogram(self, N=1e6, bins=15, range=None):
        """
        Convert the <MVGaussian> distribution to an <MVHistogram> distribution.

        Under-the-hood, this calls <<class>.sample> with `size=N` and `wrap_at=False`
        and passes the resulting array as well as the requested `bins` and `range`
        to <MVHistogram.from_data>.

        Arguments
        -----------
        * `N` (int, optional, default=1e6): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=15): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.

        Returns
        --------
        * an <MVHistogram> object
        """
        # TODO: if sample is updated to take wrap_at/wrap_ats... pass wrap_at=False here
        return MVHistogram.from_data(self.sample(size=int(N), cache_sample=False),
                                     bins=bins, range=range,
                                     units=self.units, labels=self.labels, wrap_ats=self.wrap_ats)

    def to_univariate(self, dimension):
        """
        Shortcut to <MVGaussian.to_gaussian>
        """
        return self.to_gaussian(dimension=dimension)

    def to_gaussian(self, dimension):
        """
        Convert the <MVGaussian> distribution to a <Gaussian> univariate distribution.

        Arguments
        -----------
        * `dimension` (int or str): index or label of the dimension to use for
            the univariate distribution.

        Returns
        ----------
        * a <Gaussian> object
        """
        dimension = self._get_dimension_index(dimension)

        return Gaussian(loc=self.mean[dimension], scale=_np.sqrt(self.cov[dimension, dimension]),
                         unit=self.units[dimension] if self.units is not None else None,
                         label=self.labels[dimension] if self.labels is not None else None,
                         wrap_at=self.wrap_ats[dimension] if self.wrap_ats is not None else None)

    def to_histogram(self, dimension, N=100000, bins=10, range=None, wrap_at=None):
        """
        Convert the <MVGaussian> distribution to a <Histogram> univariate distribution.

        Under-the-hood, this calls <<class>.to_gaussian> and then
        <Gaussian.to_histogram>.

        Arguments
        -----------
        * `dimension` (int or str): index or label of the dimension to use for
            the univariate distribution.
        * `N` (int, optional, default=100000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.
        * `wrap_at` (float or None, optional, default=None): value to set for
            `wrap_at` of the returned <Histogram>.  If None or not provided,
            will default to <<class>.wrap_at>.

        Returns
        --------
        * a <Histogram> object
        """
        self.to_gaussian(dimension).to_histogram(N=N, bins=bins, range=range, wrap_at=wrap_at)

class MVGaussianSlice(BaseMultivariateSliceDistribution):
    @property
    def dist_constructor_func(self):
        return _stats.norm

    @property
    def dist_constructor_argnames(self):
        return 'loc', 'scale'

    @property
    def dist_constructor_args(self):
        return self.loc, self.scale

    @property
    def loc(self):
        return self.multivariate.mean[self.dimension]


    @property
    def scale(self):
        return _np.sqrt(self.multivariate.cov[self.dimension, self.dimension])


class MVHistogram(BaseMultivariateDistribution):
    """

    Treatment under-the-hood:

    * When sampling, a random value between 0 and 1 is drawn.  The N-dimensional
    bins are then unraveled and integrated to create a flattened cdf.  The
    cdf is then linearly interpolated to find the index of the unraveled bins
    in which to sample, as well as the relative location in the bin.  The selected
    bin is then artificially subdivided by the same shape grid as the original
    binning and linearly interpolated based on the remainder to return a single
    value for <MVHistogram.sample>.

    * Means and covariances (see <MVHistogram.calculate_means_covariances>,
    <MVHistogram.calculate_means>, <MVHistogram.calculate_covariances>) are calculated
    by sampling (with a default size of 1e5), and determining the mean and covariances
    on that sample.

    """
    def __init__(self, bins, density, units=None, labels=None, wrap_ats=None):
        """
        Create an <MVHistogram> distribution from bins and density.

        See also:

        * <MVHistogram.from_data>
        * <distl.mvhistogram_from_data>

        Arguments
        --------------
        * `bins` (np.array object): the value of the bin-edges (n-dimensional).
        * `density` (np.array object): the value of the bin-densities (n-dimensional).
        * `units` (list of astropy.units objects, optional): the units of the provided values.
        * `labels` (list of strings, optional): labels for each dimension in the
            distribution.  This is used
            for the x-labels while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_ats` (list of floats, None, or False, optional, default=None): values to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * an <MVHistogram> object
        """
        super(MVHistogram, self).__init__(units, labels, wrap_ats,
                                          None, None,
                                          bins=bins, density=density)

    @property
    def bins(self):
        """
        the value of the bin-edges (n-dimensional).
        """
        return self._bins

    @bins.setter
    def bins(self, value):
        self._bins = is_nd_array(value)

    @property
    def density(self):
        """
        the value of the bin-densities (n-dimensional).
        """
        return self._density

    @density.setter
    def density(self, value):
        self._density = is_nd_array(value)

    @classmethod
    def from_data(cls, data, bins=10, range=None, weights=None,
                  units=None, labels=None, wrap_ats=None):
        """
        """
        # TODO:  what version of numpy introduced density?  Do we need a try/except or to check the version?
        try:
            hist, bin_edges = _np.histogramdd(data, bins=bins, range=range, weights=weights, density=True)
        except TypeError:
            hist, bin_edges = _np.histogramdd(data, bins=bins, range=range, weights=weights, normed=True)


        return cls(_np.asarray(bin_edges), hist, units=units, labels=labels, wrap_ats=wrap_ats)

    def pdf(self, x, unit=None):
        # TODO: N-dimension interpolation of (self.bins, self.density)
        raise NotImplementedError("pdf not supported for {} distribution".format(self.__class__.__name__))

    def logpdf(self, x, unit=None):
        raise NotImplementedError("logpdf not supported for {} distribution".format(self.__class__.__name__))

    @property
    def _cdf_per_bin(self):
        cdf = _np.cumsum(self.density)
        cdf /= float(cdf[-1])
        return cdf

    def cdf(self, x, unit=None):
        # TODO: N-dimensional interpolation of (self.bins, self._cdf_per_bin)
        raise NotImplementedError("cdf not supported for {} distribution".format(self.__class__.__name__))

    def logcdf(self, x, unit=None):
        raise NotImplementedError("logcdf not supported for {} distribution".format(self.__class__.__name__))

    def _ppf(self, q, dimension=None):
        # this is hidden because although it does work for random drawing, I'm
        # not sure calling it ppf is really fair.


        # adapted from: https://stackoverflow.com/a/17822210
        #
        # MV case
        # density, bins = np.histogramdd(chain_flat, normed=True)
        # bins = np.asarray(bins)
        #
        # 1D case
        # density, bins = np.histogram(np.random.rand(1000), normed=True)
        # bins = np.asarray([bins])

        if dimension is not None:
            dimension = self._get_dimension_index(dimension)

        if _np.any(q > 1) or _np.any(q < 0):
            raise ValueError("q must be between 0 and 1")

        if isinstance(q, float):
            return_single = True
            q = _np.asarray([q])
        else:
            return_single = False
            q = _np.asarray(q)

        # this is cdf on the unraveled densities
        cdf = self._cdf_per_bin

        unraveled_index_float = _np.interp(q, cdf, range(len(cdf)))
        # print("unraveled_index_float: {}".format(unraveled_index_float))

        unraveled_index = unraveled_index_float.astype(int)
        unraveled_index_rem = unraveled_index_float - unraveled_index
        # print("unraveled_index: {}, unraveled_index_rem: {}".format(unraveled_index, unraveled_index_rem))

        # multivariate case
        ind_per_dim = _np.unravel_index(unraveled_index, self.density.shape)
        # now we'll essentially subdivide each bin by the shape of the original grid so that we can linearly interpolate inside the chosen bin
        rem_per_dim = _np.asarray(_np.unravel_index((unraveled_index_rem*len(cdf)).astype(int), self.density.shape)).astype(float) / _np.asarray(self.density.shape)[:, _np.newaxis]
        bin_width_per_dim = _np.row_stack([_np.diff(b) for b in self.bins])
        # print("ind_per_dim: {}".format(ind_per_dim))
        # print("bin_width_per_dim: {}".format(bin_width_per_dim))
        # print("rem_per_dim: {}".format(rem_per_dim))

        # b[ind] is the lower-edge of the bin... so we want to interpolate based on that bin_width
        values_from_bins = _np.column_stack([self.bins[dim,ind_this_dim]+bin_width_per_dim[dim,ind_this_dim]*rem_per_dim[dim] for dim,ind_this_dim in enumerate(ind_per_dim)])

        if return_single:
            if dimension is not None:
                return values_from_bins[:, dimension][0]
            return values_from_bins[0]
        else:
            if dimension is not None:
                return values_from_bins[:, dimension]
            return values_from_bins

    @property
    def ndimensions(self):
        """
        Access the number of dimensions in the <MVHistogram> distribution.

        Returns
        --------
        * int
        """
        return self.bins.shape[0]

    def slice(self, dimension):
        """
        Take a single dimension from the multivariate distribution while
        retaining the covariances.  The returned <MVHistogramSlice> object
        keeps the full multivariate distribution while acting somewhat
        like a univariate distribution.

        See also:

        * <<class>.to_histogram>
        * <<class>.to_gaussian>
        * <MVHistorgramSlice.dimension>

        Arguments
        ----------
        * `dimension` (int or string): the label or index of the dimension to
            take.

        Returns
        ------------
        * <MVHistogramSlice> object
        """
        return MVHistogramSlice(self, dimension)

    def take_dimensions(self, dimensions):
        """
        Take multiple dimensions from the multivariate distribution (and remove
        all others), returning another <MVHistogram> object.

        See also:

        * <MVHistogram.slice>
        * <MVHistogram.to_univariate>

        Arguments
        ----------
        * `dimension` (list of strings or ints): the labels or indices of the
            dimensions to include in the new distribution.

        Returns
        ----------
        * <MVHistogram> object or <Histogram> if only one dimension provided
        """
        if isinstance(dimensions, int) or isinstance(dimensions, str) or isinstance(dimensions, unicode):
            dimensions = [dimensions]

        dimensions = [self._get_dimension_index(d) for d in dimensions]

        if len(dimensions) == 1:
            return self.to_univariate(dimensions[0])

        bins = _np.asarray(self.bins)[dimensions]
        density = _np.sum(self.density, axis=tuple([d for d in range(self.ndimensions) if d not in dimensions]))

        return MVHistogram(bins=bins, density=density,
                           units=[self.units[d] for d in dimensions] if self.units is not None else None,
                           labels=[self.labels[d] for d in dimensions] if self.labels is not None else None,
                           wrap_ats=[self.wrap_ats[d] for d in dimensions] if self.wrap_ats is not None else None)

    def sample(self, size=None, dimension=None, seed=None, cache_sample=True):
        """

        Arguments
        ----------
        * `size`
        * `dimension`
        * `seed` (int, optional): seed to pass to np.random.seed
            prior to sampling.
        * `cache_sample` (bool, optional, default=True): whether to override the
            existing <<class>.cached_sample>.

        """
        # if dimension is not None:
        #     dimension = self._get_dimension_index(dimension)
        #     bins = self.bins[dimension]
        #     density = self.density[dimension]
        # else:
        #     bins = self.bins
        #     density = self.density

        if isinstance(seed, dict):
            seed = seed.get(self.hash, None)

        if seed is not None:
            _np.random.seed(seed)

        q = _np.random.rand(size if size is not None else 1)
        sample = self._ppf(q, dimension=dimension)

        if cache_sample:
            self._cache_sample = sample

        # TODO: units, as_quantity, wrapping
        return sample

    def plot(self, *args, **kwargs):
        """
        """
        # TODO: add plot_mvgaussian or plot_gaussian options to overplot the MVGaussian pdfs/contours

        dimension = kwargs.pop('dimension', None)
        if dimension is not None:
            dimension = self._get_dimension_index(dimension)
        # dimension = self.get_dimension_by_label(kwargs.get('dimension', None))
        if dimension is not None:
            kwargs.setdefault('bins', self.bins[dimension])

        return super(MVHistogram, self).plot(*args, **kwargs)

    def calculate_means_covariances(self, N=1e5):
        """
        Return the weighted mean values and covariances from the histogram.

        See also:

        * <MVHistogram.calculate_means>
        * <MVHistogram.calculate_covariances>

        Arguments
        ---------
        * `N` (int, default=1e5): number of samples to use to pass to
            `np.cov`.

        Returns
        -------
        * means (array of floats), covariances (matrix)
        """
        # means could also be self.ppf(0.5)

        # TODO: pass wrap_at=False once supported
        samples = self.sample(size=int(N), cache_sample=False)
        # means = [_np.mean(samples[:,d] for d in range(self.dimensions))]
        means = _np.mean(samples, axis=0)
        covariances = _np.cov(samples.T)
        return means, covariances

    def calculate_means(self, N=1e5):
        """
        Return the weighted mean values from the histogram.

        See also:

        * <MVHistogram.calculate_covariances>
        * <MVHistogram.calculate_means_covariances>

        Arguments
        ---------
        * `N` (int, default=1e5): number of samples to use to pass to
            `np.cov`.

        Returns
        -------
        * list of floats: the mean value per dimension
        """
        return self.calculate_means_covariances(N=N)[0]

    def calculate_covariances(self, N=1e5):
        """
        Return the covariances about the mean from the histogram.

        Under-the-hood, this calls `np.cov` on the output from <<class>.sample>
        with `N` samples.

        See also:

        * <MVHistogram.calculate_means>
        * <MVHistogram.calculate_means_covariances>

        Arguments
        ---------
        * `N` (int, default=1e5): number of samples to use to pass to
            `np.cov`.

        Returns
        ---------
        * MxM square matrix of floats.
        """
        return self.calculate_means_covariances(N=N)[1]


    def to_mvgaussian(self, N=1e5, allow_singular=False):
        """
        Convert the <MVHistogram> distribution to an <MVGaussian> distribution.

        See also:

        * <MVHistogram.calculate_means>
        * <MVHistogram.calculate_covariances>

        Arguments
        ---------
        * `N` (int, default=1e5): number of samples to use when calling
            <<class>.calculate_means> and <<class>.calculate_covariances>.
        * `allow_singular` (bool, optional, default=False): value to pass to
            <MVGaussian>.

        Returns
        --------
        * an <MVGaussian> object
        """
        mean, cov = self.calculate_means_covariances(N)
        return MVGaussian(mean, cov, allow_singular=allow_singular,
                          units=self.units, labels=self.labels, wrap_ats=self.wrap_ats)

    def to_univariate(self, dimension):
        """
        Shortcut to <MVHistogram.to_histogram>
        """
        return self.to_histogram(dimension=dimension)

    def to_histogram(self, dimension, wrap_at=None):
        """
        Convert the <MVHistogram> distribution to a <Histogram> univariate distribution.

        Arguments
        -----------
        * `dimension` (int or str): index or label of the dimension to use for
            the univariate distribution.
        * `wrap_at` (float or None, optional, default=None): value to set for
            `wrap_at` of the returned <Histogram>.  If None or not provided,
            will default to <<class>.wrap_at>.

        Returns
        --------
        * a <Histogram> object
        """
        dimension = self._get_dimension_index(dimension)
        bins_flat = self.bins[dimension]
        density_flat = _np.sum(self.density, axis=tuple([d for d in range(self.ndimensions) if d!=dimension]))
        return Histogram(bins=bins_flat, density=density_flat,
                         unit=self.units[dimension] if self.units is not None else None,
                         label=self.labels[dimension] if self.labels is not None else None,
                         wrap_at=self.wrap_ats[dimension] if self.wrap_ats is not None else None)

    def to_gaussian(self, dimension):
        """
        Convert the <MVHistogram> distribution to a <Gaussian> univariate distribution.

        Under-the-hood, this calls <MVHistogram.to_histogram> followed by <Histogram.to_gaussian>.

        Arguments
        -----------
        * `dimension` (int or str): index or label of the dimension to use for
            the univariate distribution.

        Returns
        ----------
        * a <Gaussian> object
        """
        return self.to_histogram(dimension).to_gaussian()

class MVHistogramSlice(BaseMultivariateSliceDistribution):
    @property
    def dist_constructor_func(self):
        return _stats_custom.generic_pdf_cdf_ppf

    @property
    def dist_constructor_argnames(self):
        raise NotImplementedError()

    @property
    def dist_constructor_args(self):
        return _hist_pdf_cdf_ppf_callables(self.bins, self.density)

    # def pdf(self, x, unit=None):
    #     # TODO: N-dimension interpolation of (self.bins, self.density)
    #     raise NotImplementedError("pdf not supported for {} distribution".format(self.__class__.__name__))
    #
    # def logpdf(self, x, unit=None):
    #     raise NotImplementedError("logpdf not supported for {} distribution".format(self.__class__.__name__))
    #
    # def cdf(self, x, unit=None):
    #     # TODO: N-dimensional interpolation of (self.bins, self._cdf_per_bin)
    #     raise NotImplementedError("cdf not supported for {} distribution".format(self.__class__.__name__))
    #
    # def logcdf(self, x, unit=None):
    #     raise NotImplementedError("logcdf not supported for {} distribution".format(self.__class__.__name__))

    @property
    def bins(self):
        return self.multivariate.bins[self.dimension]

    @property
    def density(self):
        return _np.sum(self.multivariate.density, axis=tuple([d for d in range(self.multivariate.ndimensions) if d!=self.dimension]))
