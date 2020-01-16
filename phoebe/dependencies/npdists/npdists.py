import numpy as _np
import json as _json
from collections import OrderedDict

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
    from scipy.stats import norm as _norm
except ImportError:
    _has_scipy = False
else:
    _has_scipy = True

try:
    from astropy import units as _units
except ImportError:
    _has_astropy = False
else:
    _has_astropy = True

try:
    import dill as _dill
except ImportError:
    _has_dill = False
else:
    _has_dill = True

_math_symbols = {'__mul__': '*', '__add__': '+', '__sub__': '-', '__div__': '/'}

_builtin_attrs = ['unit', 'label', 'wrap_at', 'dimension', 'sample_args']

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


def sample_from_dists(dists, *args, **kwargs):
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
    * `dists` (list or tuple of distribution objects): distribution objects from
        which to sample.
    * `*args`: all positional arguments are sent to <BaseDistribution.sample>
        for each item in `dists`.
    * `**kwargs`: all keyword arguments are sent to <BaseDistribution.sample>
        for each item in `dists`.  Note: `seed` is forbidden and will raise
        a ValueError.

    Returns
    -------------
    * (list): list of samples, in same order as `dists`.

    Raises
    ----------
    * ValueError: if `seed` is passed.
    """
    if 'seed' in kwargs.keys():
        raise ValueError("seeds are automatically determined: cannot pass seed")

    seeds = kwargs.pop('seeds', {})
    if seeds is None:
        seeds = {}

    if isinstance(dists, BaseDistribution):
        dists = [dists]
        flatten = True
    else:
        flatten = False

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
    for dist in dists:
        dists_all += unpack_dists(dist)

    for dist in dists_all:
        seeds.setdefault(dist.hash, get_random_seed())
    # print "*** seeds for hashes", seeds.keys()
    samples = [dist.sample(*args, seed=seeds, **kwargs) for dist in dists]
    if flatten:
        return samples[0]
    else:
        return _np.asarray(samples).T

def sample_ppf_from_dists(dists, ppf, *args, **kwargs):
    """

    Raises
    ----------
    * NotImplementedError
    """
    raise NotImplementedError

def logp_from_dists(dists, values):
    """
    """
    logp = 0.0
    dists_dict = {}
    values_dict = {}

    for dist,value in zip(dists, values):
        hash = dist.hash
        if hash in dists_dict.keys():
            dists_dict[hash] += [dist]
            values_dict[hash] += [value]
        else:
            dists_dict[hash] = [dist]
            values_dict[hash] = [value]

    for dists, values in zip(dists_dict.values(), values_dict.values()):
        for dist, value in zip(dists, values):
            logp += dist.logp(value)
        # logp += dists[0].logp(values, dimension=[dist.dimension for dist in dists]) #* len(dists)

    return logp

def sample_func_from_dists(dists, func, x, N=1000, func_kwargs={}):
    """
    Draw samples from a callable function.

    See also:

    * <npdists.plot_func_from_dists>

    Arguments
    -----------
    * `dists` (list or tuple of distribution objects): distribution objects from
        which to sample.
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
    sample_args = [sample_from_dists(dists) for i in range(N)]
    models = _np.array([func(x, *sample_args[i], **func_kwargs) for i in range(N)])
    return models

def plot_func_from_dists(dists, func, x, N=1000, func_kwargs={}, show=False):
    """
    Draw samples from a callable function and plot.

    The passed callable `func` will be called with arguments `x` followed by
    the individually drawn values from each distribution in `dists` (in order
    provided) and then any additional `func_kwargs`.

    See also:

    * <npdists.sample_func_from_dists>
    * <npdists.sample_from_dists>

    Arguments
    -----------
    * `dists` (list or tuple of distribution objects): distribution objects from
        which to sample.
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
        raise ImportError("plot_from_dists requires matplotlib.")

    models = sample_func_from_dists(dists, func, x, N=N, func_kwargs=func_kwargs)

    # TODO: allow options for sigma boundaries
    bounds = _np.percentile(models, 100 * _norm.cdf([-2, -1, 1, 2]), axis=0)

    ret1 = _plt.fill_between(x, bounds[0, :], bounds[-1, :],
                     label="95\% uncertainty", facecolor="#03A9F4", alpha=0.4)
    ret2 = _plt.fill_between(x, bounds[1, :], bounds[-2, :],
                     label="68\% uncertainty", facecolor="#0288D1", alpha=0.4)

    if show:
        _plt.show()

    return ret1, ret2


################## VALIDATORS ###################

# these all must accept a single value and return a boolean if it matches the condition as well as any alterations to the value
# NOTE: the docstring is used as the error message if the test fails

def is_distribution(value):
    """must be an npdists Distribution object"""
    if isinstance(value, dict) and 'npdists' in value.keys():
        # TODO: from_dict probably not available since its in __init__.py
        return True, from_dict(value)
    return isinstance(value, BaseDistribution), value

def is_distribution_or_none(value):
    """must be an npdists Distribution object or None"""
    if value is None:
        return True, value
    else:
        return is_distribution(value)

def is_math(value):
    """must be a valid math operator"""
    # TODO: make this more robust checking
    valid_maths = ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__div__', '__rdiv__']
    valid_maths += ['sin', 'cos', 'tan']
    return value in valid_maths, value

def is_callable(value):
    """must be a callable function"""
    if isinstance(value, str):
        # try to "undill"
        if _has_dill:
            value = _dill.loads(value)
        else:
            raise ImportError("'dill' package required to load functions")
    return hasattr(value, 'func_name'), value

def is_callable_or_none(value):
    if value is None:
        return True, value
    else:
        return is_callable(value)

def is_unit(value):
    """must be an astropy unit"""
    if not _has_astropy:
        raise ImportError("astropy must be installed for unit support")
    if (isinstance(value, _units.Unit) or isinstance(value, _units.IrreducibleUnit) or isinstance(value, _units.CompositeUnit)):
        return True, value
    else:
        return False, value

def is_unit_or_unitstring(value):
    """must be an astropy.unit"""
    if is_unit(value)[0]:
        return True, value
    try:
        unit = _units.Unit(value)
    except:
        return False, value
    else:
        return True, unit

def is_unit_or_unitstring_or_none(value):
    """must be an astropy unit or None"""
    if value is None:
        return True, value
    return is_unit_or_unitstring(value)

def is_bool(value):
    """must be boolean"""
    return isinstance(value, bool), value

def is_float(value):
    """must be a float"""
    return isinstance(value, float) or isinstance(value, int) or isinstance(value, _np.float64), float(value)

def is_int(value):
    """must be an integer"""
    return isinstance(value, int), value

def is_int_positive(value):
    """must be a positive integer"""
    return isinstance(value, int) and value > 0, value

def is_int_positive_or_none(value):
    """must be a postive integer or None"""
    return is_int_positive or value is None, value

def is_valid_shape(value):
    """must be a positive integer or a tuple/list of positive integers"""
    if is_int_positive(value):
        return True, value
    elif isinstance(value, tuple) or isinstance(value, list):
        for v in value:
            if not is_int_positive(v):
                return False, value
        return True, value
    else:
        return False, value

def is_iterable(value):
    """must be an iterable (list, array, tuple)"""
    return isinstance(value, _np.ndarray) or isinstance(value, list) or isinstance(value, tuple), value

def is_square_matrix(value):
    """must be a square 2D matrix"""
    return isinstance(value, _np.ndarray) and len(value.shape)==2 and value.shape[0]==value.shape[1], value



######################## DISTRIBUTION FUNCTIONS ###############################

def delta(x, value):
    return _np.asarray(x==value, dtype='int')

def gaussian(x, loc, scale):
    return 1./_np.sqrt(2*_np.pi*scale**2) * _np.exp(-(x-loc)**2/(2.*scale**2))

def uniform(x, low, high):
    return _np.asarray((x >= low) * (x <= high), dtype='int') / float(high - low)

def histogram(x, bins, density):
    out = _np.zeros_like(x)
    filter_in_range = (x >= bins.min()) & (x < bins.max())
    out[filter_in_range] = density[_np.digitize(x[filter_in_range], bins)-1]
    return out

def mvgaussian(x, locs, cov, dimension=None):
    if dimension is None:
        raise NotImplementedError
    else:
        return gaussian(x, locs[dimension], cov[dimension, dimension])


############################### SAMPLE FUNCTIONS ###############################

def _sample_from_hist(bins, density, size=None):


    ppf = _np.random.rand(size if size is not None else 1)
    return _sample_ppf_from_hist(ppf if size is not None else ppf[0], bins, density)

def _sample_ppf_from_hist(ppf, bins, density):
    # adapted from: https://stackoverflow.com/a/17822210

    # MV case
    # density, bins = np.histogramdd(chain_flat, normed=True)
    # bins = np.asarray(bins)


    # 1D case
    #density, bins = np.histogram(np.random.rand(1000), normed=True)
    #bins = np.asarray([bins])
    if np.any(pff > 1) or np.any(ppf < 0):
        raise ValueError("ppf must be between 0 and 1")

    if isinstance(ppf, float):
        return_single = True
        ppf = _np.asarray([ppf])
    else:
        return_single = False
        ppf = _np.asarray(ppf)


    cdf = _np.cumsum(density)
    cdf = cdf / float(cdf[-1])

    value_bins = _np.searchsorted(cdf, ppf)

    if len(bins.shape) > 1:
        inds = _np.unravel_index(value_bins, density.shape)

        bin_widths = _np.column_stack([_np.diff(b) for b in bins])
        values_from_bins = _np.column_stack([b[ind]+bin_widths[ind,dim] * (ppf % 1) for dim,(b,ind) in enumerate(zip(bins, inds))])
    else:
        bin_widths = _np.diff(bins)
        values_from_bins = bins[value_bins] + bin_widths[value_bins] * (ppf % 1)

    if return_single:
        return values_from_bins[0]
    else:
        return values_from_bins

def _raise_import_scipy(*args, **kwargs):
    raise ImportError("sampling requires scipy to be installed")

######################## DISTRIBUTION ABSTRACT CLASS ###########################

class BaseDistribution(object):
    """
    BaseDistribution is the parent class for all distributions and should
    not be used directly by the user.

    Any subclass distribution should override the following:

    * <BaseDistribution.__init__>
    * <BaseDistribution.mean>
    * <BaseDistribution.std>
    """
    def __init__(self, unit, label, wrap_at,
                 dist_func, dist_args,
                 sample_func, sample_ppf_func, sample_args,
                 *args):
        """
        BaseDistribution is the parent class for all distributions and should
        not be used directly by the user.

        Any subclass distribution should override the init but call this via
        super.

        For example:

        ```py
        def __init__(self, start, stop, step):
            super(MyClass, self).__init__(('start', start, is_float), ('stop', stop, is_float), ('step', step, is_float))
        ```

        All of these "descriptors" will then be available to get and set via
        their attribute names
        """
        self._descriptors = OrderedDict()
        self._validators = OrderedDict()

        self._dist_func = dist_func
        self._dist_args = dist_args

        self._sample_func = sample_func
        self._sample_ppf_func = sample_ppf_func
        self._sample_args = sample_args

        self.label = label
        self.unit = unit
        self.wrap_at = wrap_at

        for item in args:
            if item[0] in _builtin_attrs:
                raise KeyError("{} is a protected attribute.".format(item[0]))

            valid, validated_value = item[2](item[1])
            if valid:
                self._descriptors[item[0]] = validated_value
            else:
                raise ValueError("{} {}, got {}".format(item[0], item[2].__doc__, item[1]))
            self._validators[item[0]] = item[2]

    def __getattr__(self, name):
        """
        for anything that isn't overriden here, call the method on the array itself
        """
        if name in _builtin_attrs or (name.startswith("_") and not name.startswith('__') and not name.endswith('_')):
            # then we need to actually get the attribute
            return super(BaseDistribution, self).__getattr__(name)
        elif name in self._descriptors.keys():
            # then get the item in the dictionary
            return self._descriptors.get(name)
        else:
            raise AttributeError("{} does not have attribute '{}'".format(self.__class__.__name__.lower(), name))

    def __setattr__(self, name, value):
        """
        """
        if name in _builtin_attrs or (name.startswith("_") and not name.startswith('__') and not name.endswith('_')):
            return super(BaseDistribution, self).__setattr__(name, value)
        elif name in self._descriptors.keys():
            valid, validated_value = self._validators[name](value)
            if valid:
                self._descriptors[name] = validated_value
            else:
                raise ValueError("{} {}".format(name, validator.__doc__))
        else:
            raise AttributeError("{} does not have attribute '{}'".format(self.__class__.__name__.lower(), name))

    def __repr__(self):
        descriptors = " ".join(["{}={}".format(k,v) for k,v in self._descriptors.items()])
        if self.unit is not None:
            descriptors += " unit={}".format(self.unit)
        if self.wrap_at is not None:
            descriptors += " wrap_at={}".format(self.wrap_at)
        if hasattr(self, 'dimension'):
            descriptors += " dimension={}".format(self.dimension)
        return "<npdists.{} {}>".format(self.__class__.__name__.lower(), descriptors)

    def __str__(self):
        if self.label is not None:
            return "{"+self.label+"}"
        else:
            return self.__repr__()

    def __float__(self):
        """
        by default, have the float representation come from sampling, but
        subclasses can/should override this to be the central/median/mode if
        possible
        """
        return self.sample()

    def __lt__(self, other):
        if isinstance(other, BaseDistribution):
            return self.__float__() < other.__float__()
        return self.__float__() < other

    def __le__(self, other):
        if isinstance(other, BaseDistribution):
            return self.__float__() <= other.__float__()
        return self.__float__() <= other

    def __gt__(self, other):
        if isinstance(other, BaseDistribution):
            return self.__float__() > other.__float__()
        return self.__float__() > other

    def __ge__(self, other):
        if isinstance(other, BaseDistribution):
            return self.__float__() >= other.__float__()
        return self.__float__() >= other

    def __copy__(self):
        return self.__class__(unit=self.unit, label=self.label, **self._descriptors)

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

    def __mul__(self, other):
        if _has_astropy and is_unit(other)[0]:
            copy = self.copy()
            copy.unit = other
            return copy

        elif isinstance(other, BaseDistribution):
            return Composite("__mul__", self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__mul__(Delta(other))
        else:
            raise TypeError("cannot multiply {} by type {}".format(self.__class__.__name__, type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, BaseDistribution):
            return Composite("__div__", self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__div__(Delta(other))
        else:
            raise TypeError("cannot divide {} by type {}".format(self.__class__.__name__, type(other)))

    def __rdiv__(self, other):
        if isinstance(other, BaseDistribution):
            return Composite("__rdiv__", self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__rdiv__(Delta(other))
        else:
            raise TypeError("cannot divide {} by type {}".format(self.__class__.__name__, type(other)))

    def __add__(self, other):
        if isinstance(other, BaseDistribution):
            return Composite("__add__", self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__add__(Delta(other))
        else:
            raise TypeError("cannot add {} by type {}".format(self.__class__.__name__, type(other)))

    # def __radd__(self, other):
    #     return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, BaseDistribution):
            return Composite("__sub__", self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__sub__(Delta(other))
        else:
            raise TypeError("cannot subtract {} by type {}".format(self.__class__.__name__), type(other))

    def sin(self):
        if self.unit is not None:
            dist = self.to(_units.rad)
        else:
            dist = self

        return Composite("sin", dist, label="sin({})".format(self.label) if self.label is not None else None)

    def cos(self):
        if self.unit is not None:
            dist = self.to(_units.rad)
        else:
            dist = self

        return Composite("cos", dist, label="cos({})".format(self.label) if self.label is not None else None)

    def tan(self):
        if self.unit is not None:
            dist = self.to(_units.rad)
        else:
            dist = self

        return Composite("tan", dist, label="tan({})".format(self.label) if self.label is not None else None)


    @property
    def label(self):
        """
        The label of the distribution object.  When not None, this is used for
        the x-label when plotting (see <<class>.plot>) and for the
        string representation for any math in a <Composite> or <Function>
        distribution.
        """
        return self._label

    @label.setter
    def label(self, label):
        if not (label is None or isinstance(label, str)):
            raise TypeError("label must be of type str")

        self._label = label

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
        if not (unit is None or isinstance(unit, _units.Unit) or isinstance(unit, _units.CompositeUnit) or isinstance(unit, _units.IrreducibleUnit)):
            raise TypeError("unit must be of type astropy.units.Unit")

        self._unit = unit

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

        if self.unit is None:
            raise ValueError("distribution object does not have a unit")

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

    @property
    def mean(self):
        """
        mean is not implemented for this distribution type.

        Raises
        --------
        * NotImplementedError
        """
        raise NotImplementedError

    @property
    def std(self):
        """
        std is not implemented for this distribution type.

        Raises
        --------
        * NotImplementedError
        """
        raise NotImplementedError

    @property
    def dist_func(self):
        """
        Return the callable function to access the distribution function, if
        available.

        See also:

        * <<class>.dist_args>
        * <<class>.distribution>

        Returns
        -------
        * callable function
        """
        return self._dist_func

    @property
    def dist_args(self):
        """
        Return the arguments sent to the distribution function.

        See also:

        * <<class>.dist_func>
        * <<class>.distribution>

        Returns
        --------
        * tuple
        """

        return tuple(getattr(self, k) for k in self._dist_args)

    @property
    def sample_func(self):
        """
        Return the callable function to sample the distribution, if available.

        See also:

        * <<class>.sample_args>
        * <<class>.sample>

        Returns
        --------
        * callable function
        """
        return self._sample_func

    @property
    def sample_ppf_func(self):
        """
        Return the callable function to sample the distribution from a ppf, if available.

        See also:

        * <<class>.sample_args>
        * <<class>.sample_ppf>

        Returns
        ---------
        * callable function
        """
        return self._sample_ppf_func

    @property
    def sample_args(self):
        """
        Return the arguments sent to the sample function.

        See also:

        * <<class>.sample_func>
        * <<class>.sample>

        Returns
        --------
        * tuple
        """
        return tuple(getattr(self, k) for k in self._sample_args)

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

    def sample(self, size=None, unit=None, as_quantity=False, wrap_at=None, seed=None):
        """
        Sample from the distribution.

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

        Returns
        ---------
        * float or array: float if `size=None`, otherwise a numpy array with
            shape defined by `size`.
        """
        if isinstance(seed, dict):
            seed = seed.get(self.hash, None)

        if seed is not None:
            _np.random.seed(seed)

        return self._return_with_units(self.wrap(self.sample_func(*self.sample_args, size=size), wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def sample_ppf(self, ppf, unit=None, as_quantity=False, wrap_at=None):
        """
        Sample the distribution by mapping a given percent point function or ppf
        (a value between [0, 1)).

        Arguments
        ------------
        * `ppf` (float or array of floats): ppf value(s) with values in the range
            [0, 1).
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

        Returns
        -----------
        * float or array: float or array depending on the input of `ppf`.
        """

        return self._return_with_units(self.wrap(self.sample_ppf_func(ppf, *self.sample_args), wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)

    def distribution(self, x, unit=None):
        """
        Give the density (y) values of the underlying distribution for a given
        array of values (x).

        Arguments
        ----------
        * `x` (array): x-values at which to compute the densities.  If `unit` is
            not None, the value of `x` are assumed to be in the original units
            <<class>.unit>, not `unit`.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * array: array of density/y values.
        """
        # x is assumed to be in the new units
        if unit is not None:
            if self.unit is None:
                raise ValueError("can only convert units on Distributions with units set")
            # convert to original units
            x = (x * unit).to(self.unit).value

        # print "*** x passed to dist_func", x.min(), x.max()
        return self.dist_func(x, *self.dist_args)

    def logp(self, x, unit=None):
        """
        Give the log probability of the underlying distribution for a given value
        x.

        Arguments
        ----------
        * `x` (float or array array): x-values at which to compute the logps.
            If `unit` is not None, the value of `x` are assumed to be in the
            original units
            <<class>.unit>, not `unit`.
        * `unit` (astropy.unit, optional, default=None): unit of the values
            in `x`.  If None or not provided, will assume they're provided in
            <<class>.unit>.

        Returns
        ---------
        * array: array of density/y values.
        """
        densities = self.distribution(x=x, unit=unit)
        return _np.log(densities)

    # def plot_func(self, show=False, **kwargs):
    #     ret = _plt.hist(self.sample(size), **kwargs)
    #     if show:
    #         _plt.show()
    #
    #     return ret

    def _xlabel(self, unit=None, label=None):
        label = label if label is not None else self.label
        l = 'value' if label is None else label
        if _has_astropy and self.unit is not None and self.unit not in [_units.dimensionless_unscaled]:
            l += ' ({})'.format(unit if unit is not None else self.unit)

        return l


    def plot(self, size=100000, unit=None,
             wrap_at=None, seed=None,
             plot_sample=True, plot_sample_kwargs={'color': 'gray'},
             plot_dist=True, plot_dist_kwargs={'color': 'red'},
             plot_gaussian=False, plot_gaussian_kwargs={'color': 'blue'},
             label=None, show=False, **kwargs):
        """
        Plot both the analytic distribution function as well as a sampled
        histogram from the distribution.  Requires matplotlib to be installed.

        See also:

        * <<class>.plot_sample>
        * <<class>.plot_dist>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `size` (int, optional, default=100000): number of points to sample for
            the histogram.  See also <<class>.sample>.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `seed` (int, optional): seed to use when sampling.  See also
            <<class>.sample>.
        * `plot_sample` (bool, optional, default=True): whether to plot the
            histogram from sampling.  See also <<class>.plot_sample>.
        * `plot_sample_kwargs` (dict, optional, default={'color': 'gray'}):
            keyword arguments to send to <<class>.plot_sample>.
        * `plot_dist` (bool, optional, default=True): whether to plot the
            analytic form of the underlying distribution, if applicable.
            See also <<class>.plot_dist>.
        * `plot_dist_kwargs` (dict, optional, default={'color': 'red'}):
            keyword arguments to send to <<class>.plot_dist>.
        * `plot_gaussian` (bool, optional, default=False): whether to plot
            a guassian distribution fit to the sample.  Only supported for
            distributions that have <<class>.to_gaussian> methods.
        * `plot_gaussian_kwargs` (dict, optional, default={'color': 'blue'}):
            keyword arguments to send to <<class>.plot_gaussian>.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: all keyword arguments (except for `bins`) will be passed
            on to <<class>.plot_dist> and <<class>.plot_gaussian> and all
            keyword arguments will be passed on to <<class>.plot_sample>.
            Keyword arguments defined in `plot_sample_kwargs`,
            `plot_dist_kwargs`, and `plot_gaussian_kwargs`
            will override the values sent in `kwargs`.

        Returns
        --------
        * tuple: the return values from <<class>.plot_sample> (or None if
            `plot_sample=False`), <<class>.plot_dist> (or None if `plot_dist=False`),
            and <Gaussian.plot_dist> (or None if `plot_gaussian=False`).

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
            ret_sample = self.plot_sample(size=size, unit=unit, wrap_at=wrap_at, seed=seed, show=False, **plot_sample_kwargs)
        else:
            ret_sample = None

        if plot_gaussian or plot_dist:
            # we need to know the original x-range, before wrapping
            sample = self.sample(size=size, unit=unit, wrap_at=False)
            xmin = _np.min(sample)
            xmax = _np.max(sample)

            x = _np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 1001)

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

        if plot_dist:
            # we have to make a copy here, otherwise setdefault will change the
            # defaults in the function declaration for successive calls
            plot_dist_kwargs = plot_dist_kwargs.copy()
            for k,v in kwargs.items():
                if k in ['bins']:
                    continue
                plot_dist_kwargs.setdefault(k,v)
            ret_dist = self.plot_dist(x, unit=unit, wrap_at=wrap_at, show=False, **plot_dist_kwargs)
        else:
            ret_dist = None

        if show:
            _plt.xlabel(self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()

        return (ret_sample, ret_dist, ret_gauss)


    def plot_sample(self, size=100000, unit=None,
                    wrap_at=None, seed=None,
                    label=None, show=False, **kwargs):
        """
        Plot both a sampled histogram from the distribution.  Requires
        matplotlib to be installed.

        See also:

        * <<class>.plot>
        * <<class>.plot_dist>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `size` (int, optional, default=100000): number of points to sample for
            the histogram.  See also <<class>.sample>.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `seed` (int, optional): seed to use when sampling.  See also
            <<class>.sample>.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.
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
            if wrap_at or self.wrap_at:
                kwargs.setdefault('bins', len(self.bins))
            else:
                kwargs.setdefault('bins', self.bins)
        else:
            kwargs.setdefault('bins', 25)

        # TODO: wrapping can sometimes cause annoying things with bins due to a large datagap.
        # Perhaps we should bin and then wrap?  Or bin before wrapping and get a guess at the
        # appropriate bins
        try:
            ret = _plt.hist(self.sample(size, unit=unit, wrap_at=wrap_at, seed=seed), density=True, **kwargs)
        except AttributeError:
            # TODO: determine which version of matplotlib
            # TODO: this still doesn't handle the same
            ret = _plt.hist(self.sample(size, unit=unit, wrap_at=wrap_at, seed=seed), normed=True, **kwargs)

        if show:
            _plt.xlabel(self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()

        return ret

    def plot_dist(self, x, unit=None, wrap_at=None,
                  label=None, show=False, **kwargs):
        """
        Plot the analytic distribution function.  Requires matplotlib to be installed.

        See also:

        * <<class>.plot>
        * <<class>.plot_sample>
        * <<class>.plot_gaussian>

        Arguments
        -----------
        * `x` (np array): the numpy array at which to sample the value on the
            x-axis.  If `unit` is not None, the value of `x` are assumed to be
            in the original units <<class>.unit>, not `unit`.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.
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

        # x is assumed to be in new units
        if self.dist_func is not None:
            y = self.distribution(x, unit=unit)
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
            ret = None

        if show:
            _plt.xlabel(self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()

        return ret

    def plot_gaussian(self, x, unit=None, wrap_at=None,
                      label=None, show=False, **kwargs):
        """
        Plot the gaussian distribution that would result from calling
        <<class>.to_gaussian> with the same arguments.

        Note that for distributions in which <<class>.to_gaussian> calls
        <<class>.to_histogram> under-the-hood, this could result in slightly
        different distributions for each call.

        See also:

        * <<class>.plot>
        * <<class>.plot_sample>
        * <<class>.plot_dist>

        Arguments
        -----------
        * `x` (np array): the numpy array at which to sample the value on the
            x-axis. If `unit` is not None, the value of `x` are assumed to be
            in the original units <<class>.unit>, not `unit`.
        * `unit` (astropy.unit, optional, default=None): units to use along
            the x-axis.  Astropy must be installed.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  See <<class>.wrap>.  If not provided or None,
            will use the value from <<class>.wrap_at>.  Note: wrapping is
            computed before changing units, so `wrap_at` must be provided
            according to <<class>.unit> not `unit`.
        * `label` (string, optional, default=None): override the label on the
            x-axis.  If not provided or None, will use <<class>.label>.  Will
            only be used if `show=True`.
        * `show` (bool, optional, default=True): whether to show the resulting
            matplotlib figure.
        * `**kwargs`: keyword arguments for `sigma`, `N`, `bins`, `range` will
            be passed on to <<class>.to_gaussian> (must be accepted by the
            given distribution type).  All other keyword arguments will be passed
            on to <Gaussian.plot_dist> on the resulting distribution.

        Returns
        --------
        * the return from plt.plot

        Raises
        --------
        * ImportError: if matplotlib dependency is not met.
        """
        if not _has_mpl:
            raise ImportError("matplotlib required for plotting")

        to_gauss_keys = ['sigma', 'N', 'bins', 'range']
        g = self.to_gaussian(**{k:v for k,v in kwargs.items() if k in to_gauss_keys})

        if unit is not None:
            g = g.to(unit)
            if wrap_at is not None and wrap_at is not False:
                wrap_at = (wrap_at * self.unit).to(unit).value

        # TODO: this time wrap_at is assumed to be in the plotted units, not the original... do we need to convert?
        ret = g.plot_dist(x, wrap_at=wrap_at, **{k:v for k,v in kwargs.items() if k not in to_gauss_keys})

        if show:
            _plt.xlabel(self._xlabel(unit, label=label))
            _plt.ylabel('density')
            _plt.show()
        return ret

    def to_histogram(self, N=1000, bins=10, range=None):
        """
        Convert the <<class>> distribution to a <Histogram> distribution.

        Under-the-hood, this calls <<class>.sample> with `size=N` and `wrap_at=False`
        and passes the resulting array as well as the requested `bins` and `range`
        to <Histogram.from_data>.

        Arguments
        -----------
        * `N` (int, optional, default=1000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.

        Returns
        --------
        * a <Histogram> object
        """
        return Histogram.from_data(self.sample(size=N, wrap_at=False),
                                   bins=bins, range=range,
                                   unit=self.unit, label=self.label, wrap_at=self.wrap_at)

    @property
    def hash(self):
        """
        """
        return hash(frozenset({k:v for k,v in self.to_dict().items() if k not in ['dimension']}))

    def to_dict(self):
        """
        Return the dictionary representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <npdists.from_dict>.

        See also:

        * <<class>.to_json>
        * <<class>.to_file>

        Returns
        --------
        * dictionary
        """
        def _json_safe(v):
            if isinstance(v, _np.ndarray):
                return v.tolist()
            elif isinstance(v, list) or isinstance(v, tuple):
                return [_json_safe(li) for li in v]
            elif isinstance(v, BaseDistribution):
                return v.to_dict()
            elif hasattr(v, 'func_name'):
                if _has_dill:
                    return _dill.dumps(v)
                else:
                    raise ImportError("'dill' package required to export functions to dictionary")
            # elif is_unit(v)[0]:
            #     return v.to_string()
            else:
                return v
        d = {k:_json_safe(v) for k,v in self._descriptors.items()}
        d['npdists'] = self.__class__.__name__.lower()
        if self.unit is not None:
            d['unit'] = self.unit.to_string()
        if self.label is not None:
            d['label'] = self.label
        if self.wrap_at is not None:
            d['wrap_at'] = self.wrap_at
        if hasattr(self, 'dimension') and self.dimension is not None:
            d['dimension'] = self.dimension
        return d

    def to_json(self, **kwargs):
        """
        Return the json representation of the distribution object.

        The resulting dictionary can be restored to the original object
        via <npdists.from_json>.

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
        try:
            return _json.dumps(self.to_dict(), ensure_ascii=True, **kwargs)
        except:
            if _has_dill:
                return _dill.dumps(self)
            else:
                raise ImportError("dumping file requires 'dill' package")

    def to_file(self, filename, **kwargs):
        """
        Save the json representation of the distribution object to a file.

        The resulting file can be restored to the original object
        via <npdists.from_file>.

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


####################### DISTRIBUTION SUB-CLASSES ###############################

class Composite(BaseDistribution):
    """
    A composite distribution consisting of some math operator between one or two
    other Distribution objects.

    For example:

    ```py
    g = npdists.gaussian(10, 2)
    u = npdists.gaussian(1, 5)

    c = g * u
    print(c)
    ```

    or:

    ```py
    import numpy as np
    g = npdists.gaussian(0, 1)
    sing = np.sin(g)
    print(sing)
    ```

    Currently supported operators include:

    * multiplication, division, addition, subtraction
    * np.sin, np.cos, np.tan (but not math.sin, etc)

    When doing math between a distribution and a float or integer, that float/int
    will be treated as a <Delta> distribution.  In some simple cases, the
    applicable distribution type will be returned, but in other cases,
    a <Composite> distribution will be returned.  For example, multiplying
    a <Uniform> or <Gaussian> distribution by a float will return another
    <Uniform> or <Gaussian> distribution, respectively.

    """
    def __init__(self, math, dist1, dist2=None, unit=None, label=None, wrap_at=None):
        """
        Create a <Composite> distribution from two other distributions.

        Most likely, users will create Composite objects through math operators
        directly.  See examples on the <Composite> overview page.

        Arguments
        ----------
        * `math`: operator to be used between the two distributions.  Must
            be a valid and implemented operator.
        * `dist1` (<BaseDistribution>)
        * `dist2` (<BaseDistribution>, optional, default=None): the second
            distribution is required for most operators.  Some operators
            (e.g. sin, cos, tan) only take one distribution as an argument.
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
                                        None, None,
                                        self._sample_from_children, self._sample_ppf_from_children, ('math', 'dist1', 'dist2'),
                                        ('math', math, is_math), ('dist1', dist1, is_distribution), ('dist2', dist2, is_distribution_or_none))

        if _has_astropy:
            if dist1.unit is not None:
                if dist2 is None:
                    # trig always gives unitless
                    self.unit = _units.dimensionless_unscaled
                elif dist2.unit is not None:
                    if math in ['__add__', '__sub__']:
                        if dist1.unit == dist2.unit:
                            self.unit = dist1.unit
                        else:
                            # TODO: if they're convertible, we should handle the scaling automatically
                            raise ValueError("units do not match")
                    elif hasattr(dist1.unit, math):
                        self.unit = getattr(dist1.unit, math)(dist2.unit)
                    else:
                        raise ValueError("cannot determine new unit from {}{}{}".format(dist1.unit, _math_symbols.get(math, math), dist2.unit))
                else:
                    self.unit = dist1.unit
            elif dist2 is not None and dist2.unit is not None:
                self.unit = dist2.unit
            else:
                self.unit = None


    def __repr__(self):
        return "<npdists.{} {} unit={}>".format(self.__class__.__name__.lower(), self.__str__(), self.unit if self.unit is not None else "None")

    def __str__(self):
        if self.dist2 is not None:
            return "{}{}{}".format(self.dist1.__str__(), _math_symbols.get(self.math, self.math), self.dist2.__str__())
        else:
            return "{}({})".format(_math_symbols.get(self.math, self.math), self.dist1.__str__())

    @property
    def dists(self):
        if self.dist2 is not None:
            return [self.dist1, self.dist2]
        else:
            return [self.dist1]

    @property
    def hash(self):
        """
        """
        if self.dist2 is not None:
            if self.dist1.hash == self.dist2.hash:
                return self.dist1.hash
            else:
                # NOTE (IMPORTANT): then we are going to "forget" these when
                # nesting ComposisteDistributions
                # return super(CompositeDistribution, self).hash()
                return [self.dist1.hash, self.dist2.hash]
        else:
            return self.dist1.hash

    def _sample_from_children(self, math, dist1, dist2, seed={}, size=None):
        if self.dist2 is not None:
            # NOTE: this will account for multivariate, but only for THESE 2
            # if there are nested CompositeDistributions, then the seed will be lost

            # TODO: need to pass seeds on somehow
            samples = sample_from_dists((dist1, dist2), seeds=seed, size=size)
            if size is not None:
                return getattr(samples[:,0], math)(samples[:,1])
            else:
                return getattr(samples[0], math)(samples[1])
        else:
            # if math in ['sin', 'cos', 'tan'] and _has_astropy and dist1.unit is not None:
            #     unit = _units.rad
            # else:
            #     unit = None
            return getattr(_np, math)(dist1.sample(size=size, seed=seed, as_quantity=_has_astropy and self.unit not in [None, _units.dimensionless_unscaled]))

    def _sample_ppf_from_children(self, math, dist1, dist2, ppf):
        raise NotImplementedError("sample_ppf_from_dists not yet implemented")
        if self.dist2 is not None:
            samples = sample_ppf_from_dists((dist1, dist2), ppf)
            if not isinstance(ppf, float):
                return getattr(samples[:,0], math)(samples[:,1])
            else:
                return getattr(samples[0], math)(samples[1])
        else:
            return getattr(_np, math)(dist1.sample_ppf(ppf, as_quantity=_has_astropy and self.unit not in [None, _units.dimensionless_unscaled]))



    def sample(self, size=None, unit=None, as_quantity=False, wrap_at=None, seed={}):
        """
        Sample from the distribution.

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

        Returns
        ---------
        * float or array: float if `size=None`, otherwise a numpy array with
            shape defined by `size`.
        """

        return self._return_with_units(self.wrap(self.sample_func(*self.sample_args, size=size, seed=seed), wrap_at=wrap_at), unit=unit, as_quantity=as_quantity)



    def __float__(self):
        return self.mean
        # return self.sample()

        # if self.dist2 is not None:
            # return getattr(dist1.mean, math)(dist2.mean)
        # else:
            # return getattr(_np, math)(dist1.mean)



    @property
    def mean(self):
        """
        Determine the mean sampled value.

        This is done under-the-hood by converting to a histogram via
        <Composite.to_histogram>, sampling 10000 times with 100 bins and
        calling <Histogram.mean>.
        """
        return self.to_histogram(N=10000, bins=100).mean

    @property
    def std(self):
        """
        Determine the standard deviations of the sampled values.

        This is done under-the-hood by converting to a histogram via
        <Composite.to_histogram>, sampling 10000 times with 100 bins and
        calling <Histogram.std>.
        """
        return self.to_histogram(N=10000, bins=100).std

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

class Function(BaseDistribution):
    """
    A Function distribution allows for any python callable to be stored that
    utilizes distributions under-the-hood.  When calling <Function.sample>,
    any argument passed to the function that is a <BaseDistribution> object
    will be sampled prior to being passed to the callable function.

    In order to save or load these distributions, it is necessary to have
    the `dill` package installed.  Note that you should not load from untrusted
    sources, as any executable could be contained in the callable function.

    See:

    * <Function.to_dict>
    * <Function.to_json>
    * <Function.to_file>

    for documentation on loading and saving Function distributions.
    """
    def __init__(self, func, func_ppf, unit, label, wrap_at, *args):
        """
        Create a <Function> distribution from some callable function and
        any number of arguments, including distribution objects.

        This can also be created from a function at the top-level as:

        * <npdists.function>

        Arguments
        ----------
        * `func` (callable function, or None): the callable function to be called to
            sample the distribution.  If None, <<class>.sample> will raise a
            TypeError.
        * `func_ppf` (callable function, or None): the callable function to be
            called when sampling from a ppf.  If None, <<class>.sample_ppf> will
            raise a TypeError.
        * `unit` (astropy.units object or None): the units of the provided values.
        * `label` (string or None): a label for the distribution.  This is used
            for the x-label while plotting the distribution, as well as a shorthand
            notation when creating a <Composite> distribution.
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.
        * `*args`: all additional positional arguments will be passed on to
            `func` when sampling.  These can be, but are not limited to,
            other distribution objects.

        Returns
        ---------
        * a <Function> object.
        """
        super(Function, self).__init__(unit, label, wrap_at,
                                       None, None,
                                       self._sample_from_function, self._sample_ppf_from_function, ('func', 'args'),
                                       ('func', func, is_callable_or_none), ('func_ppf', func_ppf, is_callable_or_none), ('args', args, is_iterable))

    def _sample_from_function(self, func, args, size=None):
        if func is None:
            raise TypeError("cannot sample from function when set to None")
        args = (a.sample(size=size) if isinstance(a, BaseDistribution) else a for a in args)
        return func(*args)

    def _sample_pff_from_function(self, func_ppf, args, ppf):
        if func_ppf is None:
            raise TypeError("cannot sample from ppf function when set to None")
        # TODO: not sure this is still what we want here...
        args = (a.sample(size=size) if isinstance(a, BaseDistribution) else a for a in args)
        return func_ppf(ppf, *args)

    def __float__(self):
        return self.mean
        # return self.sample()

    @property
    def mean(self):
        """
        Determine the mean sampled value.

        This is done under-the-hood by converting to a histogram via
        <Function.to_histogram>, sampling 10000 times with 100 bins and
        calling <Histogram.mean>.
        """
        return self.to_histogram(N=10000, bins=100).mean

    @property
    def std(self):
        """
        Determine the standard deviations of the sampled values.

        This is done under-the-hood by converting to a histogram via
        <Function.to_histogram>, sampling 10000 times with 100 bins and
        calling <Histogram.std>.
        """
        return self.to_histogram(N=10000, bins=100).std

    def to_gaussian(self, N=1000, bins=10, range=None):
        """
        Convert the <Function> distribution to a <Gaussian> distribution via
        a <Histogram> distribution.

        Under-the-hood, this calls <Function.to_histogram> with the requested
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
        Convert the <Function> distribution to a <Uniform> distribution via
        a <Histogram> distribution.

        Under-the-hood, this calls <Function.to_histogram> with the requested
        values of `N`, `bins`, and `range` and then calls <Histogram.to_uniform>
        with the requested value of `sigma` (which in turn is calling
        <Histogram.to_gaussian> first).

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


class Histogram(BaseDistribution):
    """
    A Histogram distribution stores a discrete PDF and allows sampling from
    that binned density distribution.

    To create a Histogram distribution from already binned data, see
    <npdists.histogram_from_bins> or <Histogram.__init__>.  To create a
    Histogram distribtuion from the data array itself, see
    <npdists.histogram_from_data> or <Histogram.from_data>.
    """
    def __init__(self, bins, density, unit=None, label=None, wrap_at=None):
        """
        Create a <Histogram> distribution from bins and density.

        This can also be created from a function at the top-level as:

        * <npdists.histogram_from_bins>

        See also:

        * <Histogram.from_data>
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
        * `wrap_at` (float, None, or False, optional, default=None): value to
            use for wrapping.  If None and `unit` are angles, will default to
            2*pi (or 360 degrees).  If None and `unit` are cycles, will default
            to 1.0.

        Returns
        --------
        * a <Histogram> object
        """
        super(Histogram, self).__init__(unit, label, wrap_at,
                                        histogram, ('bins', 'density'),
                                        _sample_from_hist, _sample_ppf_from_hist, ('bins', 'density'),
                                        ('bins', bins, is_iterable), ('density', density, is_iterable))

    @classmethod
    def from_data(cls, data, bins=10, range=None, weights=None,
                  label=None, unit=None, wrap_at=None):
        """
        Create a <Histogram> distribution from data.

        This can also be created from a function at the top-level as:

        * <npdists.histogram_from_data>

        See also:

        * <Histogram.__init__>
        * <npdists.histogram_from_bins>

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

    def __float__(self):
        return self.mean
        # return self.sample()

    @property
    def mean(self):
        """
        Return the weighted mean value from the histogram.

        See also:

        * <Histogram.std>

        Returns
        -------
        * float: the mean value
        """
        # bin_midpoints = self.bins[:-1] + _np.diff(self.bins)/2
        # mean = _np.average(bin_midpoints, weights=self.density)
        # return mean
        return self.sample_ppf(0.5)

    @property
    def std(self):
        """
        Return the standard deviation about the mean from the histogram.

        See also:

        * <Histogram.mean>

        Returns
        ---------
        * float: the standard deviation
        """
        mean, std = self._mean_std()
        return std

    def _mean_std(self):
        """
        Compute the mean and standard deviation.  Useful just to avoid multiple
        calls when both are needed.
        """
        bin_midpoints = self.bins[:-1] + _np.diff(self.bins)/2
        mean = _np.average(bin_midpoints, weights=self.density)

        var = _np.average((bin_midpoints - mean)**2, weights=self.density)
        sigma = _np.sqrt(var)

        return mean, sigma

    def logp(self, x, unit=None):
        """
        Give the log probability of the underlying distribution for a given value
        x.

        Raises
        ----------
        * NotImplementedError
        """
        raise NotImplementedError("logp not implemented for Histogram distributions")

    def to_gaussian(self):
        """
        Convert the <Histogram> distribution to a <Gaussian> distribution by
        adopting the values of <Histogram.mean> and <Histogram.std>.

        Returns
        --------
        * a <Gaussian> object
        """
        mean, std = self._mean_std()
        return Gaussian(mean, std, label=self.label, unit=self.unit, wrap_at=self.wrap_at)

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
        return self.to_gaussian(label=self.label, unit=self.unit).to_uniform(sigma=sigma)

class Delta(BaseDistribution):
    """
    A Delta distribution will _always_ return the central values.  In most cases,
    there is no need to manually create a Delta distribution.  But when doing
    math on other <BaseDistribution> objects, <Delta> distributions are often
    created for clarity.

    Can be created from the top-level via the <npdists.delta> convenience function.
    """
    def __init__(self, value=0.0, unit=None, label=None, wrap_at=None):
        """
        Create a <Delta> distribution.

        This can also be created from a function at the top-level as:

        * <npdists.delta>

        Arguments
        --------------
        * `value` (float or int, default=0.0): the value at which the delta function is True.
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
                                    delta, ('value',),
                                    self._sample_from_delta, self._sample_ppf_from_delta, ('value',),
                                    ('value', value, is_float))

    def _sample_from_delta(self, value, size=None):
        if size is None:
            return value
        else:
            return _np.full(size, value)

    def _sample_ppf_from_delta(self, ppf, value):
        if isinstance(ppf, float):
            return value
        else:
            return _np.full_like(ppf, value)

    def __mul__(self, other):
        if isinstance(other, Delta):
            other = other.value

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.value *= other
            return dist

        return super(Delta, self).__mul__(other)

    def __div__(self, other):
        return self.__mul__(1./other)

    def __add__(self, other):
        if isinstance(other, Delta):
            other = other.value

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.value += other
            return dist

        return super(Delta, self).__add__(other)

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __float__(self):
        return self.value

    @property
    def mean(self):
        """
        The mean value of a delta function is the value itself.

        Returns
        -------
        * float: the value
        """
        return self.value

    @property
    def std(self):
        """
        The standard deviation of a delta function is 0.0 by definition

        Returns
        --------
        * float: 0.0
        """
        return 0.0

    def to_uniform(self):
        """
        Convert the <Delta> distribution to a <Uniform> distribution in which
        both the lower and upper bounds are the same as the value.

        Returns
        ----------
        * a <Uniform> object
        """
        low = self.value
        high = self.value
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
        return Gaussian(self.value, 0.0, label=self.label, unit=self.unit, wrap_at=self.wrap_at)


class Gaussian(BaseDistribution):
    """
    A Gaussian (or Normal) distribution uses [numpy.random.normal](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html)
    to sample values from a gaussian function.

    Can be created from the top-level via the <npdists.gaussian> or
    <npdists.normal> convenience functions.
    """
    def __init__(self, loc=0.0, scale=1.0, unit=None, label=None, wrap_at=None):
        """
        Create a <Gaussian> distribution.

        This can also be created from a function at the top-level as:

        * <npdists.gaussian>

        Arguments
        --------------
        * `loc` (float or int, default=0.0): the central value of the gaussian distribution.
        * `scale` (float or int, default=1.0): the scale (sigma) of the gaussian distribution.
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
                                       gaussian, ('loc', 'scale'),
                                       _np.random.normal, _norm.ppf if _has_scipy else _raise_import_scipy, ('loc', 'scale'),
                                       ('loc', loc, is_float), ('scale', scale, is_float))

    def __mul__(self, other):
        if isinstance(other, Delta):
            other = other.value

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
            other = other.value

        if (isinstance(other, float) or isinstance(other, int)):
            dist = self.copy()
            dist.loc += other
            return dist

        return super(Gaussian, self).__add__(other)

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __float__(self):
        return self.loc
        # return self.sample()

    @property
    def mean(self):
        """
        The mean of a <Gaussian> distribution is the value of `loc`, by definition.

        See also:

        * <Gaussian.std>

        Returns
        --------
        * float
        """
        return self.loc

    @property
    def std(self):
        """
        The standard deviation of a <Gaussian> distribution is the value of `scale`,
        by definition.

        See also:

        * <Gaussian.mean>

        Returns
        --------
        * float
        """
        return self.scale

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


class Uniform(BaseDistribution):
    """
    A Uniform (or Boxcar) distribution gives equal weights to all values within
    the defined range and uses [numpy.random.uniform](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html)
    to sample values.

    Can be created from the top-level via the <npdists.uniform> or
    <npdists.boxcar> convenience functions.
    """
    def __init__(self, low=0.0, high=1.0, unit=None, label=None, wrap_at=None):
        """
        Create a <Uniform> distribution.

        This can also be created from a function at the top-level as:

        * <npdists.uniform>

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
                                      uniform, ('low', 'high'),
                                      self._sample_uniform, self._sample_ppf_uniform, ('low', 'high'),
                                      ('low', low, is_float), ('high', high, is_float))


    def __mul__(self, other):
        if isinstance(other, Delta):
            other = other.value

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
            other = other.value

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

    def __float__(self):
        return self.mean
        # return self.sample()

    def _sample_uniform(self, low, high, size=None):
        if low > high:
            # NOTE: we cannot send the wrap_at argument from sample or plot
            wrap_at = self.get_wrap_at()
            if not wrap_at:
                raise ValueError("low must be >= high unless wrap_at is set or units are angular.")

            # unwrap low
            high = high + wrap_at

        return _np.random.uniform(low, high, size=size)

    def _sample_ppf_uniform(self, ppf, low, high):
        return ppf*(high-low)+low


    @property
    def mean(self):
        """
        The mean value of a <Uniform> distribution is the average of the `low`
        and `high` values.

        Returns
        -------
        * float
        """
        return (self.low+self.high) / 2.0

    @property
    def std(self):
        """
        Determine the standard deviations of the sampled values, adopting
        `sigma=1.0`.

        See also:

        * <Uniform.to_gaussian>
        """
        sigma = 1.0
        return (self.high - self.low) / (2.0 * sigma)

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
        loc = self.mean
        scale = (self.high - self.low) / (2.0 * sigma)
        return Gaussian(loc, scale, unit=self.unit, label=self.label, wrap_at=self.wrap_at)

######################## MULTI-VARIATE DISTRIBUTIONS ###########################


class BaseMultivariateDistribution(BaseDistribution):
    def __init__(self, *args, **kwargs):
        # TODO: handle units, labels, wrap_ats
        self.dimension = kwargs.pop('dimension', None)

        super(BaseMultivariateDistribution, self).__init__(*args, **kwargs)

    @property
    def label(self):
        """
        """
        if self.dimension is None:
            return self._label
        else:
            return self._label[self.dimension]

    @label.setter
    def label(self, label):
        if not (label is None or isinstance(label, list)):
            raise TypeError("label must be of type list")

        self._label = label

    @property
    def dist_args(self):
        """
        Return the arguments sent to the distribution function.

        See also:

        * <<class>.dist_func>
        * <<class>.distribution>

        Returns
        --------
        * tuple
        """

        return tuple([getattr(self, k) for k in self._dist_args]+[self.dimension])

    @property
    def dimensions(self):
        """
        """
        return range(self.ndimensions)

    def get_dimension_by_label(self, dimension):
        """
        """
        if isinstance(dimension, str) and dimension in self.label:
            dimension = self.label.index(dimension)
        return dimension

    @property
    def dimension(self):
        """
        See also:

        * <<class>.sample>

        Returns
        ---------
        * (int or None): dimension of the multivariate distribution to sample.
            If None, will return an array of values for all available parameters.
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dimension):
        dimension = self.get_dimension_by_label(dimension)

        if not (isinstance(dimension, int) or dimension is None):
            raise TypeError("dimension must be of type int")

        # TODO: check to make sure within valid range?  Then we'll probably have to set after super

        self._dimension = dimension

    def take_dimension(self, dimension=None):
        """

        Arguments
        ----------
        * `dimension` (int, list of ints, or None, optional, default=None)
        """
        if dimension is None:
            dimension = self.dimensions
        if not isinstance(dimension, str) and hasattr(dimension, '__iter__'):
            return [self.take_dimension(d) for d in dimension]

        d = self.copy()
        d.dimension = dimension
        return d

    def logp(self, x, dimension=None, unit=None):
        """
        """
        if dimension is None:
            dimension = self.dimension
        if dimension is None:
            dimension = self.dimensions

        if len(x) != len(dimension):
            raise ValueError("x must be same length as dimensions ({})".format(len(dimension)))

        # need to "slice" through the covariance matrix (for gaussian at least),
        # then return a single logp (whould it be multiplied by len(x)?)
        raise NotImplementedError


    def sample(self, *args, **kwargs):
        """

        * `dimension`: (int, optional): dimension of the multivariate distribution
            to sample.  If not provided or None, will default to <<class>.dimension>.
        * `*args`, `**kwargs`: all additional arguments and keyword arguments
            are passed on to <BaseDistribution.sample>.
        """
        dimension = self.get_dimension_by_label(kwargs.pop('dimension', self.dimension))
        sample = super(BaseMultivariateDistribution, self).sample(*args, **kwargs)

        if dimension is not None:
            if len(sample.shape) == 1:
                return sample[dimension]
            else:
                return sample[:, dimension]
        else:
            return sample

    def plot(self, *args, **kwargs):
        """
        """
        dimension = self.get_dimension_by_label(kwargs.pop('dimension', None))
        if dimension is not None:
            return self.take_dimension(dimension).plot(*args, **kwargs)
        elif self.dimension is not None:
            return super(BaseMultivariateDistribution, self).plot(*args, **kwargs)
        else:
            # then we need to do a corner plot
            if not _has_corner:
                raise ImportError("corner must be installed to plot multivariate distributions.  Either install corner or pass a value to dimension to plot a 1D distribution.")

            return corner.corner(self.sample(size=100000), labels=self.label, **kwargs)

    def to_histogram(self, N=1000, bins=10, range=None, dimension=None):
        """
        Convert the <<class>> distribution to a <Histogram> distribution.

        Under-the-hood, this calls <<class>.sample> with `size=N` and `wrap_at=False`
        and passes the resulting array as well as the requested `bins` and `range`
        to <Histogram.from_data>.

        Arguments
        -----------
        * `N` (int, optional, default=1000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.
        * `dimension` (int or string, default=None): dimension to use
            when flattening to the 1-D histogram distribution. If not proivded
            or None, will use value from <<class>.dimension>.  `dimension` is
            therefore REQUIRED if <<class>.dimension> is None.

        Returns
        --------
        * a <Histogram> object

        Raises
        ---------
        * ValueError: if `dimension` and <<class>.dimension> are both None.
        """
        if dimension is None:
            dimension = self.dimension

        dimension = self.get_dimension_by_label(dimension)

        if dimension is None:
            raise ValueError("must provide dimension.")

        unit = self.unit[dimension] if isinstance(self.unit, list) else self.unit
        label = self.label[dimension] if isinstance(self.label, list) else self.label
        wrap_at = self.wrap_at[dimension] if isinstance(self.wrap_at, list) else self.wrap_at

        return Histogram.from_data(self.sample(dimension=dimension, size=N, wrap_at=False),
                                   bins=bins, range=range,
                                   unit=unit, label=label, wrap_at=wrap_at)


class MVGaussian(BaseMultivariateDistribution):
    def __init__(self, locs=0.0, cov=1.0, unit=None, label=None, wrap_at=None):
        """
        Create a <MVGaussian> distribution.

        This can also be created from a function at the top-level as:

        * <npdists.mvgaussian>

        Arguments
        --------------
        * `locs` (float or int, default=0.0): the central value of the gaussian distribution.
        * `cov` (float or int, default=1.0): the scale (sigma) of the gaussian distribution.
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
        * a <MVGaussian> object
        """
        super(MVGaussian, self).__init__(unit, label, wrap_at,
                                       mvgaussian, ('locs', 'cov'),
                                       _np.random.multivariate_normal, None, ('locs', 'cov'),
                                       ('locs', locs, is_iterable), ('cov', cov, is_square_matrix))

    @property
    def ndimensions(self):
        """
        """
        return len(self.locs)

    @property
    def means(self):
        """
        Return the weighted mean values from `locs`.

        See also:

        * <MVGaussian.covariances>

        Returns
        -------
        * list of floats: the mean value per dimension
        """
        return self.locs

    @property
    def covariances(self, N=1000):
        """
        Return the covariances from `cov`

        See also:

        * <MVGaussian.means>

        Returns
        ---------
        * NxN square matrix of floats.
        """
        return self.cov

    def to_mvhistogram(self, N=1000, bins=10, range=None):
        """
        Convert the <<class>> distribution to an <MVHistogram> distribution.

        Under-the-hood, this calls <<class>.sample> with `size=N` and `wrap_at=False`
        and passes the resulting array as well as the requested `bins` and `range`
        to <MVHistogram.from_data>.

        Arguments
        -----------
        * `N` (int, optional, default=1000): number of samples to use for
            the histogram.
        * `bins` (int, optional, default=10): number of bins to use for the
            histogram.
        * `range` (tuple or None): range to use for the histogram.

        Returns
        --------
        * an <MVHistogram> object
        """
        return MVHistogram.from_data(self.sample(size=N, wrap_at=False),
                                   bins=bins, range=range,
                                   unit=self.unit, label=self.label, wrap_at=self.wrap_at)


class MVHistogram(BaseMultivariateDistribution):
    """
    """
    def __init__(self, bins, density, unit=None, label=None, wrap_at=None):
        """
        """
        super(MVHistogram, self).__init__(unit, label, wrap_at,
                                        histogram, ('bins', 'density'),
                                        _sample_from_hist, None, ('bins', 'density'),
                                        ('bins', bins, is_iterable), ('density', density, is_iterable))

    @classmethod
    def from_data(cls, data, bins=10, range=None, weights=None,
                  label=None, unit=None, wrap_at=None):
        """
        """
        hist, bin_edges = _np.histogramdd(data, bins=bins, range=range, weights=weights, normed=True) # what version of numpy introduced density?

        return cls(_np.asarray(bin_edges), hist, label=label, unit=unit, wrap_at=wrap_at)

    @property
    def ndimensions(self):
        """
        """
        return self.bins.shape[0]

    def plot(self, *args, **kwargs):
        """
        """
        dimension = self.get_dimension_by_label(kwargs.get('dimension', None))
        if dimension is not None:
            kwargs.setdefault('bins', self.bins[dimension])

        return super(MVHistogram, self).plot(*args, **kwargs)


    @property
    def means(self):
        """
        Return the weighted mean values from the histogram.

        See also:

        * <MVHistogram.covariances>

        Returns
        -------
        * list of floats: the mean value per dimension
        """
        return [self.to_histogram(dimension=d).mean for d in self.dimensions]

    def get_covariances(self, N=1e5):
        """
        Return the covariances about the mean from the histogram.

        Under-the-hood, this calls `np.cov` on the output from <<class>.sample>
        with `N` samples.

        See also:

        * <MVHistogram.covariances>
        * <MVHistogram.means>

        Arguments
        ---------
        * `N` (int, default=1e5): number of samples to use to pass to
            `np.cov`.

        Returns
        ---------
        * NxN square matrix of floats.
        """
        return _np.cov(self.sample(int(N)).T)

    @property
    def covariances(self):
        """
        Return the covariances about the mean from the histogram.

        Under-the-hood, this calls `np.cov` on the output from <<class>.sample>
        with 1e5 samples.  To adjust the number of samples, use <<class>.get_covariances> instead.

        See also:

        * <MVHistogram.get_covariances>
        * <MVHistogram.means>

        Returns
        ---------
        * NxN square matrix of floats.
        """
        return self.get_covariances()

    def to_mvgaussian(self, N=1e5):
        """
        Convert the <<class>> distribution to an <MVGaussian> distribution.

        See also:

        * <MVHistogram.means>
        * <MVHistogram.get_covariances>

        Arguments
        ---------
        * `N` (int, default=1e5): number of samples to use when calling
            <<class>.get_covariances>.

        Returns
        --------
        * an <MVGaussian> object
        """
        return MVGaussian(self.means, self.get_covariances(N), unit=self.unit, label=self.label, wrap_at=self.wrap_at)
