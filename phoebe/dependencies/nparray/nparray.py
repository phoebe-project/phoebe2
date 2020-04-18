import numpy as np
from collections import OrderedDict
from copy import deepcopy as _deepcopy
import json
import sys

try:
    from astropy import units
except ImportError:
    _has_astropy = False
else:
    _has_astropy = True

if sys.version_info[0] == 3:
  unicode = str

################## VALIDATORS ###################

# these all must accept a single value and return a boolean if it matches the condition as well as any alterations to the value
# NOTE: the docstring is used as the error message if the test fails

def is_unit(value):
    """must be an astropy unit"""
    if not _has_astropy:
        raise ImportError("astropy must be installed for unit support")
    if (isinstance(value, units.Unit) or isinstance(value, units.IrreducibleUnit) or isinstance(value, units.CompositeUnit)):
        return True, value
    else:
        return False, value

def is_unit_or_unitstring(value):
    """must be an astropy.unit"""
    if is_unit(value)[0]:
        return True, value
    try:
        unit = units.Unit(value)
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
    if isinstance(value, str) or isinstance(value, unicode):
        if value.upper() == 'TRUE':
            return True, True
        elif value.upper() == 'FALSE':
            return True, False
        else:
            return False, None

    return isinstance(value, bool), value

def is_float(value):
    """must be a float"""
    if isinstance(value, str) or isinstance(value, unicode):
        try:
            value = float(value)
        except:
            return False, None
        else:
            return True, value
    return isinstance(value, float) or isinstance(value, int) or isinstance(value, np.float64), float(value)

def is_int(value):
    """must be an integer"""
    if isinstance(value, str) or isinstance(value, unicode):
        if "." in value:
            return False, None

        try:
            value = int(float(value))
        except:
            return False, None
        else:
            return True, value
    return isinstance(value, int), value

def is_int_positive(value):
    """must be a positive integer"""
    _is_int, value = is_int(value)
    return _is_int and value > 0, value

def is_int_positive_or_none(value):
    """must be a postive integer or None"""
    if value is None:
        return True, value
    return is_int_positive(value)

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
    return isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple), value


############# WRAPPERS ###################

class ArrayWrapper(object):
    """
    ArrayWrapper should be subclassed by any type of array creation classes
    and the corresponding function should be added to __init__.py.  Any subclass
    MUST define the following:

    __init__ (see docs below)
    array (@property - see docs below)
    __math__(self, operator, other)
    """
    def __init__(self, *args):
        """
        all subclasses MUST parse args and send in as tuples via super so that
        order is preserved.

        For example:
        def __init__(self, start, stop, step):
            super(MyClass, self).__init__(('start', start, is_float), ('stop', stop, is_float), ('step', step, is_float))

        All of these "descriptors" will then be available to get and set via
        their attribute names
        """
        self._descriptors = OrderedDict()
        self._validators = OrderedDict()

        for item in args:
            valid, validated_value = item[2](item[1])
            if valid:
                self._descriptors[item[0]] = validated_value
            else:
                raise ValueError("{} {}, got {}".format(item[0], item[2].__doc__, item[1]))
            self._validators[item[0]] = item[2]

    @property
    def array(self):
        """
        all subclasses MUST define a array property method which returns a
        np.ndarray object
        """
        raise NotImplementedError

    @property
    def quantity(self):
        """
        Return the underlying astropy quantity (if astropy is installed).

        Returns
        --------
        * astropy quantity
        """
        if not _has_astropy:
            raise ImportError("astropy must be installed for unit/quantity support")

        if self.unit is None:
            raise ValueError("unit is not set, cannot convert to quantity")

        return self.array * self.unit

    def to(self, unit):
        """
        Convert between units.  Returns a new nparray object with the new units.

        Arguments
        ---------
        * `unit` (astropy unit or string): new units

        Returns
        -------
        * nparray object of the same class as the original
        """
        if not _has_astropy:
            raise ImportError("astropy must be installed for unit/quantity support")

        if self.unit is None:
            raise ValueError("no units currently set")

        if not is_unit_or_unitstring(unit)[0]:
            raise ValueError("unit not recognized")

        mult_factor = self.unit.to(unit)
        copy = self.copy() * mult_factor
        copy.unit = unit
        return copy

    # @property
    # def __class__(self):
    #     """
    #     override the class so that isinstance(object, np.ndarray) returns True
    #
    #     NOTE: this is really hacky
    #     """
    #     return np.ndarray

    def __getattr__(self, name):
        """
        for anything that isn't overriden here, call the method on the array itself
        """
        if name in ['_descriptors', '_validators']:
            # then we need to actually get the attribute
            return super(ArrayWrapper, self).__getattr__(name)
        elif name in self._descriptors.keys():
            # then get the item in the dictionary
            return self._descriptors.get(name)
        elif hasattr(self.array, name):
            # then fallback on the underlying array object
            return getattr(self.array, name)
        else:
            raise AttributeError("neither '{}' or '{}' have attribute '{}'".format(self.__class__.__name__.lower(), 'numpy.ndarray', name))

    def __setattr__(self, name, value):
        """
        """
        if name in ['_descriptors', '_validators', '__class__']:
            return super(ArrayWrapper, self).__setattr__(name, value)
        elif name in self._descriptors.keys():
            valid, validated_value = self._validators[name](value)
            if valid:
                self._descriptors[name] = validated_value
            else:
                raise ValueError("{} {}".format(name, validator.__doc__))
        else:
            return setattr(self.array, name, value)

    def __getitem__(self, index):
        return self.array.__getitem__(index)

    def __setitem__(self, index, value):
        """
        setting an item in the array needs to fallback on the Array class
        """
        arrayvalue = self.array
        arrayvalue.__setitem__(index, value)
        self._convert_to_array(arrayvalue)


    # def __dir__(self):
    #     """
    #     override the dir() functionality to also include all of the
    #     methods/attributes of self.array
    #     """

    def __repr__(self):
        descriptors = " ".join(["{}={}".format(k,v) for k,v in self._descriptors.items()])
        return "<{} {}>".format(self.__class__.__name__.lower(), descriptors)

    def __copy__(self):
        return self.__class__(**self._descriptors)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = self.__class__.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _deepcopy(v, memo))
        return result

    def copy(self):
        """
        Create and return a copy.
        """
        return self.__copy__()

    def to_dict(self):
        """
        Dump a representation of the nparray object to a dictionary.  The
        nparray object should then be able to be fully restored via
        <nparray.from_dict>

        Returns
        ----------
        * (dictionary)
        """
        def _json_safe(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif is_unit(v)[0]:
                return v.to_string()
            else:
                return v
        d = {k:_json_safe(v) for k,v in self._descriptors.items()}
        d['nparray'] = self.__class__.__name__.lower()
        return d

    def to_json(self, **kwargs):
        """
        Dump a representation of the nparray object to a json-formatted string.
        The nparray object should then be able to be fully restored via
        <nparray.from_json>.

        Arguments
        -----------
        * `**kwargs`: all keyword arguments are sent to json.dumps

        Returns
        ---------
        * (string): json formatted string.
        """
        return json.dumps(self.to_dict(), **kwargs)

    def to_file(self, filename, **kwargs):
        """
        Dump a representation of the nparray object to a json-formatted file.
        The nparray object should then be able to be fully restored via
        <nparray.from_file>.

        Arguments
        -----------
        * `filename` (string): path to the file to be created (will overwrite
            if already exists)
        * `**kwargs`: additional keyword arguments are passed to
            json.dumps.

        Returns
        -----------
        * (string): the filename
        """
        f = open(filename, 'w')
        f.write(self.to_json(**kwargs))
        f.close()
        return filename

    def to_array(self):
        """
        Convert to an nparray <Array> object (**note**: not a numpy array, to access
        the underlying numpy object, call the .array property)

        Returns
        ---------
        * <Array>
        """
        return Array(self.array)

    def _convert_to_array(self, value=None):
        if value is None:
            value = self.array
        self.__class__ = Array
        self.__init__(value)

    def __add__(self, other):
        return self.__math__('__add__', other)

    def __radd__(self, other):
        return self.__math__('__radd__', other)

    def __sub__(self, other):
        return self.__math__('__sub__', other)

    def __rsub__(self, other):
        return self.__math__('__rsub__', other)

    def __mul__(self, other):
        if _has_astropy and is_unit(other)[0]:
            copy = self.copy()
            copy.unit = other
            return copy
        else:
            return self.__math__('__mul__', other)

    def __rmul__(self, other):
        return self.__math__('__rmul__', other)

    def __div__(self, other):
        return self.__math__('__div__', other)

    def __rdiv__(self, other):
        return self.__math__('__rdiv__', other)

    def __pow__(self, other):
        return self.__math__('__pow__', other)

    def __rpow__(self, other):
        return Array(self.array.__rpow__(other))

    def __abs__(self):
        return Array(abs(self.array))

    def __len__(self):
        return len(self.array)

    @property
    def shape(self):
        if 'shape' in self._descriptors.keys():
            # Ones, Full, Zeros has a shape attribute directly
            shape = self._descriptors.get('shape')
            if isinstance(shape, int):
                return (shape,)
            return shape
        return self.array.shape

    def __comparison__(self, operator, other):
        """
        determine comparisons based on the underyling arrays
        """
        return getattr(self.array, operator)(other)

    def __eq__(self, other):
        return self.__comparison__('__eq__', other)

    def __ne__(self, other):
        return self.__comparison__('__ne__', other)

    def __lt__(self, other):
        return self.__comparison__('__lt__', other)

    def __lte__(self, other):
        return self.__comparison__('__le__', other)

    def __gt__(self, other):
        return self.__comparison__('__gt__', other)

    def __ge__(self, other):
        return self.__comparison__('__ge__', other)

    def __contains__(self, other):
        return self.__comparison__('__contains__', other)

class Array(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.array>.
    """
    def __init__(self, value, unit=None):
        """
        This is available as a top-level convenience function as <nparray.array>.

        Arguments
        ------------
        * `value` (array or list): array or list of values.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Array>
        """
        super(Array, self).__init__(('value', value, is_iterable), ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.array(value)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.array(self.value)

    def __math__(self, operator, other):
        value = getattr(np.asarray(self.value), operator)(other)
        return Array(value, self.unit)

    def __setitem__(self, index, value):
        """
        for Arrays, we simply edit the underlying numpy array/list/tuple
        """
        # TODO: will this cause issues if value is a tuple? what is the expected behavior
        self.value.__setitem__(index, value)

class Arange(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.arange>.
    """
    def __init__(self, start, stop, step, unit=None):
        """
        This is available as a top-level convenience function as <nparray.arange>.

        Arguments
        ------------
        * `start` (int or float): the starting point of the sequence.
        * `stop` (int or float): the ending point of the sequence.  The interval
            does not include this value, except in some cases where `step` is not an
            integer and floating point round-off affects the length of the array.
        * `step` (int or float): the stepsize between each item in the sequence.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Arange>
        """
        super(Arange, self).__init__(('start', start, is_float),
                                     ('stop', stop, is_float),
                                     ('step', step, is_float),
                                     ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.arange(start, stop, step)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.arange(self.start, self.stop, self.step)

    def to_linspace(self):
        """
        Convert from <Arange> to <Linspace>.

        ```py
        num = int((self.stop - self.start)/self.step)
        Linspace(self.start, self.stop-self.step, num)
        ```

        Returns
        -------
        * <Linspace>
        """
        num = int((self.stop-self.start)/self.step)
        return Linspace(self.start, self.stop-self.step, num)

    def to_array(self):
        """
        Convert from <Arange> to <Array> by executing <Arange.array>.

        Returns
        --------
        * <Array>
        """
        return Array(self.array)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            # NOTE: we have to cast the start and stop to a float because
            # int.__mul__(float) is not implemented in Python (instead it uses
            # float.__rmul__(int)).
            # Alternatively,we could check for this case and do
            # other.__r{operator}(self.start) but that seems just as ugly.
            return Arange(getattr(float(self.start), operator)(other), getattr(float(self.stop), operator)(other), self.step, self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))

class Linspace(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.linspace>.
    """
    def __init__(self, start, stop, num, endpoint=True, unit=None):
        """
        This is available as a top-level convenience function as <nparray.linspace>.

        Arguments
        ------------
        * `start` (int or float): the starting point of the sequence.
        * `stop` (int or float): the ending point of the sequence, unless `endpoint`
            is set to False.  In that case, the sequence consists of all but the
            last of ``num + 1`` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is False.
        * `num` (int): number of samples to generate.
        * `endpoint` (bool, optional, default=True): If True, `stop` is the last
            sample. Otherwise, it is not included.
        * `unit` (astropy unit or string, optional, default=None): unit
            corresponding to the passed values.

        Returns
        -----------
        * <Linspace>
        """
        super(Linspace, self).__init__(('start', start, is_float),
                                       ('stop', stop, is_float),
                                       ('num', num, is_int_positive),
                                       ('endpoint', endpoint, is_bool),
                                       ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.linspace(start, stop, num, endpoint)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.linspace(self.start, self.stop, self.num, self.endpoint)

    def to_arange(self):
        """
        Convert from <Linspace> to <Arange>

        ```py
        arr, step = np.linspace(self.start, self.stop, self.num, self.endpoint, retstep=True)
        Arange(self.start, self.stop+step, step)
        ```

        Returns
        ----------
        * <Arange>
        """
        arr, step = np.linspace(self.start, self.stop, self.num, self.endpoint, retstep=True)
        return Arange(self.start, self.stop+step, step)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            # NOTE: we have to cast the start and stop to a float because
            # int.__mul__(float) is not implemented in Python (instead it uses
            # float.__rmul__(int)).
            # Alternatively,we could check for this case and do
            # other.__r{operator}(self.start) but that seems just as ugly.
            return Linspace(getattr(float(self.start), operator)(other), getattr(float(self.stop), operator)(other), self.num, self.endpoint, self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))

class Invspace(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.invspace>.
    """
    def __init__(self, start, stop, num, endpoint=True, unit=None):
        """
        This is available as a top-level convenience function as <nparray.invspace>.

        Arguments
        ------------
        * `start` (int or float): the starting point of the sequence.
        * `stop` (int or float): the ending point of the sequence, unless `endpoint`
            is set to False.  In that case, the sequence consists of all but the
            last of ``num + 1`` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is False.
        * `num` (int): number of samples to generate.
        * `endpoint` (bool, optional, default=True): If True, `stop` is the last
            sample. Otherwise, it is not included.
        * `unit` (astropy unit or string, optional, default=None): unit
            corresponding to the passed values.

        Returns
        -----------
        * <Invspace>
        """
        super(Invspace, self).__init__(('start', start, is_float),
                                       ('stop', stop, is_float),
                                       ('num', num, is_int_positive),
                                       ('endpoint', endpoint, is_bool),
                                       ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        1./np.linspace(1./start, 1./stop, num, endpoint)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return 1./np.linspace(1./self.start, 1./self.stop, self.num, self.endpoint)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            # NOTE: we have to cast the start and stop to a float because
            # int.__mul__(float) is not implemented in Python (instead it uses
            # float.__rmul__(int)).
            # Alternatively,we could check for this case and do
            # other.__r{operator}(self.start) but that seems just as ugly.
            return Invspace(getattr(float(self.start), operator)(other), getattr(float(self.stop), operator)(other), self.num, self.endpoint, self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))

class Logspace(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.logspace>.
    """
    def __init__(self, start, stop, num, endpoint=True, base=10.0, unit=None):
        """
        This is available as a top-level convenience function as <nparray.logspace>.

        Arguments
        ------------
        * `start` (int or float): ``base ** start`` is the starting value of the sequence.
        * `stop` (int or float): ``base ** stop`` is the final value of the sequence,
            unless `endpoint` is False.  In that case, ``num + 1`` values are spaced
            over the interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        * `num` (int): number of samples to generate.
        * `endpoint` (bool, optional, default=True): If True, `stop` is the last
            sample. Otherwise, it is not included.
        * `base` (float, optional, default=10.0): The base of the log space. The
            step size between the elements in ``ln(samples) / ln(base)``
            (or ``log_base(samples)``) is uniform.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Logspace>
        """
        super(Logspace, self).__init__(('start', start, is_float),
                                       ('stop', stop, is_float),
                                       ('num', num, is_int_positive),
                                       ('endpoint', endpoint, is_bool),
                                       ('base', base, is_float),
                                       ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.logspace(start, stop, num, endpoint, base)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.logspace(self.start, self.stop, self.num, self.endpoint, self.base)

    def _math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            return Logspace(getattr(self.start, operator)(other), getattr(self.stop, operator)(other), self.num, self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))


class Geomspace(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.geomspace>.
    """
    def __init__(self, start, stop, num, endpoint=True, unit=None):
        """
        This is available as a top-level convenience function as <nparray.geomspace>.

        Arguments
        ------------
        * `start` (int or float): the starting point of the sequence.
        * `stop` (int or float): the final value of the sequence, unless `endpoint`
            is False.  In that case, ``num + 1`` values are spaced over the
            interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        * `num` (int): number of samples to generate.
        * `endpoint` (bool, optional, default=True): If True, `stop` is the last
            sample. Otherwise, it is not included.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Geomspace>
        """
        super(Geomspace, self).__init__(('start', start, is_float),
                                       ('stop', stop, is_float),
                                       ('num', num, is_int_positive),
                                       ('endpoint', endpoint, is_bool),
                                       ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.geomspace(start, stop, num, endpoint)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.geomspace(self.start, self.stop, self.num, self.endpoint)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            # NOTE: we have to cast the start and stop to a float because
            # int.__mul__(float) is not implemented in Python (instead it uses
            # float.__rmul__(int)).
            # Alternatively,we could check for this case and do
            # other.__r{operator}(self.start) but that seems just as ugly.
            return Geomspace(getattr(float(self.start), operator)(other), getattr(float(self.stop), operator)(other), self.num, self.endpoint, self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))

class Full(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.full> or
    <nparray.full_like>.
    """
    def __init__(self, shape, fill_value, unit=None):
        """
        This is available as a top-level convenience function as <nparray.full>
        or <nparray.full_like>.

        Arguments
        ------------
        * `shape` (int or sequence of ints): Shape of the new array, e.g.,
            ``(2, 3)`` or ``2``.
        * `fill_value` (int or float): Value to fill each element in the array.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Full>
        """
        super(Full, self).__init__(('shape', shape, is_valid_shape),
                                    ('fill_value', fill_value, is_float),
                                    ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.full(shape, fill_value)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.full(self.shape, self.fill_value)

    def to_linspace(self):
        """
        Convert from <Full> to <Linspace> (only supported for flat arrays in
        which `shape` is an integer, not a tuple).

        ```py
        Linspace(self.fill_value, self.fill_value, self.shape)
        ```

        Returns
        --------
        * <Linspace>

        Raises
        --------
        * NotImplementedError: if the `shape` is not flat.
        """
        if hasattr(self.shape, '__len__'):
            raise NotImplementedError("can only convert flat Full arrays to linspace")
        return Linspace(self.fill_value, self.fill_value, self.shape)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            # NOTE: we have to cast the fill_value to a float because
            # int.__mul__(float) is not implemented in Python (instead it uses
            # float.__rmul__(int)).
            # Alternatively,we could check for this case and do
            # other.__r{operator}(self.fill_value) but that seems just as ugly.
            return Full(self.shape, getattr(float(self.fill_value), operator)(other), self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, Full) or isinstance(other, Zeros) or isinstance(other, Ones):
            if self.shape==other.shape:
                if isinstance(other, Full):
                    other_fill_value = other,fill_value
                elif isinstance(other, Zeros):
                    other_fill_value = 0.0
                elif isinstance(other, Ones):
                    other_fill_value = 1.0
                return Full(self.shape, getattr(self.fill_value, operator)(other_fill_value), self.unit)
            else:
                raise ValueError("cannot add with unmatching shapes")
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))


class Zeros(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.zeros>
    or <nparray.zeros_like>.
    """
    def __init__(self, shape, unit=None):
        """
        This is available as a top-level convenience function as <nparray.zeros>
        or <nparray.zeros_like>.

        Arguments
        ------------
        * `shape` (int or sequence of ints): Shape of the new array, e.g.,
            ``(2, 3)`` or ``2``.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Zeros>
        """
        super(Zeros, self).__init__(('shape', shape, is_valid_shape),
                                    ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.zeros(shape)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.zeros(self.shape)

    def to_full(self):
        """
        Convert from <Zeros> to <Full>.

        ```py
        Full(self.shape, 0)
        ```

        Returns
        ---------
        * <Full>
        """
        return Full(self.shape, 0)

    def to_linspace(self):
        """
        Convert from <Zeros> to <Linspace> (only supported for flat arrays in
        which `shape` is an integer, not a tuple).

        ```py
        Linspace(0, 0, self.shape)
        ```

        Returns
        --------
        * <Linspace>

        Raises
        --------
        * NotImplementedError: if the `shape` is not flat.
        """
        if hasattr(self.shape, '__len__'):
            raise NotImplementedError("can only convert flat Zeros arrays to linspace")
        return Linspace(0, 0, self.shape)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            return Full(self.shape, getattr(0.0, operator)(other), self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, Full) or isinstance(other, Zeros) or isinstance(other, Ones):
            if self.shape==other.shape:
                if isinstance(other, Full):
                    other_fill_value = other.fill_value
                elif isinstance(other, Zeros):
                    other_fill_value = 0.0
                elif isinstance(other, Ones):
                    other_fill_value = 1.0
                return Full(self.shape, getattr(0.0, operator)(other_fill_value), self.unit)
            else:
                raise ValueError("cannot add with unmatching shapes")
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))

class Ones(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.ones> or
    <nparray.ones_like>.
    """
    def __init__(self, shape, unit=None):
        """
        This is available as a top-level convenience function as <nparray.ones>
        or <nparray.ones_like>.

        Arguments
        ------------
        * `shape` (int or sequence of ints): Shape of the new array, e.g.,
            ``(2, 3)`` or ``2``.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Ones>
        """
        super(Ones, self).__init__(('shape', shape, is_valid_shape),
                                   ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.ones(shape)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.ones(self.shape)

    def to_full(self):
        """
        Convert from <Ones> to <Full>.

        ```py
        Full(self.shape, 1)
        ```

        Returns
        --------
        * <Full>
        """
        return Full(self.shape, 1)

    def to_linspace(self):
        """
        Convert from <Ones> to <Linspace> (only supported for flat arrays in
        which `shape` is an integer, not a tuple).

        ```py
        Linspace(1, 1, self.shape)
        ```

        Returns
        --------
        * <Linspace>

        Raises
        --------
        * NotImplementedError: if the `shape` is not flat.
        """
        if hasattr(self.shape, '__len__'):
            raise NotImplementedError("can only convert flat Ones arrays to linspace")
        return Linspace(1, 1, self.shape)

    def __math__(self, operator, other):
        if isinstance(other, float) or isinstance(other, int):
            return Full(self.shape, getattr(1.0, operator)(other), self.unit)
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            value = getattr(self.array, operator)(other)
            return Array(value, self.unit)
        elif isinstance(other, Full) or isinstance(other, Zeros) or isinstance(other, Ones):
            if self.shape==other.shape:
                if isinstance(other, Full):
                    other_fill_value = other.fill_value
                elif isinstance(other, Zeros):
                    other_fill_value = 0.0
                elif isinstance(other, Ones):
                    other_fill_value = 1.0
                return Full(self.shape, getattr(1.0, operator)(other_fill_value), self.unit)
            else:
                raise ValueError("cannot add with unmatching shapes")
        elif isinstance(other, ArrayWrapper):
            value = getattr(self.array, operator)(other.array)
            return Array(value, self.unit)
        else:
            raise ValueError("{} not supported with type {}".format(operator, type(other)))

class Eye(ArrayWrapper):
    """
    This is available as a top-level convenience function as <nparray.eye>.
    """
    def __init__(self, M, N=None, k=0, unit=None):
        """
        This is available as a top-level convenience function as <nparray.eye>.

        Arguments
        ------------
        * `M` (int): Number of rows in the output.
        * `N` (int or None, optional, default=None): Number of columns in the output.
            If None, defaults to `N`.
        * `k` (int, optional, default=1): Index of the diagonal: 0 (the default)
            refers to the main diagonal, a positive value refers to an upper
            diagonal, and a negative value to a lower diagonal.
        * `unit` (astropy unit or string, optional, default=None): unit
          corresponding to the passed values.

        Returns
        -----------
        * <Eye>
        """
        super(Eye, self).__init__(('M', M, is_int_positive),
                                  ('N', N, is_int_positive_or_none),
                                  ('k', k, is_int_positive_or_none),
                                  ('unit', unit, is_unit_or_unitstring_or_none))

    @property
    def array(self):
        """
        Compute the underyling numpy array by calling:

        ```py
        np.eye(M, N, k)
        ```

        Returns
        ---------
        * (numpy array): the underlying (computed) numpy array
        """
        return np.eye(self.M, self.N, self.k)

    def __math__(self, operator, other):
        value = getattr(self.to_array(), operator)(other)
        return Array(value, self.unit)
