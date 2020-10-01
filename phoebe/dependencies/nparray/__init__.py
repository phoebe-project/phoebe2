from . import nparray as _wrappers
import numpy as np
import json
import os

from distutils.version import LooseVersion

__version__ = '1.2.0'
version = __version__

# allow isinstance(obj, nparray.ndarray) to be similar to numpy
ndarray = _wrappers.ArrayWrapper

__docprefix__ = """
This is an nparray wrapper around the numpy function.  The
numpy documentation is included below.  Currently most kwargs
should be accepted with the exception of 'dtype'.  The returned
object should act exactly like the numpy array itself, but with
several extra helpful methods and attributes.  Call help on the
resulting object for more information.

If you have astropy installed, units are supported by passing unit=astropy.unit
to the instantiation functions or by multiplying an array with a unit object.

"""

__docsep__ = """

===============================================================

** numpy documentation for underlying function: **

"""

def array(value, unit=None):
    """
    Arguments
    ------------
    * `value` (array or list): array or list of values.
    * `unit` (astropy unit or string, optional, default=None): unit
      corresponding to the passed values.

    Returns
    -----------
    * <Array>
    """
    return _wrappers.Array(value, unit)

array.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in array.__doc__.split("\n")]) + __docsep__ + np.array.__doc__.replace("&gt;", ">")

def arange(start, stop, step, unit=None):
    """
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
    return _wrappers.Arange(start, stop, step, unit)

arange.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in arange.__doc__.split("\n")]) + __docsep__ + np.arange.__doc__.replace("&gt;", ">")

def linspace(start, stop, num, endpoint=True, unit=None):
    """
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
    return _wrappers.Linspace(start, stop, num, endpoint, unit)

linspace.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in linspace.__doc__.split("\n")]) + __docsep__ + np.linspace.__doc__.replace("&gt;", ">")

def invspace(start, stop, num, endpoint=True, unit=None):
    """
    Evenly sampled numbers in inverted space.  This is equivalent to:

    ```py
    1./linspace(1./start, 1./stop, num, endpoint, unit)
    ```

    See also:

    * <nparray.linspace>

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
    return _wrappers.Invspace(start, stop, num, endpoint, unit)

def logspace(start, stop, num, endpoint=True, base=10.0, unit=None):
    """
    See also:

    * <nparray.geomspace>

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
    return _wrappers.Logspace(start, stop, num, endpoint, base, unit)

logspace.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in logspace.__doc__.split("\n")]) + __docsep__ + np.logspace.__doc__.replace("&gt;", ">")

def geomspace(start, stop, num, endpoint=True, unit=None):
    """
    See also:

    * <nparray.logspace>

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
    if LooseVersion(np.__version__) >= LooseVersion("1.13"):
        return _wrappers.Geomspace(start, stop, num, endpoint, unit)
    else:
        raise NotImplementedError("geomspace requires numpy version >= 1.13")

if LooseVersion(np.__version__) >= LooseVersion("1.13"):
    geomspace.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in geomspace.__doc__.split("\n")]) + __docsep__ + np.geomspace.__doc__.replace("&gt;", ">")

def full(shape, fill_value, unit=None):
    """
    See also:

    * <nparray.full_like>

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
    return _wrappers.Full(shape, fill_value, unit)

full.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in full.__doc__.split("\n")]) + __docsep__ + np.full.__doc__.replace("&gt;", ">")

def full_like(a, fill_value, unit=None):
    """
    Note: unlike in the numpy version, the data-type of `a` is not currently
    guaranteed to be maintained.

    See also:

    * <nparray.full>

    Arguments
    ------------
    * `a` (list or array): The shape of `a` define these same attributes of the
        returned array.
    * `fill_value` (int or float): Value to fill each element in the array.
    * `unit` (astropy unit or string, optional, default=None): unit
      corresponding to the passed values.

    Returns
    -----------
    * <Full>
    """
    return _wrappers.Full(a.shape, fill_value, unit)

full_like.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in full_like.__doc__.split("\n")]) + __docsep__ + np.full_like.__doc__.replace("&gt;", ">")

def zeros(shape, unit=None):
    """
    See also:

    * <nparray.zeros_like>

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
    return _wrappers.Zeros(shape, unit)

zeros.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in zeros.__doc__.split("\n")]) + __docsep__ + np.zeros.__doc__.replace("&gt;", ">")

def zeros_like(a, unit=None):
    """
    Note: unlike in the numpy version, the data-type of `a` is not currently
    guaranteed to be maintained.

    See also:

    * <nparray.zeros>

    Arguments
    ------------
    * `a` (list or array): The shape of `a` define these same attributes of the
        returned array.
    * `unit` (astropy unit or string, optional, default=None): unit
      corresponding to the passed values.

    Returns
    -----------
    * <Zeros>
    """
    return _wrappers.Zeros(a.shape, unit)

zeros_like.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in zeros_like.__doc__.split("\n")]) + __docsep__ + np.zeros_like.__doc__.replace("&gt;", ">")

def ones(shape, unit=None):
    """
    See also:

    * <nparray.ones_like>

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
    return _wrappers.Ones(shape, unit)

ones.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in ones.__doc__.split("\n")]) + __docsep__ + np.ones.__doc__.replace("&gt;", ">")

def ones_like(a, unit=None):
    """
    Note: unlike in the numpy version, the data-type of `a` is not currently
    guaranteed to be maintained.

    See also:

    * <nparray.ones>

    Arguments
    ------------
    * `a` (list or array): The shape of `a` define these same attributes of the
        returned array.
    * `unit` (astropy unit or string, optional, default=None): unit
      corresponding to the passed values.

    Returns
    -----------
    * <Ones>
    """
    return _wrappers.Ones(a.shape, unit)

ones_like.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in ones_like.__doc__.split("\n")]) + __docsep__ + np.ones_like.__doc__.replace("&gt;", ">")

def eye(M, N=None, k=0, unit=None):
    """
    Arguments
    ------------
    * `M` (int): Number of rows in the output.
    * `N` (int or None, optional, default=None): Number of columns in the output.
        If None, defaults to `N`.
    * `k` (int, optional, default=0): Index of the diagonal: 0 (the default)
        refers to the main diagonal, a positive value refers to an upper
        diagonal, and a negative value to a lower diagonal.
    * `unit` (astropy unit or string, optional, default=None): unit
      corresponding to the passed values.

    Returns
    -----------
    * <Eye>
    """
    return _wrappers.Eye(M, N, k, unit)

eye.__doc__ = __docprefix__ + "\n".join([l.lstrip() for l in eye.__doc__.split("\n")]) + __docsep__ + np.eye.__doc__.replace("&gt;", ">")

def from_dict(d):
    """
    Load an nparray object from a dictionary.

    See also:

    * <nparray.from_json>
    * <nparray.from_file>

    Arguments
    ----------
    * `d` (dictionary): dictionary representing a valid nparray object

    Returns
    ---------
    * the instatiated nparray object.
    """
    if isinstance(d, str):
        return from_json(d)

    if not isinstance(d, dict):
        raise TypeError("argument must be of type dict")
    if 'nparray' not in d.keys():
        raise ValueError("input dictionary missing 'nparray' entry")

    classname = d.get('nparray').title()
    # instead of popping npdarray (which would happen in memory and make that json
    # unloadable again), we'll do a dictionary comprehension.  If this causes
    # performance issues, we could instead accept and ignore nparray as
    # a keyword argument to __init__
    return getattr(_wrappers, classname)(**{k:v for k,v in d.items() if k!='nparray'})

def from_json(j):
    """
    Load an nparray object from a json-formatted string.

    See also:

    * <nparray.from_dict>
    * <nparray.from_file>

    Arguments
    -----------
    * `j` (string): a json-formatted string representing a valid nparray object

    Returns
    ----------
    * the instantiated nparray object.
    """
    if isinstance(j, dict):
        return from_dict(j)

    if not (isinstance(j, str) or isinstance(j, unicode)):
        raise TypeError("argument must be of type str")

    return from_dict(json.loads(j))

def from_file(filename):
    """
    Load an nparray object from a file.

    See also:

    * <nparray.from_dict>
    * <nparray.from_json>

    Arguments
    -----------
    * `filename` (string): a filename pointing to a valid nparray object.

    Returns
    ----------
    * the instantiated nparray object.
    """
    f = open(filename, 'r')
    j = json.load(f)
    f.close()

    return from_dict(j)

def monkeypatch():
    """
    monkeypatch built-in numpy functions to call those provided by nparray instead.

    ```py
    import nparray
    import numpy as np

    nparray.monkeypatch()
    print(np.linspace(0,1,11))
    ```

    """
    np.array = array
    np.arange = arange
    np.linspace = linspace
    np.logspace = logspace
    np.geomspace = geomspace
    np.full = full
    np.full_like = full_like
    np.zeros = zeros
    np.zeros_like = zeros_like
    np.ones = ones
    np.ones_like = ones_like
    np.eye = eye
