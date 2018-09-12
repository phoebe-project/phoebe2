# nparray

Create numpy arrays (via arange, linspace, etc) and manipulate the creation arguments at any time.  The created object acts as a numpy array but only stores the input parameters until its value is accessed.

The following snippet should give you a better idea of the purpose of nparray.  Note that under-the-hood are *actual* numpy arrays, meaning passing directly to matplotlib works fine, but using isinstance or type will currently not recognize the numpy array (at least for now - see [this issue](https://github.com/kecnry/nparray/issues/6)).

```
import nparray as npa

a = npa.arange(0,1,0.1)
print a
# <arange start=0 stop=1 step=0.1>
print a[-1]
# 0.9
a.step=0.5
print a
# <arange start=0 stop=1 step=0.5>
print a[-1]
# 0.5
b = a.to_linspace()
print b
# <linspace start=0 stop=0.5 num=2, endpoint=True>
print b*3
# <linspace start=0, stop=1.5, num=2, endpoint=True>
b[1] = 99
print b
# <array value=[0, 99]>
```

nparray currently supports the following with all arguments (except for dtype - see [open issue](https://github.com/kecnry/nparray/issues/8)):
* [array](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.array.html#numpy.array)
* [arange](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.arange.html#numpy.arange) (convert to array, linspace)
* [linspace](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.linspace.html#numpy.linspace) (convert to array, arange)
* [logspace](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.logspace.html#numpy.logspace) (convert to array)
* [geomspace](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.geomspace.html#numpy.geomspace) (convert to array)
* [full](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.full.html#numpy.full) (convert to array)
* [full_like](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.full_like.html#numpy.full_like) (creates a full array)
* [zeros](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.zeros.html#numpy.zeros) (convert to array, full, linspace)
* [zeros_like](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.zeros_like.html#numpy.zeros_like) (creates a zeros array)
* [ones](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ones.html#numpy.ones) (convert to array, full, linspace)
* [ones_like](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ones_like.html#numpy.ones_like) (creates a ones array)
* [eye](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.eye.html#numpy.eye) (convert to array)

## Dependencies

* [numpy](https://github.com/numpy/numpy)
* collections (should be standard python module)

nparray is currently tested on linux with Python 2.7 and 3.6 and numpy 1.10+.  See the [travis report](https://travis-ci.org/kecnry/nparray) for details on the full testing-matrix.

## Installation

Installation is done using the standard python setup.py commands.

To install globally:

```
python setup.py build
sudo python setup.py install
```

Or to install locally:

```
python setup.py build
python setup.py install --user
```

## Basic Usage

See the snippet above or the examples in the examples directory.

## Contributing

Contributions are welcome! Feel free to file an issue or fork and create a pull-request.
