# astropy-units-IAU2015
light wrapper around astropy.units that changes constants and units to match those in the IAU 2015 resolution.

As of astropy 2.0+, the IAU 2015 constants are adopted by default.  This module simply allows an astropy-version-independent way of ensuring that you are dealing with IAU 2015 constants.

See the following discussions on the [astropy](https://www.github.com/astropy/astropy) project:
* [Implement updated IAU B3 2015 constants if approved](https://github.com/astropy/astropy/issues/4026)
* [Organize constants into version modules](https://github.com/astropy/astropy/pull/6083)
* [Implement updated IAU B3 2015 constants](https://github.com/astropy/astropy/pull/4397)
* [astropy updated constants](https://github.com/astropy/astropy/pull/4956)

If you have astropy 2.0+ installed, this does essentially nothing (except introducing a solTeff unit and T_sun constant).  If, however, you have a version of astropy older than 2.0, it will hack the internal units to match IAU 2015 conventions.

This is currently only tested with python 2.7 and astropy 1.3.1.

## Dependencies

* [astropy](https://www.github.com/astropy/astropy)

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

```
from unitsiau2015 import u, c
print c.G.value
```

Note that once unitsiau2015 is imported, the values within the astropy package will change as well (for that python session).  See [simple.py example](https://github.com/kecnry/astropy-units-IAU2015/blob/master/examples/simple.py) to see this in action.

## Contributing

Contributions are welcome! Feel free to file an issue or fork and create a pull-request.
