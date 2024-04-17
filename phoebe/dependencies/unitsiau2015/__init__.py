from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy import __version__ as astropyversion

# import these so they'll be available as unitsiau2015.u and unitsiau2015.c
import astropy.constants as c
import astropy.units as u

from astropy.units import def_unit
from astropy.constants import Constant

def _register_unit(unit):
    """
    register this unit in the phoebe.u namespace
    """
    for name in unit._names:
        setattr(u, name, unit)
        for reg in u.core._unit_registries:
            reg._registry[name] = unit


# TODO: pass namespace for units package so that prefixes are loaded (otherwise calling ksolMass will call from the old constants)
ns = None


"""
T_sun is not provided by astropy 2.0, so we'll set the constant and the unit
for any version of astropy
"""

T_sun = Constant('T_sun', "Solar effective temperature", 5772, 'K', None, "IAU 2015 Resolution B3", system='si')
c.T_sun = T_sun
c.si.T_sun = T_sun

solTeff = def_unit(['solTeff', 'T_sun', 'Tsun'], c.si.T_sun, namespace=ns,
        prefixes=True, doc="Solar effective temperature (nominal)",
        format={'latex': r'{\mathcal{T}^{\rm N}_\odot}'})

_register_unit(solTeff)

"""
astropy also doesn't have any ability to convert to solar units, so we'll provide
a function for that.

TODO: eventually consider adopting this as a "base" unit
"""
_physical_types_to_solar = {'length': 'solRad',
                            'area': 'solRad2',
                            'volume': 'solRad3',
                            'mass': 'solMass',
                            'temperature': 'solTeff',
                            'power': 'solLum',
                            'time': 'd',
                            'speed': 'solRad/d',
                            'angle': 'rad',
                            'angular speed': 'rad/d',
                            'angular velocity': 'rad/d',
                            'angular frequency': 'rad/d',
                            'dimensionless': ''}

_physical_types_to_si = {'length': 'm',
                         'area': 'm2',
                         'volume': 'm3',
                         'mass': 'kg',
                         'temperature': 'K',
                         'power': 'W',
                         'time': 's',
                         'speed': 'm/s',
                         'angle': 'rad',
                         'angular speed': 'rad/s',
                         'angular velocity': 'rad/s',
                         'angular frequency': 'rad/s',
                         'dimensionless': ''}

def _get_physical_type_list(object):

    if hasattr(object, 'physical_type'):
        unit = object
    elif isinstance(object, u.Quantity):
        return _get_physical_type_list(object.unit)
    else:
        raise NotImplementedError("object {} with type={} not supported for _get_physical_type_list".format(object, type(object)))

    if hasattr(unit.physical_type, '_physical_type_list'):
        return unit.physical_type._physical_type_list
    else:
        return [str(unit.physical_type)]

def _get_physical_type(object):
    physical_type_list = _get_physical_type_list(object)
    for physical_type in physical_type_list:
        if str(physical_type) in _physical_types_to_solar.keys():
            return str(physical_type)
    return str(physical_type_list[0])

def can_convert_to_solar(object):
    return _get_physical_type(object) in _physical_types_to_solar.keys()

def to_solar(object):
    """
    Arguments
    ----------
    * `object` (quantity or unit)

    Returns
    ----------
    * quantity or unit in applicable solar units

    Raises
    --------
    * NotImplementedError: if cannot convert to solar
    """
    physical_type = _get_physical_type(object)

    if physical_type not in _physical_types_to_solar.keys():
        raise NotImplementedError("cannot convert object with physical_type={} to solar units".format(physical_type))

    return object.to(_physical_types_to_solar.get(physical_type))

u.can_convert_to_solar = can_convert_to_solar
u.to_solar = to_solar
u._get_physical_type = _get_physical_type
u._physical_types_to_si  = _physical_types_to_si
u._physical_types_to_solar = _physical_types_to_solar



"""
And lastly, let's do all remaining cleanup
"""

del T_sun
del Constant

del solTeff
del ns
del def_unit
