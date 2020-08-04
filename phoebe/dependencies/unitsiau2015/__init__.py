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

from distutils.version import LooseVersion
if LooseVersion(astropyversion) < LooseVersion('2.0'):


    """
    see https://www.iau.org/static/resolutions/IAU2015_English.pdf

    Here we'll override astropy's constants to adhere to the IAU Resolution for nominal units
    """

    # TODO: find correct error estimate for this value of G
    G = Constant('G', "Gravitational constant", 6.67408e-11, 'm3 / (kg s2)', 0.00080e-11, 'NSFA 2011', system='si')
    c.G = G
    c.si.G = G

    GM_sun = Constant('GM_sun', "Solar G*M", 1.3271244e20,  'm3 / (s2)', None, "IAU 2015 Resolution B3", system='si')
    c.GM_sun = GM_sun
    c.si.GM_sun = GM_sun

    GM_earth = Constant('GM_earth', "Earth G*M", 3.986004e14, 'm3 / (s2)', None, "IAU 2015 Resolution B3", system='si')
    c.GM_earth = GM_earth
    c.si.GM_earth = GM_earth

    GM_jup = Constant('GM_jup', "Juptiter G*M", 1.2668653e17, 'm3 / (s2)', None, "IAU 2015 Resolution B3", system='si')
    c.GM_jup = GM_jup
    c.si.GM_jup = GM_jup

    M_sun = Constant('M_sun', "Solar mass", c.GM_sun.value/c.G.value, 'kg', None, "IAU 2015 Resolution B3 (derived from GM_sun and G)", system='si')
    c.M_sun = M_sun
    c.si.M_sun = M_sun

    R_sun = Constant('R_sun', "Solar radius", 6.957e8, 'm', None, "IAU 2015 Resolution B3", system='si')
    c.R_sun = R_sun
    c.si.R_sun = R_sun

    L_sun = Constant('L_sun', "Solar luminosity", 3.828e26, 'W', None, "IAU 2015 Resolution B3", system='si')
    c.L_sun = L_sun
    c.si.L_sun = L_sun

    """
    Now we need to redefine the units to use these constants
    """


    solMass = def_unit(['solMass', 'M_sun', 'Msun'], c.si.M_sun, namespace=ns,
             prefixes=True, doc="Solar mass (nominal)",
             format={'latex': r'{\mathcal{M}^{\rm N}_\odot}'})

    _register_unit(solMass)


    solRad = def_unit(['solRad', 'R_sun', 'Rsun'], c.si.R_sun, namespace=ns,
             doc="Solar radius (nominal)", prefixes=True,
             format={'latex': r'{\mathcal{R}^{\rm N}_\odot}'})

    _register_unit(solRad)

    solLum = def_unit(['solLum', 'L_sun', 'Lsun'], c.si.L_sun, namespace=ns,
             prefixes=True, doc="Solar luminance (nominal)",
             format={'latex': r'{\mathcal{L}^{\rm N}_\odot}'})

    _register_unit(solLum)

    """
    Let's cleanup the imports so these entries don't
    show at the top-level
    """

    del G
    del GM_sun
    del M_sun
    del R_sun
    del L_sun
    del solMass
    del solRad
    del solLum


else:
    # current defaults in astropy (as of 2.0) are codata2014 and iau2015
    pass


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
                            'mass': 'solMass',
                            'temperature': 'solTeff',
                            'power': 'solLum',
                            'time': 'd',
                            'speed': 'solRad/d',
                            'angle': 'rad',
                            'angular speed': 'rad/d',
                            'dimensionless': ''}

_physical_types_to_si = {'length': 'm',
                            'mass': 'kg',
                            'temperature': 'K',
                            'power': 'W',
                            'time': 's',
                            'speed': 'm/s',
                            'angle': 'rad',
                            'angular speed': 'rad/s',
                            'dimensionless': ''}

def _get_physical_type(object):

    if hasattr(object, 'physical_type'):
        unit = object
    elif isinstance(object, u.Quantity):
        unit = object.unit
    else:
        raise NotImplementedError("object {} with type={} not supported for _get_physical_type".format(object, type(object)))

    return unit.physical_type

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
del LooseVersion

del T_sun
del Constant

del solTeff
del ns
del def_unit
