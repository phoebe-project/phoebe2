from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import these so they'll be available as phoebe.constants.c (but then imported as phoebe2.c)
import astropy.constants as c
import astropy.units as u


from astropy.constants import Constant
from astropy.units import def_unit


_use_resolution = True
# NOTE: changing this to True will break the dynamics nosetest (until n-body is fixed to also handle IAU constants)

"""
see https://www.iau.org/static/resolutions/IAU2015_English.pdf

Here we'll override astropy's constants to adhere to the IAU Resolution for nominal units
"""

# TODO: avogrado to 6.0221409e23 (to match legacy)

if _use_resolution:
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

    T_sun = Constant('T_sun', "Solar effective temperature", 5772, 'K', None, "IAU 2015 Resolution B3", system='si')
    c.T_sun = T_sun
    c.si.T_sun = T_sun

"""
Now we need to redefine the units to use these constants
"""


def _register_unit(unit):
    """
    register this unit in the phoebe.u namespace
    """
    for name in unit._names:
        setattr(u, name, unit)
        for reg in u.core._unit_registries:
            reg._registry[name] = unit


if _use_resolution:
    # TODO: pass namespace for units package so that prefixes are loaded (otherwise calling ksolMass will call from the old constants)
    ns = None

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


    solTeff = def_unit(['solTeff', 'T_sun', 'Tsun'], c.si.T_sun, namespace=ns,
            prefixes=True, doc="Solar effective temperature (nominal)",
            format={'latex': r'{\mathcal{T}^{\rm N}_\odot}'})

    _register_unit(solTeff)


"""
And lastly, let's cleanup the imports so these entries don't
show at the top-level of phoebe
"""
if _use_resolution:
    del G
    del GM_sun
    del M_sun
    del R_sun
    del L_sun
    del T_sun
    del solMass
    del solRad
    del solLum
    del solTeff
    del ns

del Constant
del def_unit
del _use_resolution
