
import numpy as np

from phoebe import c, u
import libphoebe


def rotfreq_to_omega(rotfreq, scale=c.R_sun.si.value, solar_units=False):
    """
    TODO: add documentation

    NOTE: everything MUST be in SI units
    """
    if solar_units:
        omega = rotfreq / (2*np.pi) / np.sqrt(c.GM_sun.to(u.solRad**3/u.d**2).value/scale**3)
    else:
        # then SI units
        omega = rotfreq / (2*np.pi) / np.sqrt(c.GM_sun.value/scale**3)

    print "*** rotstar.rotfreq_to_omega", rotfreq, scale, solar_units, omega

    return omega


def rpole2potential(rpole, rotfreq, solar_units=False):
    """
    Transforms polar radius to surface potential
    """
    rpole_ = np.array([0., 0., rpole])
    omega = rotfreq_to_omega(rotfreq, solar_units=solar_units)
    print "*** rotstar.rpole2potential", rpole, rotfreq, solar_units, omega
    return libphoebe.rotstar_Omega(omega, rpole_)

def potential2rpole(pot, rotfreq, solar_units=False):
    """
    Transforms surface potential to polar radius
    """
    omega = rotfreq_to_omega(rotfreq, scale=1.0, solar_units=solar_units)
    print "*** rotstar.potential2rpole", pot, rotfreq, solar_units, omega
    rpole = libphoebe.rotstar_pole(omega, pot)
    if solar_units:
        return rpole
    else:
        return rpole*c.R_sun.si.value

