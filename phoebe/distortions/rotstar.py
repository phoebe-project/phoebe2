
import numpy as np

from phoebe import c, u
import libphoebe


def rotfreq_to_omega(rotfreq, scale=c.R_sun.si.value, solar_units=False):
    """
    Translate from rotation frequency `rotfreq` to `omega`.

    NOTE: everything MUST be in consistent units according to `solar_units` bool

    Arguments
    ----------
    * `rotfreq`
    * `scale` (float, optional, default=c.R_sun.si.value)
    * `solar_units` (bool, optional, default=False): whether `scale` is provided
        in solar units or SI.

    Returns
    ---------
    * float
    """
    if solar_units:
        omega = rotfreq / (2*np.pi) / np.sqrt(c.GM_sun.to(u.solRad**3/u.d**2).value/scale**3)
    else:
        # then SI units
        omega = rotfreq / (2*np.pi) / np.sqrt(c.GM_sun.value/scale**3)

    # print "*** rotstar.rotfreq_to_omega", rotfreq, scale, solar_units, omega

    return omega

#
# def rpole2potential(rpole, rotfreq, solar_units=False):
#     """
#     Transforms polar radius to surface potential
#     """
#     if not solar_units:
#         rpole = rpole/c.R_sun.si.value
#     rpole_ = np.array([0., 0., rpole])
#     omega = rotfreq_to_omega(rotfreq, solar_units=solar_units)
#     pot =  libphoebe.rotstar_Omega(omega, rpole_)
#     # print "*** rotstar.rpole2potential", rpole, rotfreq, solar_units, omega, pot
#     return pot
#
#
# def potential2rpole(pot, rotfreq, solar_units=False):
#     """
#     Transforms surface potential to polar radius
#     """
#     omega = rotfreq_to_omega(rotfreq, scale=1.0, solar_units=solar_units)
#     # print "*** rotstar.potential2rpole", pot, rotfreq, solar_units, omega
#     rpole = libphoebe.rotstar_pole(omega, pot)
#     if solar_units:
#         return rpole
#     else:
#         return rpole*c.R_sun.si.value
