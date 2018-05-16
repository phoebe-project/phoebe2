import numpy as np
# from numpy import cos, sin, tan, pi, sqrt
from math import sqrt, sin, cos, acos, atan2, trunc, pi

# from phoebe import c
# import libphoebe

# from scipy.optimize import newton



def q_for_component(q, component=1):
    """
    """
    if component==1:
        return q
    elif component==2:
        return 1./q
    else:
        raise NotImplementedError

def pot_for_component(pot, q, component=1, reverse=False):
    """

    q for secondaries should already be flipped (via q_for_component)
    """
    if component==1:
        return pot
    elif component==2:
        if reverse:
            return pot/q + 0.5*(q-1)/q
        else:
            return q*pot - 0.5 * (q-1)
    else:
        raise NotImplementedError
#
# def rpole2potential(rpole, q, e, F, sma=1.0, component=1):
#     """
#     Transforms polar radius to surface potential at periastron
#     """
#     d = 1-e
#     q = q_for_component(q, component=component)
#     rpole_ = np.array([0, 0, rpole/sma])
#     pot = libphoebe.roche_Omega(q, F, d, rpole_)
#     if component == 1:
#         return pot
#     elif component == 2:
#         return pot/q + 0.5*(q-1)/q
#     else:
#         raise NotImplementedError
#
# def potential2rpole(pot, q, e, F, sma=1.0, component=1):
#     """
#     Transforms surface potential to polar radius at periastron
#     """
#     d = 1-e
#     q = q_for_component(q, component=component)
#     Phi = pot_for_component(pot, q, component=component)
#     return libphoebe.roche_pole(q, F, d, Phi) * sma
#
# def criticalL1(q, e, F, component=1):
#     """
#     determine the potential at periastron to fill the critical potential at L1
#     """
#     d = 1-e
#     q = q_for_component(q, component=component)
#
#     critical_pots = libphoebe.roche_critical_potential(q, F, d,
#                                                        L1=True,
#                                                        L2=False,
#                                                        L3=False)
#
#     return critical_pots['L1']


def BinaryRoche (r, D, q, F, Omega=0.0):
    r"""
    Computes a value of the asynchronous, eccentric Roche potential.

    If :envvar:`Omega` is passed, it computes the difference.

    The asynchronous, eccentric Roche potential is given by [Wilson1979]_

    .. math::

        \Omega = \frac{1}{\sqrt{x^2 + y^2 + z^2}} + q\left(\frac{1}{\sqrt{(x-D)^2+y^2+z^2}} - \frac{x}{D^2}\right) + \frac{1}{2}F^2(1+q)(x^2+y^2)

    @param r:      relative radius vector (3 components)
    @type r: 3-tuple
    @param D:      instantaneous separation
    @type D: float
    @param q:      mass ratio
    @type q: float
    @param F:      synchronicity parameter
    @type F: float
    @param Omega:  value of the potential
    @type Omega: float
    """
    return 1.0/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]) + q*(1.0/sqrt((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])-r[0]/D/D) + 0.5*F*F*(1+q)*(r[0]*r[0]+r[1]*r[1]) - Omega

def dBinaryRochedx (r, D, q, F):
    """
    Computes a derivative of the potential with respect to x.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[0]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*(r[0]-D)*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 -q/D/D + F*F*(1+q)*r[0]

def d2BinaryRochedx2(r, D, q, F):
    """
    Computes second derivative of the potential with respect to x.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return (2*r[0]*r[0]-r[1]*r[1]-r[2]*r[2])/(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**2.5 +\
          q*(2*(r[0]-D)*(r[0]-D)-r[1]*r[1]-r[2]*r[2])/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**2.5 +\
          F*F*(1+q)

def dBinaryRochedy (r, D, q, F):
    """
    Computes a derivative of the potential with respect to y.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[1]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[1]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 + F*F*(1+q)*r[1]

def  dBinaryRochedz(r, D, q, F):
    """
    Computes a derivative of the potential with respect to z.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[2]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[2]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5

def dBinaryRochedr (r, D, q, F):
    """
    Computes a derivative of the potential with respect to r.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """

    r2 = (r*r).sum()
    r1 = np.sqrt(r2)

    return -1./r2 - q*(r1-r[0]/r1*D)/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**1.5 - q*r[0]/r1/D/D + F*F*(1+q)*(1-r[2]*r[2]/r2)*r1

def Lag1(q):
    # L1
    dxL = 1.0
    L1 = 1e-3
    while abs(dxL) > 1e-6:
        dxL = - dBinaryRochedx([L1, 0.0, 0.0], 1., q, 1.) / d2BinaryRochedx2([L1, 0.0, 0.0], 1., q, 1.)
        L1 = L1 + dxL

    return L1
