import numpy as np
# from numpy import cos, sin, tan, pi, sqrt

from phoebe import c
import libphoebe

from scipy.optimize import newton



def q_for_component(q, component=1):
    """
    """
    if component==1:
        return q
    elif component==2:
        return 1./q
    else:
        raise NotImplementedError

def pot_for_component(pot, q, component=1):
    """

    q for secondaries should already be flipped (via q_for_component)
    """
    if component==1:
        return pot
    elif component==2:
        return q*pot - 0.5 * (q-1)
    else:
        raise NotImplementedError

def rpole2potential(rpole, q, e, F, sma=1.0, component=1):
    """
    Transforms polar radius to surface potential at periastron
    """
    d = 1-e
    q = q_for_component(q, component=component)
    rpole_ = np.array([0, 0, rpole/sma])
    pot = libphoebe.roche_Omega(q, F, d, rpole_)
    if component == 1:
        return pot
    elif component == 2:
        return pot/q + 0.5*(q-1)/q
    else:
        raise NotImplementedError

def potential2rpole(pot, q, e, F, sma=1.0, component=1):
    """
    Transforms surface potential to polar radius at periastron
    """
    d = 1-e
    q = q_for_component(q, component=component)
    Phi = pot_for_component(pot, q, component=component)
    return libphoebe.roche_pole(q, F, d, Phi) * sma

def criticalL1(q, e, F, component=1):
    """
    determine the potential at periastron to fill the critical potential at L1
    """
    d = 1-e
    q = q_for_component(q, component=component)

    critical_pots = libphoebe.roche_critical_potential(q, F, d,
                                                       L1=True,
                                                       L2=False,
                                                       L3=False)

    return critical_pots['L1']
