import numpy as np
# from phoebe.distortions.roche import potential2rpole as _potential2rpole
# from phoebe.distortions.roche import rpole2potential as _rpole2potential
from phoebe.distortions import roche as _roche
from phoebe.distortions import rotstar as _rotstar

def rochepotential2rpole(*args, **kwargs):
    """
    """
    return _roche.potential2rpole(*args, **kwargs)

def rocherpole2potential(*args, **kwargs):
    """
    """
    return _roche.rpole2potential(*args, **kwargs)

def rotstarpotential2rpole(*args, **kwargs):
    """
    """
    return _rotstar.potential2rpole(*args, **kwargs)

def rotstarrpole2potential(*args, **kwargs):
    """
    """
    return _rotstar.rpole2potential(*args, **kwargs)

def rochecriticalL12potential(*args, **kwargs):
    """
    """
    return _roche.criticalL1(*args, **kwargs)

def rochecriticalL12rpole(q, e, F, sma, compno, **kwargs):
    """
    """
    critical_pot =  _roche.criticalL1(q, e, F, compno, **kwargs)
    return _roche.potential2rpole(critical_pot, q, e, F, sma, compno)


def esinw2per0(ecc, esinw):
    """
    """
    # print "*** constraints.builtin.esinw2per0", ecc, esinw

    if ecc==0.:
        return 0.
    else:
        per0 = np.arcsin(esinw/ecc)
        if np.isnan(per0):
            raise ValueError("esinw={} and ecc={} results in nan for per0, please REVERT value for esinw".format(esinw, ecc))
        return per0

def ecosw2per0(ecc, ecosw):
    """
    """
    # print "*** constraints.builtin.ecosw2per0", ecc, ecosw

    if ecc==0.:
        return 0.
    else:
        per0 = np.arccos(ecosw/ecc)
        if np.isnan(per0):
            raise ValueError("ecosw={} and ecc={} results in nan for per0, please REVERT value for ecosw".format(ecosw, ecc))
        return per0

def _delta_t_supconj_perpass(period, ecc, per0):
    """
    time shift between superior conjuction and periastron passage
    """
    ups_sc = np.pi/2-per0
    E_sc = 2*np.arctan( np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_sc/2) )
    M_sc = E_sc - ecc*np.sin(E_sc)
    return period*(M_sc/2./np.pi)

def t0_perpass_to_supconj(t0_perpass, period, ecc, per0):
    """
    """
    return t0_perpass + _delta_t_supconj_perpass(period, ecc, per0)

def t0_supconj_to_perpass(t0_supconj, period, ecc, per0):
    """
    """
    return t0_supconj - _delta_t_supconj_perpass(period, ecc, per0)

def _delta_t_supconj_ref(period, ecc, per0):
    """
    time shift between superior conjunction and reference time (time at
    which the primary star crosses the barycenter along line-of-sight)
    """
    ups_sc = np.pi/2-per0
    E_sc = 2*np.arctan( np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_sc/2) )
    M_sc = E_sc - ecc*np.sin(E_sc)
    return period*((M_sc+per0)/2./np.pi - 1./4)

def t0_ref_to_supconj(t0_ref, period, ecc, per0):
    """
    """
    return t0_ref + _delta_t_supconj_ref(period, ecc, per0)

def t0_supconj_to_ref(t0_supconj, period, ecc, per0):
    """
    """
    return t0_supconj - _delta_t_supconj_ref(period, ecc, per0)
