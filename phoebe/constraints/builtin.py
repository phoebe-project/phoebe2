import numpy as np
from phoebe.distortions import roche as _roche
from phoebe.backend import mesh as _mesh
import libphoebe
from scipy.optimize import newton

import logging
logger = logging.getLogger("BUILTIN")
logger.addHandler(logging.NullHandler())

# expose these at top-level so they're available to constraints
from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt

def requiv_critical(q, syncpar, ecc, sma, incl_star, long_an_star, incl_orb, long_an_orb, compno, **kwargs):
    """
    """
    d = 1-ecc # compute at periastron
    true_anom = 0.0 # compute at periastron

    spin_xyz = _mesh.spin_in_system(incl_star, long_an_star)
    s = _mesh.spin_in_roche(spin_xyz, true_anom, long_an_orb, incl_orb)

    logger.debug("roche.roche_misaligned_critical_requiv(q={}, F={}, d={}, s={}, scale={})".format(q, syncpar, d, s, sma))
    q = _roche.q_for_component(q, compno)
    critical_requiv = _roche.roche_misaligned_critical_requiv(q, syncpar, d, s, sma)

    return critical_requiv


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

def pot_to_volume(pot, q, d, vequiv, choice):
    """ 
    """
    nekmin = libphoebe.roche_contact_neck_min(q, d, pot, np.pi / 2.)['xmin']
    volume = libphoebe.roche_contact_partial_area_volume(nekmin, q, d, pot, choice)['lvolume']
    return volume - vequiv

def requiv_to_pot_contact(requiv, q, choice=1):
    """
    :param requiv: user-provided equivalent radius
    :param q: mass ratio
    :param sma: semi-major axis (d = sma because we explicitly assume circular orbits for contacts)
    :param choice: 1 for primary, 2 for secondary
    :return: potential and fillout factor
    """

    # since the functions called here work with normalized r, we need to set d=D=sma=1.
    # or provide sma as a function parameter and normalize r here as requiv = requiv/sma

    sma = 1.
    vequiv = 4.*np.pi*requiv**3/3.
    pot_init = _roche.BinaryRoche([0., 0., requiv], q=q, D=sma, F=1.)

    try:
        pot_final = newton(pot_to_volume, pot_init, args=(q, sma, vequiv, choice))
        crit_pots = libphoebe.roche_critical_potential(q=q, d=sma, F=1.)
        ff = (pot_final - crit_pots['L1']) / (np.max((crit_pots['L2'], crit_pots['L3'])) - crit_pots['L1'])
        return pot_final, ff

    except:
        # replace this with actual check in the beginning or before function call
        raise ValueError('requiv probably out of bounds for contact envelope')

def pot_to_requiv_contact(pot, q, choice=1):
    """
    """
    sma = 1.
    try:
        nekmin = libphoebe.roche_contact_neck_min(q, pot, np.pi / 2.)['xmin']
        volume_equiv = libphoebe.roche_contact_partial_area_volume(nekmin, q, sma, pot, choice)['lvolume']
        # returns normalized requiv
        return (3 * volume_equiv / (4. * np.pi)) ** (1. / 3)
    except:
        # replace this with actual check in the beginning or before function call
        raise ValueError('potential probably out of bounds for contact envelope')



