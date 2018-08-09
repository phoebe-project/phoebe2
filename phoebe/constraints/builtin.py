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

def requiv_L1(q, syncpar, ecc, sma, incl_star, long_an_star, incl_orb, long_an_orb, compno, **kwargs):
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

def requiv_contact_L1(q, sma, compno, **kwargs):
    """
    for the contact case we can make the assumption of aligned, synchronous, and circular
    """
    return requiv_L1(q=1, syncpar=1, ecc=0, sma=sma, incl_star=0, long_an_star=0, incl_orb=0, long_an_orb=0, compno=compno, **kwargs)

def requiv_contact_L23(q, sma, compno, **kwargs):
    """
    for the contact case we can make the assumption of aligned, synchronous, and circular
    """
    logger.debug("requiv_contact_L23(q={}, sma={}, compno={})".format(q, sma, compno))
    crit_pots = libphoebe.roche_critical_potential(q, 1., 1.)
    crit_pot_L23 = max([crit_pots['L2'], crit_pots['L3']])

    q = _roche.q_for_component(q, compno)
    logger.debug("libphoebe.roche_contact_neck_min(q={}, d=1., crit_pot_L23={}, phi=pi/2)".format(q, crit_pot_L23))
    nekmin = libphoebe.roche_contact_neck_min(q, 1., crit_pot_L23, np.pi/2.)['xmin']
    logger.debug("libphoebe.roche_contact_partial_area_volume(nekmin={}, q={}, d=1, Omega={}, compno={})".format(nekmin, q, crit_pot_L23, compno))
    crit_vol_L23 = libphoebe.roche_contact_partial_area_volume(nekmin, q, 1., crit_pot_L23, compno)['lvolume']

    logger.debug("resulting vol: {}, requiv: {}".format(crit_vol_L23, (3./4*1./np.pi*crit_vol_L23)**1./3 * sma))

    return (3./4*1./np.pi*crit_vol_L23)**(1./3) * sma


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

def requiv_to_pot_contact(requiv, q, sma, compno=1):
    """
    :param requiv: user-provided equivalent radius
    :param q: mass ratio
    :param sma: semi-major axis (d = sma because we explicitly assume circular orbits for contacts)
    :param compno: 1 for primary, 2 for secondary
    :return: potential and fillout factor
    """
    def pot_to_volume(pot, q, d, vequiv, compno):
        """
        """
        nekmin = libphoebe.roche_contact_neck_min(q, d, pot, np.pi / 2.)['xmin']
        try:
            volume = libphoebe.roche_contact_partial_area_volume(nekmin, q, d, pot, compno)['lvolume']
        except TypeError:
            # then lobe boundaries probably failed, since this is only used for converging
            # let's return inf
            return np.inf
        else:
            return volume - vequiv

    logger.debug("requiv_to_pot_contact(requiv={}, q={}, sma={}, compno={})".format(requiv, q, sma, compno))

    # since the functions called here work with normalized r, we need to set d=D=sma=1.
    # or provide sma as a function parameter and normalize r here as requiv = requiv/sma
    requiv = requiv/sma
    vequiv = 4./3*np.pi*requiv**3
    d = 1.
    F = 1.
    pot_init = _roche.BinaryRoche([0., 0., requiv], q=q, D=d, F=F)
    try:
        logger.debug("newton(pot_to_volume, pot_init={}, q={}, d={}, vequiv={}, compno={})".format(pot_init, q, d, vequiv, compno))
        pot_final = newton(pot_to_volume, pot_init, args=(q, d, vequiv, compno))
        logger.debug("libphoebe.roche_critical_potential(q={}, d={}, F={})".format(q, d, F))
        crit_pots = libphoebe.roche_critical_potential(q, d, F)
        # ff = (pot_final - crit_pots['L1']) / (np.max((crit_pots['L2'], crit_pots['L3'])) - crit_pots['L1'])
        # ff is a np.float64
        return pot_final

    except:
        # replace this with actual check in the beginning or before function call
        raise ValueError('requiv probably out of bounds for contact envelope')

def pot_to_requiv_contact(pot, q, sma, compno=1):
    """
    """
    logger.debug("pot_to_requiv_contact(pot={}, q={}, sma={}, compno={})".format(pot, q, sma, compno))
    d = 1.
    try:
        logger.debug("libphoebe.roche_contact_neck_min(q={}, d={}, pot={}, pi/2)".format(q, d, pot))
        nekmin = libphoebe.roche_contact_neck_min(q, d, pot, np.pi / 2.)['xmin']
        logger.debug("libphoebe.roche_contact_partial_area_volume(nekmin={}, q={}, d={}, pot={}, compno={})".format(nekmin, q, d, pot, compno))
        volume_equiv = libphoebe.roche_contact_partial_area_volume(nekmin, q, d, pot, compno)['lvolume']
        # returns normalized vequiv, should multiply by sma back for requiv in SI
        return sma * (3./4 * 1./np.pi * volume_equiv)**(1./3)
    except:
        # replace this with actual check in the beginning or before function call
        raise ValueError('potential probably out of bounds for contact envelope')
