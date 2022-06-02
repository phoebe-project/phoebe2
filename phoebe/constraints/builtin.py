import numpy as np
from phoebe.distortions import roche as _roche
from phoebe.backend import mesh as _mesh
from phoebe.dependencies import distl as _distl
import libphoebe
from scipy.optimize import newton, bisect

import logging
logger = logging.getLogger("BUILTIN")
logger.addHandler(logging.NullHandler())

# expose these at top-level so they're available to constraints
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, sqrt, log10

def phases_to_times(phase, period_choice, period, period_anom, dpdt_choice, dpdt, t0_choice, t0_supconj, t0_perpass, t0_ref):
    # print("*** builtin.phases_to_times period_choice: {}, t0_choice: {}".format(period_choice, t0_choice))
    if period_choice == 'period':
        period = period
    elif period_choice == 'period_anom':
        period = period_anom
    else:
        raise NotImplementedError()

    if t0_choice == 't0_supconj':
        t0 = t0_supconj
    elif t0_choice == 't0_perpass':
        t0 = t0_perpass
    elif t0_choice == 't0_ref':
        t0 = t0_ref
    else:
        raise NotImplementedError()

    if dpdt_choice == 'dpdt':
        dpdt = dpdt
    elif dpdt_choice == 'none':
        dpdt = 0.0
    else:
        raise NotImplementedError()

    # print("*** builtin.phases_to_times period: {}, t0: {}".format(period, t0))

    if isinstance(phase, list):
        phase = np.asarray(phase)

    if dpdt != 0:
        # NOTE: this seems to be incorrect and giving ridiculous answers
        time = t0 + 1./dpdt*(np.exp(dpdt*(phase))-period)
    else:
        time = t0 + (phase)*period

    return time

def times_to_phases(time, period_choice, period, period_anom, dpdt_choice, dpdt, t0_choice, t0_supconj, t0_perpass, t0_ref):
    # print("*** builtin.times_to_phases", period_choice, t0_choice)
    if period_choice == 'period':
        period = period
    elif period_choice == 'period_anom':
        period = period_anom
    else:
        raise NotImplementedError()

    if t0_choice == 't0_supconj':
        t0 = t0_supconj
    elif t0_choice == 't0_perpass':
        t0 = t0_perpass
    elif t0_choice == 't0_ref':
        t0 = t0_ref
    else:
        raise NotImplementedError()

    if dpdt_choice == 'dpdt':
        dpdt = dpdt
    elif dpdt_choice == 'none':
        dpdt = 0.0
    else:
        raise NotImplementedError()

    if isinstance(time, list):
        time = np.asarray(time)


    if dpdt != 0:
        phase = np.mod(1./dpdt * np.log(period + dpdt*(time-t0)), 1.0)
    else:
        phase = np.mod((time-t0)/period, 1.0)

    if isinstance(phase, float):
        if phase > 0.5:
            phase -= 1
    else:
        # then should be an array
        phase[phase > 0.5] -= 1

    return phase


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

def potential_contact_L1(q, **kwargs):
    """
    """
    crit_pots = libphoebe.roche_critical_potential(q, 1., 1.)
    return crit_pots['L1']

def potential_contact_L23(q, **kwargs):
    """
    """
    crit_pots = libphoebe.roche_critical_potential(q, 1., 1.)
    return max(crit_pots['L2'], crit_pots['L3'])

def requiv_contact_L1(q, sma, compno, **kwargs):
    """
    for the contact case we can make the assumption of aligned, synchronous, and circular
    """
    return requiv_L1(q=q, syncpar=1, ecc=0, sma=sma, incl_star=0, long_an_star=0, incl_orb=0, long_an_orb=0, compno=compno, **kwargs)

def requiv_contact_L23(q, sma, compno, **kwargs):
    """
    for the contact case we can make the assumption of aligned, synchronous, and circular
    """
    logger.debug("requiv_contact_L23(q={}, sma={}, compno={})".format(q, sma, compno))
    crit_pot_L23 = potential_contact_L23(q)

    logger.debug("libphoebe.roche_contact_neck_min(phi=pi/2, q={}, d=1., crit_pot_L23={})".format(q, crit_pot_L23))
    nekmin = libphoebe.roche_contact_neck_min(np.pi/2., q, 1., crit_pot_L23)['xmin']
    # we now have the critical potential and nekmin as if we were the primary star, so now we'll use compno=0 regardless
    logger.debug("libphoebe.roche_contact_partial_area_volume(nekmin={}, q={}, d=1, Omega={}, compno={})".format(nekmin, q, crit_pot_L23, compno-1))
    crit_vol_L23 = libphoebe.roche_contact_partial_area_volume(nekmin, q, 1., crit_pot_L23, compno-1, lvolume=True, ldvolume=False, larea=False)['lvolume']

    logger.debug("resulting vol: {}, requiv: {}".format(crit_vol_L23, (3./4*1./np.pi*crit_vol_L23)**(1./3) * sma))

    return (3./4*1./np.pi*crit_vol_L23)**(1./3) * sma

def pot_to_fillout_factor(q, pot, **kwargs):
    # calling libphoebe.roche_critical_potential is fairly cheap, so we'll
    # just parameterize this directly with q and make multiple calls to
    # critical potentials rather than having a nightmare of logics
    # between constraints.

    pot_L1 = potential_contact_L1(q)
    pot_L23 = potential_contact_L23(q)
    return (pot - pot_L1) / (pot_L23 - pot_L1)

def fillout_factor_to_pot(q, fillout_factor, **kwargs):
    pot_L1 = potential_contact_L1(q)
    pot_L23 = potential_contact_L23(q)
    return fillout_factor * (pot_L23 - pot_L1) + pot_L1


def esinw2per0(ecc, esinw, allow_nan=False):
    """
    """
    # print "*** constraints.builtin.esinw2per0", ecc, esinw

    if ecc==0.:
        # multiply to support arrays
        return 0.*esinw
    elif isinstance(ecc, np.ndarray):
        return np.array([esinw2per0(e, esinw) for e in ecc])
    else:
        per0 = np.arcsin(esinw/ecc)
        if not allow_nan and np.any(np.isnan(per0)):
            raise ValueError("esinw={} and ecc={} results in nan for per0".format(esinw, ecc))
        return per0

def ecosw2per0(ecc, ecosw, allow_nan=False):
    """
    """
    # print "*** constraints.builtin.ecosw2per0", ecc, ecosw

    if ecc==0.:
        # multiply to support arrays
        return 0.*ecosw
    elif isinstance(ecc, np.ndarray):
        return np.array([esinw2per0(e, esinw) for e in ecc])
    else:
        if not allow_nan and np.any(np.abs(ecosw/ecc) > 1):
            # raise the error in advance to avoid a runtime error
            raise ValueError("ecosw={} and ecc={} results in nan for per0".format(ecosw, ecc))
        per0 = np.arccos(ecosw/ecc)
        if not allow_nan and np.any(np.isnan(per0)):
            raise ValueError("ecosw={} and ecc={} results in nan for per0".format(ecosw, ecc))
        return per0

def esinw2ecc(esinw, per0, allow_nan=False):
    """
    """
    # print("*** constraints.builtin.esinw2ecc", esinw, per0)
    if not allow_nan and np.any(np.sin(per0) == 0):
        raise ValueError("esinw={} and per0={} results in nan for ecc".format(esinw, per0))
    ecc = esinw/np.sin(per0)
    if not allow_nan and np.any(np.isnan(ecc)):
        raise ValueError("esinw={} and per0={} results in nan for ecc".format(esinw, per0))
    return ecc

def ecosw2ecc(ecosw, per0, allow_nan=False):
    """
    """
    # print("*** constraints.builtin.ecosw2ecc", ecosw, per0)
    if not allow_nan and np.any(np.cos(per0) == 0):
        raise ValueError("ecosw={} and per0={} results in nan for ecc".format(ecosw, per0))
    ecc = ecosw/np.cos(per0)
    if not allow_nan and np.any(np.isnan(ecc)):
        raise ValueError("ecosw={} and per0={} results in nan for ecc".format(ecosw, per0))
    return ecc



def _delta_t_supconj_perpass(t, period, ecc, per0, dpdt, dperdt, t0):
    """
    time shift between superior conjuction and periastron passage
    """
    period = period + (t-t0)*dpdt
    per0 = (per0 + (t-t0)*dperdt) % (2*np.pi)
    ups_sc = np.pi/2-per0
    E_sc = 2*np.arctan( np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_sc/2) )
    M_sc = E_sc - ecc*np.sin(E_sc)
    return period*(M_sc/2./np.pi)

def t0_perpass_to_supconj(t0_perpass, period, ecc, per0, dpdt, dperdt, t0):
    """
    """
    return t0_perpass + _delta_t_supconj_perpass(t0_perpass, period, ecc, per0, dpdt, dperdt, t0)

def t0_supconj_to_perpass(t0_supconj, period, ecc, per0, dpdt, dperdt, t0):
    """
    """
    return t0_supconj - _delta_t_supconj_perpass(t0_supconj, period, ecc, per0, dpdt, dperdt, t0)

def _delta_t_supconj_ref(t, period, ecc, per0, dpdt, dperdt, t0):
    """
    time shift between superior conjunction and reference time (time at
    which the primary star crosses the barycenter along line-of-sight)
    """
    period = period + (t-t0)*dpdt
    per0 = (per0 + (t-t0)*dperdt) % (2*np.pi)
    ups_sc = np.pi/2-per0
    E_sc = 2*np.arctan( np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_sc/2) )
    M_sc = E_sc - ecc*np.sin(E_sc)
    return period*((M_sc+per0)/2./np.pi - 1./4)

def t0_ref_to_supconj(t0_ref, period, ecc, per0, dpdt, dperdt, t0):
    """
    """
    return t0_ref + _delta_t_supconj_ref(t0_ref, period, ecc, per0, dpdt, dperdt, t0)

def t0_supconj_to_ref(t0_supconj, period, ecc, per0, dpdt, dperdt, t0):
    """
    """
    return t0_supconj - _delta_t_supconj_ref(t0_supconj, period, ecc, per0, dpdt, dperdt, t0)

def requiv_to_pot_contact(requiv, q, sma, compno=1):
    """
    :param requiv: user-provided equivalent radius
    :param q: mass ratio
    :param sma: semi-major axis (d = sma because we explicitly assume circular orbits for contacts)
    :param compno: 1 for primary, 2 for secondary
    :return: potential and fillout factor
    """
    logger.debug("requiv_to_pot_contact(requiv={}, q={}, sma={}, compno={})".format(requiv, q, sma, compno))

    # since the functions called here work with normalized r, we need to set d=D=sma=1.
    # or provide sma as a function parameter and normalize r here as requiv = requiv/sma
    requiv = requiv/sma
    vequiv = 4./3*np.pi*requiv**3
    d = 1.
    F = 1.

    logger.debug("libphoebe.roche_contact_Omega_at_partial_vol(vol={}, phi=pi/2, q={}, d={}, choice={})".format(vequiv, q, d, compno-1))
    return libphoebe.roche_contact_Omega_at_partial_vol(vequiv, np.pi/2, q, d, choice=compno-1)

def pot_to_requiv_contact(pot, q, sma, compno=1):
    """
    """
    logger.debug("pot_to_requiv_contact(pot={}, q={}, sma={}, compno={})".format(pot, q, sma, compno))
    d = 1.
    F = 1.
    crit_pots = libphoebe.roche_critical_potential(q, d, F)
    crit_pot_L1 = crit_pots['L1']
    crit_pot_L23 = max(crit_pots['L2'], crit_pots['L3'])
    if pot > crit_pot_L1:
        raise ValueError("potential > L1 critical value")
    if pot < crit_pot_L23:
        raise ValueError("potential < L2/L3 critical value")

    try:
        logger.debug("libphoebe.roche_contact_neck_min(pi/2, q={}, d={}, pot={})".format(q, d, pot))
        nekmin = libphoebe.roche_contact_neck_min(np.pi / 2., q, d, pot)['xmin']
        logger.debug("libphoebe.roche_contact_partial_area_volume(nekmin={}, q={}, d={}, pot={}, compno={})".format(nekmin, q, d, pot, compno-1))
        volume_equiv = libphoebe.roche_contact_partial_area_volume(nekmin, q, d, pot, compno-1, lvolume=True, ldvolume=False, larea=False)['lvolume']
        # returns normalized vequiv, should multiply by sma back for requiv in SI
        return sma * (3./4 * 1./np.pi * volume_equiv)**(1./3)
    except:
        # replace this with actual check in the beginning or before function call
        raise ValueError('potential probably out of bounds for contact envelope')

def distl_from_json(json):
    return _distl.from_json(json)
