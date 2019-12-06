import numpy as np
# from numpy import cos, sin, tan, pi, sqrt
from math import sqrt, sin, cos, acos, atan2, trunc, pi

# from phoebe import c
import libphoebe

# from scipy.optimize import newton

import logging
logger = logging.getLogger("ROCHE")
logger.addHandler(logging.NullHandler())

def q_for_component(q, component=1):
    """
    Flip the mass-ratio `q` if secondary component.

    Arguments
    -------------
    * `q` (float): the mass-ratio of the system/orbit (defined as m2/m1)
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    -----------
    * (float) `q` if `component` is 1, or `1/q` if component is 2

    Raises
    -----------
    * NotImplementedError: if `component` is neither 1 or 2.
    """
    if component==1:
        return q
    elif component==2:
        return 1./q
    else:
        raise NotImplementedError

def pot_for_component(pot, q, component=1, reverse=False):
    """
    Handle converting the potential into the star's frame of reference.

    Note: `q` should already be in the frame of the requested star.  If necessary,
    pass the output from <phoebe.distortions.roche.q_for_component>.

    If `component` is 1:
        * `pot` is returned as is

    If `component` is 2:
        * if `reverse` is True: `pot/q + 0.5 * (q-1)/q`
        * if `reverse` is False: `q*pot - 0.5 * (q-1)`

    Arguments
    ------------
    * `pot` (float): the equipotential of the star as defined as if it were the
        primary component.
    * `q` (float): the mass-ratio in the frame of `component`.  If necessary,
        call <phoebe.distortions.roche.q_for_component> first.
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.
    * `reverse` (bool, optional, default=False): whether to do the reverse operation.
        If True, then `pot` is assumed to be in the reference frame of `component`
        and is returned as if in the reference from of the primary component.

    Returns
    ----------
    * (float): the converted equipotential.

    Raises
    ------------
    * NotImplementedError: if `component` is neither 1 or 2.)
    """
    # currently only used by legacy wrapper: consider moving/removing
    if component==1:
        return pot
    elif component==2:
        if reverse:
            return pot/q + 0.5*(q-1)/q
        else:
            return q*pot - 0.5 * (q-1)
    else:
        raise NotImplementedError

def roche_misaligned_critical_requiv(q, F, d, s, scale=1.0):
    """
    Compute the critical value for the equivalent radius for the misaligned
    roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    Note: `q` should already be in the frame of the requested star.  If necessary,
    pass the output from <phoebe.distortions.roche.q_for_component>.

    Note: `s` should be in roche coordinates and at the requested time at which
    the critical radius is being requested.  For the aligned case, `s` should
    be [0, 0, 1].

    Arguments
    ----------
    * `q` (float): the mass-ratio in the frame of `component`.  If necessary,
        call <phoebe.distortions.roche.q_for_component> first.
    * `F` (float): roche synchronicity parameter.
    * `d` (float): instantaneous separation between the two components in the
        orbit.
    * `s` (array of length 3): rotation spin vector, in roche coordinates, at
        the requested time.
    * `scale` (float, optional, default=1.0): value by which to scale the output
        (to convert from roche to real units, for example)

    Returns
    -----------
    * (float): the critical equivalent radius.

    """
    volume_critical = libphoebe.roche_misaligned_critical_volume(q, F, d, s)
    logger.debug("libphoebe.roche_misaligned_critical_volume(q={}, F={}, d={}, s={}) => {}".format(q, F, d, s, volume_critical))

    requiv_critical = scale * (volume_critical * 3./4 * 1./np.pi)**(1./3)
    # logger.debug("roche.roche_misaligned_critical_requiv = {}".format(requiv_critical))

    return requiv_critical

def requiv_to_pot(requiv, sma, q, F, d, s=np.array([0.,0.,1.]), component=1):
    """
    Convert from equivalent radius to equipotential for the misaligned roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    Note: `q` should already be in the frame of the requested star.  If necessary,
    pass the output from <phoebe.distortions.roche.q_for_component>.

    Note: if `component` is not 1, the equipotential will be converted into the
    primary frame by calling <phoebe.distortions.roche.pot_for_component> with
    `reverse=True`.  To avoid this behavior, pass `component=1`.

    See also:
    * <phoebe.distortions.roche.pot_to_requiv>

    Arguments
    -----------
    * `requiv` (float): the equivalent radius
    * `sma` (float): the semi-major axis of the orbit
    * `q` (float): the mass-ratio in the frame of `component`.  If necessary,
        call <phoebe.distortions.roche.q_for_component> first.
    * `F` (float): the synchronicity parameter.
    * `d` (float): instantaneous separation between the two components in the
        orbit.
    * `s` (array of length 3): rotation spin vector, in roche coordinates, at
        the requested time.
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    ----------
    * (float): the equipotential in the primary frame.
    """
    logger.debug("roche.requiv_to_pot(requiv={}, sma={}, q={}, F={}, d={}, s={}, component={})".format(requiv, sma, q, F, d, s, component))
    volume = 4./3 * np.pi * requiv**3 / sma**3
    logger.debug("roche_misaligned_Omega_at_vol(volume={}, q={}, F={}, d={}, s={})".format(volume, q, F, d, s))
    Phi = libphoebe.roche_misaligned_Omega_at_vol(volume,
                                                  q, F, d, s)

    return pot_for_component(Phi, q, component=component, reverse=True)

def pot_to_requiv(pot, sma, q, F, d, s=np.array([0.,0.,1.]), component=1):
    """
    Convert from equipotential to equivalent radius for the misaligned roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    Note: if `component` is not 1, the equipotential `pot` will be first converted into the
    star's frame by calling <phoebe.distortions.roche.pot_for_component> and
    the mass-ratio `q` will be converted to the star's frame via
    <phoebe.distortions.roche.q_for_component>.  To avoid this behavior, pass
    `component=1`.

    See also:
    * <phoebe.distortions.roche.requiv_to_pot>

    Arguments
    -----------
    * `pot` (float): the equipotential of the star as defined in the primary frame.
    * `sma` (float): the semi-major axis of the orbit.  Pass as `1` to
        return `requiv` in roche coordinates.
    * `q` (float): the mass-ratio in the frame of the primary star (m2/m1).
        Do not call <phoebe.distortions.roche.q_for_component> first as this
        is called internally if `component=2`.
    * `F` (float): the synchronicity parameter.
    * `d` (float): instantaneous separation between the two components in the
        orbit.
    * `s` (array of length 3): rotation spin vector, in roche coordinates, at
        the requested time.
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    ----------
    * (float): the equivalent radius.
    """
    logger.debug("roche.pot_to_requiv(pot={}, sma={}, q={}, F={}, d={}, s={}, component={})".format(pot, sma, q, F, d, s, component))
    q = q_for_component(q, component)

    pot = pot_for_component(pot, q, component)
    logger.debug("libphoebe.roche_area_volume(q={}, F={}, d={}, Omega={})".format(q, F, d, pot))
    volume = libphoebe.roche_misaligned_area_volume(q, F, d, s, pot,
                                                    choice=0,
                                                    lvolume=True,
                                                    larea=False)['lvolume']

    # convert from roche units to scaled (solar) units
    volume *= sma**3

    # now convert from volume (in solar units) to requiv
    requiv = (volume * 3./4 * 1./np.pi)**(1./3)

    return requiv

def rpole_to_pot_aligned(rpole, sma, q, F, d, component=1):
    """
    Transforms from polar radius to equipotential under the aligned roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    Note: if `component` is not 1, the mass-ratio `q` will be converted to the star's frame via
    <phoebe.distortions.roche.q_for_component> and the resulting equipotential
    will be converted back to the primary frame via
    <phoebe.distortions.roche.pot_for_component> with `reverse=True`.
    To avoid this behavior, pass `component=1`.

    See also:
    * <phoebe.distortions.roche.pot_to_rpole_aligned>

    Arguments
    -----------
    * `rpole` (float): polar radius
    * `sma` (float): semi-major axis of the orbit
    * `q` (float): the mass-ratio in the frame of the primary star (m2/m1).
        Do not call <phoebe.distortions.roche.q_for_component> first as this
        is called internally if `component=2`.
    * `F` (float): synchronicity parameter
    * `d` (float): instantaneous unitless distance between the two components
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    ---------
    * (float) the aligned roche equipotential
    """
    q = q_for_component(q, component=component)
    rpole_ = np.array([0, 0, rpole/sma])
    logger.debug("libphoebe.roche_Omega(q={}, F={}, d={}, rpole={})".format(q, F, d, rpole_))
    pot = libphoebe.roche_Omega(q, F, d, rpole_)
    return pot_for_component(pot, component, reverse=True)

def pot_to_rpole_aligned(pot, sma, q, F, d, component=1):
    """
    Transforms from equipotential to polar radius under the aligned roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    Note: if `component` is not 1, the mass-ratio `q` will be converted to the star's frame via
    <phoebe.distortions.roche.q_for_component> and the equipotential `pot`
    will be converted to the star's frame via
    <phoebe.distortions.roche.pot_for_component>.
    To avoid this behavior, pass `component=1`.

    See also:
    * <phoebe.distortions.roche.rpole_to_pot_aligned>

    Arguments
    -----------
    * `pot` (float): the aligned roche equipotential
    * `sma` (float): semi-major axis of the orbit.  To return polar radius
        in roche units, use `sma=1`.
    * `q` (float): the mass-ratio in the frame of the primary star (m2/m1).
        Do not call <phoebe.distortions.roche.q_for_component> first as this
        is called internally if `component=2`.
    * `F` (float): synchronicity parameter
    * `d` (float): instantaneous unitless distance between the two components
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    ---------
    * (float) the aligned roche equipotential
    """
    q = q_for_component(q, component=component)
    Phi = pot_for_component(pot, q, component=component)
    logger.debug("libphobe.roche_pole(q={}, F={}, d={}, Omega={})".format(q, F, d, pot))
    return libphoebe.roche_pole(q, F, d, pot) * sma

def requiv_to_rpole_aligned(requiv, sma, q, F, d, component=1):
    """
    Transforms from equivalent radius to polar radius under the aligned roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    This is just a helper function around <phoebe.distortions.roche.requiv_to_pot>
    and <phoebe.distortions.roche.pot_to_rpole_aligned>.

    See also:
    * <phoebe.distortions.roche.rpole_to_requiv_aligned>

    Arguments
    -----------
    * `requiv` (float): the equivalent radius
    * `sma` (float): semi-major axis of the orbit.
    * `q` (float): the mass-ratio in the frame of the primary star (m2/m1).
        Do not call <phoebe.distortions.roche.q_for_component> first as this
        is called internally if `component=2`.
    * `F` (float): synchronicity parameter
    * `d` (float): instantaneous unitless distance between the two components
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    ---------
    * (float) the polar radius
    """
    pot = requiv_to_pot(requiv, sma, q, F, d, component=component)
    return pot_to_rpole_aligned(pot, sma, q, F, d, component=component)

def rpole_to_requiv_aligned(rpole, sma, q, F, d, component=1):
    """
    Transforms from equivalent radius to polar radius under the aligned roche case.

    Note: all inputs and output are floats, so it is important to pass values
    with consistent units.

    This is just a helper function around <phoebe.distortions.roche.rpole_to_pot_aligned>
    and <phoebe.distortions.roche.pot_to_requiv>.

    See also:
    * <phoebe.distortions.roche.requiv_to_rpole_aligned>

    Arguments
    -----------
    * `rpole` (float): the polar radius
    * `sma` (float): semi-major axis of the orbit.
    * `q` (float): the mass-ratio in the frame of the primary star (m2/m1).
        Do not call <phoebe.distortions.roche.q_for_component> first as this
        is called internally if `component=2`.
    * `F` (float): synchronicity parameter
    * `d` (float): instantaneous unitless distance between the two components
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component.

    Returns
    ---------
    * (float) the equivalent radius
    """
    pot = rpole_to_pot_aligned(rpole, sma, q, F, d, component=component)
    return pot_to_requiv(pot, sma, q, F, d, component=component)



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
