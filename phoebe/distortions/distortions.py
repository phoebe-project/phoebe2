import numpy as np
import libphoebe

def _compute_omega(syncpar, period, sma):
    period_star = period / syncpar
    freq_rot = 2*np.pi / period_star
    omega = rotstar.rotfreq_to_omega(freq_rot, scale=sma, solar_units=True)
    return omega

def requiv2pot_single(pot_type, requiv, rotfreq):
    """
    """
    raise NotImplementedError

    volume = 4./3 * np.pi * requiv**2
    if pot_type == 'roche':
        raise ValueError("pot_type='roche' not supported for single stars")
    elif pot_type == 'rotstar':

    elif pot_type == 'sphere':

    else:
        raise ValueError("pot_type='{}' not supported".format(pot_type))

def pot2requiv_single(pot_type, requiv, rotfreq):
    """
    """
    raise NotImplementedError

    if pot_type == 'roche':
        raise ValueError("pot_type='roche' not supported for single stars")
    elif pot_type == 'rotstar':

    elif pot_type == 'sphere':

    else:
        raise ValueError("pot_type='{}' not supported".format(pot_type))

def requiv2pot(pot_type, requiv, q, ecc, syncpar, period, sma, pitch, yaw, compno=1):
    """
    """
    volume = 4./3 * np.pi * requiv**2

    if pot_type == 'roche':
        # input volume needs to be in roche units
        volume /= sma**3
        pot = libphoebe.roche_misaligned_Omega_at_vol(volume,
                                                      q,
                                                      syncpar,
                                                      d,
                                                      spin_in_xyz_peri.astype(float))

        # TODO: need to flip this back if secondary???
        if compno==2:
            raise NotImplementedError

        return pot

    elif pot_type == 'rotstar':
        omega = _compute_omega(syncpar, period, sma)

        return libphoebe.rotstar_misaligned_Omega_at_vol(volume,
                                                         omega,
                                                         spin_in_xyz_peri.astype(float))

    elif pot_type == 'sphere':
        return libphoebe.sphere_Omega_at_vol(volume)
    else:
        raise ValueError("pot_type='{}' not supported".format(pot_type))

def pot2requiv(pot_type, pot, q, ecc, syncpar, period, sma, pitch, yaw, compno=1):
    """
    """

    # TODO: we don't have what we need for these... and this is a bit of a roundabout way to get what we need, but I'm not sure if we can do any quicker
    spin_in_uvw = mesh.spin_in_system(incl_star, long_an_star)
    spin_in_xyz_peri = mesh.spin_in_roche(spin_in_uvw, 0.0, elongan_peri, eincl_peri)


    if pot_type == 'roche':
        q = q_for_component(q, compno)
        pot = pot_for_component(pot, q, compno)

        d_peri = 1 - ecc

        av = libphoebe.roche_misaligned_area_volume(q,
                                                    syncpar,
                                                    d_peri,
                                                    spin_in_xyz_peri.astype(float),
                                                    pot,
                                                    choice=0,
                                                    larea=False,
                                                    lvolume=True)

        # volume is in roche units
       volume = av['lvolume']*sma**3

    elif pot_type == 'rotstar':
        omega = _compute_omega(syncpar, period, sma)

        av = libphoebe.rotstar_misaligned_area_volume(omega,
                                                      spin_in_xyz_peri.astype(float),
                                                      pot,
                                                      larea=False,
                                                      lvolume=True)

        # volume is in solar units because we sent omega in solar units?
        volume = av['lvolume']

    elif pot_type == 'sphere':
        av = libphoebe.sphere_area_volume(pot,
                                          larea=False,
                                          lvolume=True)

        volume = av['lvolume']

    else:
        raise ValueError("pot_type='{}' not supported".format(pot_type))

    return (3./4 * 1./np.pi * volume)**(1./3)
