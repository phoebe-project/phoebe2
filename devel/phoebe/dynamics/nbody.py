"""
"""

import numpy as np
from scipy.optimize import newton


from phoebe import u, c

# from phoebe.dynamics import burlishstoer as bs
try:
    import phoebe_burlishstoer as bs
except ImportError:
    _can_bs = False
else:
    _can_bs = True

try:
    import rebound
except ImportError:
    _can_rebound = False
else:
    _can_rebound = True

try:
    import reboundx
except ImportError:
    _can_reboundx = False
else:
    _can_reboundx = True

import logging
logger = logging.getLogger("DYNAMICS.NBODY")
logger.addHandler(logging.NullHandler())


def _ensure_tuple(item):
    """
    Simply ensure that the passed item is a tuple.  If it is not, then
    convert it if possible, or raise a NotImplementedError

    Args:
        item: the item that needs to become a tuple

    Returns:
        the item casted as a tuple

    Raises:
        NotImplementedError: if converting the given item to a tuple
            is not implemented.
    """
    if isinstance(item, tuple):
        return item
    elif isinstance(item, list):
        return tuple(item)
    elif isinstance(item, np.ndarray):
        return tuple(item.tolist())
    else:
        raise NotImplementedError

def dynamics_from_bundle(b, times, compute, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    Args:
        b: (Bundle) the bundle with a set hierarchy
        times: (list or array) times at which to run the dynamics
        stepsize: (float, optional) stepsize for the integration
            [default: 0.01]
        orbiterror: (float, optional) orbiterror for the integration
            [default: 1e-16]
        ltte: (bool, default False) whether to account for light travel time effects.
        gr: (bool, default False) whether to account for general relativity effects.

    Returns:
        t, xs, ys, zs, vxs, vys, vzs.  t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.

    """

    hier = b.hierarchy

    computeps = b.get_compute(compute, check_relevant=False, force_ps=True)
    stepsize = computeps.get_value('stepsize', check_relevant=False, **kwargs)
    ltte = computeps.get_value('ltte', check_relevant=False, **kwargs)
    gr = computeps.get_value('gr', check_relevant=False, **kwargs)
    integrator = computeps.get_value('integrator', check_relevant=False, **kwargs)

    starrefs = hier.get_stars()
    orbitrefs = [hier.get_parent_of(star) for star in starrefs]

    def mean_anom(t0, t0_perpass, period):
        # TODO: somehow make this into a constraint where t0 and mean anom
        # are both in the compute options if dynamic_method==nbody
        # (one is constrained from the other and the orbit.... nvm, this gets ugly)
        return 2 * np.pi * (t0 - t0_perpass) / period

    masses = [b.get_value('mass', u.solMass, component=component, context='component') * c.G.to('AU3 / (Msun d2)').value for component in starrefs]  # GM
    smas = [b.get_value('sma', u.AU, component=component, context='component') for component in orbitrefs]
    eccs = [b.get_value('ecc', component=component, context='component') for component in orbitrefs]
    incls = [b.get_value('incl', u.rad, component=component, context='component') for component in orbitrefs]
    per0s = [b.get_value('per0', u.rad, component=component, context='component') for component in orbitrefs]
    long_ans = [b.get_value('long_an', u.rad, component=component, context='component') for component in orbitrefs]
    t0_perpasses = [b.get_value('t0_perpass', u.d, component=component, context='component') for component in orbitrefs]
    periods = [b.get_value('period', u.d, component=component, context='component') for component in orbitrefs]

    vgamma = b.get_value('vgamma', context='system', unit=u.AU/u.d)
    t0 = b.get_value('t0', context='system', unit=u.d)

    mean_anoms = [mean_anom(t0, t0_perpass, period) for t0_perpass, period in zip(t0_perpasses, periods)]

    return dynamics(times, masses, smas, eccs, incls, per0s, long_ans, \
                    mean_anoms, t0, vgamma, stepsize, ltte, gr,
                    integrator)


def dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms,
        t0=0.0, vgamma=0.0, stepsize=0.01, ltte=False, gr=False,
        integrator='ias15'):

    if not _can_rebound:
        raise ImportError("rebound is not installed")

    if gr and not _can_reboundx:
        raise ImportError("reboundx is not installed (required for gr effects)")

    def particle_ltte(sim, particle_N, t_obs):
        c_AU_d = c.c.to(u.AU/u.d).value

        def residual(t):
            # print "*** ltte trying t:", t
            if sim.t != t:
                sim.integrate(t, exact_finish_time=True)
            ltte_dt = sim.particles[particle_N].z / c_AU_d
            t_barycenter = t - ltte_dt
            # print "***** ", t_barycenter-t_obs

            return t_barycenter - t_obs

        t_barycenter = newton(residual, t_obs)

        if sim.t != t_barycenter:
            sim.integrate(t_barycenter)

        return sim.particles[particle_N]


    times = np.asarray(times)
    # print "***", times.shape

    # TODO: implement LTTE
    # TODO: implement vgamma

    # TODO: check constants on units, since G is 1 this shouldn't matter?
    # You can check the units’ exact values and add Additional units in rebound/rebound/units.py. Units should be set before adding particles to the simulation (will give error otherwise).

    sim = rebound.Simulation()

    if gr:
        logger.info("enabling 'gr_full' in reboundx")
        rebx = reboundx.Extras(sim)
        # TODO: switch between different GR setups based on masses/hierarchy
        # http://reboundx.readthedocs.io/en/latest/effects.html#general-relativity
        params = rebx.add_gr_full()

    sim.integrator = integrator
    # NOTE: according to rebound docs: "stepsize will change for adaptive integrators such as IAS15"
    sim.dt = stepsize

    for mass, sma, ecc, incl, per0, long_an, mean_anom in zip(masses, smas, eccs, incls, per0s, long_ans, mean_anoms):
        N = sim.N
        sim.add(primary=None if N==0 else sim.particles[-1],
                m=mass,
                a=sma,
                e=ecc,
                inc=incl,
                Omega=long_an,
                omega=per0,
                M=mean_anom)

    sim.move_to_com()

    # Now handle vgamma by editing the initial vz on each particle
    for particle in sim.particles:
        particle.vz += vgamma

    xs = [np.zeros(times.shape) for m in masses]
    ys = [np.zeros(times.shape) for m in masses]
    zs = [np.zeros(times.shape) for m in masses]
    vxs = [np.zeros(times.shape) for m in masses]
    vys = [np.zeros(times.shape) for m in masses]
    vzs = [np.zeros(times.shape) for m in masses]

    au_to_solrad = (1*u.AU).to(u.solRad).value

    for i,time in enumerate(times):

        sim.integrate(time, exact_finish_time=True)

        for j in range(len(masses)):

            if ltte:
                # then we need to integrate to different times per object
                particle = particle_ltte(sim, j, time)
            else:
                # print "***", time, j, sim.N
                particle = sim.particles[j]

            # NOTE: x and y are flipped because of different coordinate system
            # conventions.  If we change our coordinate system to have x point
            # to the left, this will need to be updated to match as well.
            xs[j][i] = -1 * particle.x * au_to_solrad # solRad
            ys[j][i] = -1 * particle.y * au_to_solrad # solRad
            zs[j][i] = particle.z * au_to_solrad  # solRad
            vxs[j][i] = -1 * particle.vx * au_to_solrad # solRad/d
            vys[j][i] = -1 * particle.vy * au_to_solrad # solRad/d
            vzs[j][i] = particle.vz * au_to_solrad # solRad/d


    return times, xs, ys, zs, vxs, vys, vzs


def dynamics_from_bundle_bs(b, times, stepsize=0.01, orbiterror=1e-16, ltte=False):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    Args:
        b: (Bundle) the bundle with a set hierarchy
        times: (list or array) times at which to run the dynamics
        stepsize: (float, optional) stepsize for the integration
            [default: 0.01]
        orbiterror: (float, optional) orbiterror for the integration
            [default: 1e-16]
        ltte: (bool, default False) whether to account for light travel time effects.

    Returns:
        t, xs, ys, zs, vxs, vys, vzs.  t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.

    """

    hier = b.hierarchy

    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits()

    def mean_anom(t0, t0_perpass, period):
        # TODO: somehow make this into a constraint where t0 and mean anom
        # are both in the compute options if dynamic_method==nbody
        # (one is constrained from the other and the orbit.... nvm, this gets ugly)
        return 2 * np.pi * (t0 - t0_perpass) / period

    masses = [b.get_value('mass', u.solMass, component=component, context='component') * c.G.to('AU3 / (Msun d2)').value for component in starrefs]  # GM
    smas = [b.get_value('sma', u.AU, component=component, context='component') for component in orbitrefs]
    eccs = [b.get_value('ecc', component=component, context='component') for component in orbitrefs]
    incls = [b.get_value('incl', u.rad, component=component, context='component') for component in orbitrefs]
    per0s = [b.get_value('per0', u.rad, component=component, context='component') for component in orbitrefs]
    long_ans = [b.get_value('long_an', u.rad, component=component, context='component') for component in orbitrefs]
    t0_perpasses = [b.get_value('t0_perpass', u.d, component=component, context='component') for component in orbitrefs]
    periods = [b.get_value('period', u.d, component=component, context='component') for component in orbitrefs]

    vgamma = b.get_value('vgamma', context='system', unit=u.solRad/u.d)
    t0 = b.get_value('t0', context='system', unit=u.d)

    mean_anoms = [mean_anom(t0, t0_perpass, period) for t0_perpass, period in zip(t0_perpasses, periods)]

    return dynamics_bs(times, masses, smas, eccs, incls, per0s, long_ans, \
                    mean_anoms, t0, vgamma, stepsize, orbiterror, ltte)




def dynamics_bs(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms,
        t0=0.0, vgamma=0.0, stepsize=0.01, orbiterror=1e-16, ltte=False):
    """
    Burlisch-Stoer integration of orbits to give positions and velocities
    of any given number of stars in hierarchical orbits.  This code
    currently uses the NBody code in Josh Carter's photodynam code
    available here:

    [[TODO: include link]]

    If using the Nbody mode in PHOEBE, please cite him as well:

    [[TODO: include citation]]

    See :func:`dynamics_from_bundle` for a wrapper around this function
    which automatically handles passing everything in the correct order
    and in the correct units.

    For each iterable input, stars and orbits should be listed in order
    from primary -> secondary for each nested hierarchy level.  Each
    iterable for orbits should have length one less than those for each
    star (ie if 3 masses are provided, then 2 smas, eccs, etc need to
    be provided)

    Args:
        times: (iterable) times at which to compute positions and
            velocities for each star
        masses: (iterable) mass for each star in [solMass]
        smas: (iterable) semi-major axis for each orbit [AU]
        eccs: (iterable) eccentricities for each orbit
        incls: (iterable) inclinations for each orbit [rad]
        per0s: (iterable) longitudes of periastron for each orbit [rad]
        long_ans: (iterable) longitudes of the ascending node for each
            orbit [rad]
        mean_anoms: (iterable) mean anomalies for each orbit
        t0: (float) time at which to start the integrations
        stepsize: (float, optional) stepsize of the integrations
            [default: 0.01]
        orbiterror: (float, optional) orbiterror of the integrations
            [default: 1e-16]
        ltte: (bool, default False) whether to account for light travel time effects.

    Returns:
        t, xs, ys, zs, vxs, vys, vzs.  t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.

    """

    if not _can_bs:
        raise ImportError("phoebe_burlishstoer is not installed")

    times = _ensure_tuple(times)
    masses = _ensure_tuple(masses)
    smas = _ensure_tuple(smas)
    eccs = _ensure_tuple(eccs)
    incls = _ensure_tuple(incls)
    per0s = _ensure_tuple(per0s)
    long_ans = _ensure_tuple(long_ans)
    mean_anoms = _ensure_tuple(mean_anoms)

    # TODO: include vgamma!!!!
    d = bs.do_dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms, t0, stepsize, orbiterror, ltte)
    # d is in the format: {'t': (...), 'x': ( (1,2,3), (1,2,3), ...), 'y': ..., 'z': ...}

    nobjects = len(masses)
    ntimes = len(times)

    # TODO: need to return euler angles... if that even makes sense?? Or maybe we
    # need to make a new place in orbit??

    # define scale_factor to convert from au to solRad and return values in solRad
    scale_factor = c.au/c.R_sun

    return np.array(d['t']), \
        [(-1*np.array([d['x'][ti][oi] for ti in range(ntimes)])*scale_factor*u.solRad) for oi in range(nobjects)], \
        [(-1*np.array([d['y'][ti][oi] for ti in range(ntimes)])*scale_factor*u.solRad) for oi in range(nobjects)], \
        [(np.array([d['z'][ti][oi] for ti in range(ntimes)])*scale_factor*u.solRad) for oi in range(nobjects)], \
        [(-1*np.array([d['vx'][ti][oi] for ti in range(ntimes)])*scale_factor*u.solRad/u.d) for oi in range(nobjects)], \
        [(-1*np.array([d['vy'][ti][oi] for ti in range(ntimes)])*scale_factor*u.solRad/u.d) for oi in range(nobjects)], \
        [(np.array([d['vz'][ti][oi] for ti in range(ntimes)])*scale_factor*u.solRad/u.d) for oi in range(nobjects)]

