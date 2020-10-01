"""
"""

import numpy as np
from scipy.optimize import newton


from phoebe import u, c

try:
    import photodynam
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
except (ImportError, OSError):
    _can_reboundx = False
else:
    _can_reboundx = True

import logging
logger = logging.getLogger("DYNAMICS.NBODY")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

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

def dynamics_from_bundle(b, times, compute=None, return_roche_euler=False, use_kepcart=False, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    NOTE: you must either provide compute (the label) OR all relevant options
    as kwargs (ltte, stepsize, gr, integrator)

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

    b.run_delayed_constraints()

    hier = b.hierarchy

    computeps = b.get_compute(compute=compute, force_ps=True, **_skip_filter_checks)
    stepsize = computeps.get_value(qualifier='stepsize', stepsize=kwargs.get('stepsize', None), **_skip_filter_checks)
    ltte = computeps.get_value(qualifier='ltte', ltte=kwargs.get('ltte', None), **_skip_filter_checks)
    gr = computeps.get_value(qualifier='gr', gr=kwargs.get('gr', None), **_skip_filter_checks)
    integrator = computeps.get_value(qualifier='integrator', integrator=kwargs.get('integrator', None), **_skip_filter_checks)

    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits() if use_kepcart else [hier.get_parent_of(star) for star in starrefs]

    def mean_anom(t0, t0_perpass, period):
        # TODO: somehow make this into a constraint where t0 and mean anom
        # are both in the compute options if dynamic_method==nbody
        # (one is constrained from the other and the orbit.... nvm, this gets ugly)
        return 2 * np.pi * (t0 - t0_perpass) / period

    masses = [b.get_value(qualifier='mass', unit=u.solMass, component=component, context='component', **_skip_filter_checks) * c.G.to('AU3 / (Msun d2)').value for component in starrefs]  # GM
    smas = [b.get_value(qualifier='sma', unit=u.AU, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    eccs = [b.get_value(qualifier='ecc', component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    incls = [b.get_value(qualifier='incl', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    per0s = [b.get_value(qualifier='per0', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    long_ans = [b.get_value(qualifier='long_an', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    t0_perpasses = [b.get_value(qualifier='t0_perpass', unit=u.d, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    periods = [b.get_value(qualifier='period', unit=u.d, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]

    if return_roche_euler:
        # rotperiods are only needed to compute instantaneous syncpars
        rotperiods = [b.get_value(qualifier='period', unit=u.d, component=component, context='component', **_skip_filter_checks) for component in starrefs]
    else:
        rotperiods = None

    vgamma = b.get_value(qualifier='vgamma', context='system', unit=u.AU/u.d, **_skip_filter_checks)
    t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

    # mean_anoms = [mean_anom(t0, t0_perpass, period) for t0_perpass, period in zip(t0_perpasses, periods)]
    mean_anoms = [b.get_value(qualifier='mean_anom', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]

    return dynamics(times, masses, smas, eccs, incls, per0s, long_ans, \
                    mean_anoms, rotperiods, t0, vgamma, stepsize, ltte, gr,
                    integrator, use_kepcart=use_kepcart, return_roche_euler=return_roche_euler)


def dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms,
        rotperiods=None, t0=0.0, vgamma=0.0, stepsize=0.01, ltte=False, gr=False,
        integrator='ias15', return_roche_euler=False, use_kepcart=False):

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

    if use_kepcart:
        # print "*** bs.kep2cartesian", masses, smas, eccs, incls, per0s, long_ans, mean_anoms, t0
        init_conds = bs.kep2cartesian(_ensure_tuple(masses), _ensure_tuple(smas),
                                      _ensure_tuple(eccs), _ensure_tuple(incls),
                                      _ensure_tuple(per0s), _ensure_tuple(long_ans),
                                      _ensure_tuple(mean_anoms), t0)
        for i in range(len(masses)):
            mass = masses[i]
            x = init_conds['x'][i]
            y = init_conds['y'][i]
            z = init_conds['z'][i]
            vx = init_conds['vx'][i]
            vy = init_conds['vy'][i]
            vz = init_conds['vz'][i]
            # print "*** adding simulation particle for mass:", mass, x, y, z, vx, vy, vz
            sim.add(m=mass,
                    x=x, y=y, z=z,
                    vx=vx, vy=vy, vz=vz
                    )

            # just in case that was stupid and did things in Jacobian, let's for the positions and velocities
            p = sim.particles[-1]
            p.x = x
            p.y = y
            p.z = z
            p.vx = vx
            p.vy = vy
            p.vz = vz
    else:
        for mass, sma, ecc, incl, per0, long_an, mean_anom in zip(masses, smas, eccs, incls, per0s, long_ans, mean_anoms):
            N = sim.N
            # TODO: this assumes building from the inside out (Jacobi coordinates).
            # Make sure that will always happen.
            # This assumption will currently probably fail if inner_as_primary==False.
            # But for now even the inner_as_primary=True case is broken, probably
            # because of the way we define some elements WRT nesting orbits (sma)
            # and others WRT sky (incl, long_an, per0)

            if N==0:
                # print "*** adding simulation particle for mass:", mass
                sim.add(m=mass)
            else:
                # print "*** adding simulation particle for mass:", mass, sma, ecc, incl, long_an, per0, mean_anom
                sim.add(primary=None,
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
        # vgamma is in the direction of positive RV or negative vz
        particle.vz -= vgamma

    xs = [np.zeros(times.shape) for m in masses]
    ys = [np.zeros(times.shape) for m in masses]
    zs = [np.zeros(times.shape) for m in masses]
    vxs = [np.zeros(times.shape) for m in masses]
    vys = [np.zeros(times.shape) for m in masses]
    vzs = [np.zeros(times.shape) for m in masses]

    if return_roche_euler:
        # from instantaneous Keplerian dynamics for Roche meshing
        ds = [np.zeros(times.shape) for m in masses]
        Fs = [np.zeros(times.shape) for m in masses]

        ethetas = [np.zeros(times.shape) for m in masses]
        elongans = [np.zeros(times.shape) for m in masses]
        eincls = [np.zeros(times.shape) for m in masses]

    au_to_solrad = (1*u.AU).to(u.solRad).value

    for i,time in enumerate(times):

        sim.integrate(time, exact_finish_time=True)

        # if return_roche:
            # TODO: do we need to do this after handling LTTE???
            # orbits = sim.calculate_orbits()

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

            if return_roche_euler:
                # TODO: do we want the LTTE-adjust particles?

                # NOTE: this won't work for the first particle (as its the
                # primary in the simulation)
                if j==0:
                    particle = sim.particles[j+1]
                else:
                    particle = sim.particles[j]

                # get the orbit based on the primary component defined already
                # in the simulation.
                orbit = particle.calculate_orbit()

                # for instantaneous separation, we need the current separation
                # from the sibling component in units of its instantaneous (?) sma
                ds[j][i] = orbit.d / orbit.a
                # for syncpar (F), assume that the rotational FREQUENCY will
                # remain fixed - so we simply need to updated syncpar based
                # on the INSTANTANEOUS orbital PERIOD.
                Fs[j][i] = orbit.P / rotperiods[j]

                # TODO: need to add np.pi for secondary component
                ethetas[j][i] = orbit.f + orbit.omega # true anomaly + periastron

                elongans[j][i] = orbit.Omega

                eincls[j][i] = orbit.inc


    if return_roche_euler:
        # d, solRad, solRad/d, rad, unitless (sma), unitless, rad, rad, rad
        return times, xs, ys, zs, vxs, vys, vzs, ds, Fs, ethetas, elongans, eincls

    else:
        # d, solRad, solRad/d, rad
        return times, xs, ys, zs, vxs, vys, vzs


def dynamics_from_bundle_bs(b, times, compute=None, return_roche_euler=False, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    NOTE: you must either provide compute (the label) OR all relevant options
    as kwargs (ltte)

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

    stepsize = 0.01
    orbiterror = 1e-16
    computeps = b.get_compute(compute, check_visible=False, force_ps=True)
    ltte = computeps.get_value('ltte', check_visible=False, **kwargs)


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

    # mean_anoms = [mean_anom(t0, t0_perpass, period) for t0_perpass, period in zip(t0_perpasses, periods)]
    mean_anoms = [b.get_value('mean_anom', u.rad, component=component, context='component') for component in orbitrefs]

    return dynamics_bs(times, masses, smas, eccs, incls, per0s, long_ans, \
                    mean_anoms, t0, vgamma, stepsize, orbiterror, ltte,
                    return_roche_euler=return_roche_euler)




def dynamics_bs(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms,
        t0=0.0, vgamma=0.0, stepsize=0.01, orbiterror=1e-16, ltte=False,
        return_roche_euler=False):
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
        raise ImportError("photodynam is not installed (http://github.com/phoebe-project/photodynam)")

    times = _ensure_tuple(times)
    masses = _ensure_tuple(masses)
    smas = _ensure_tuple(smas)
    eccs = _ensure_tuple(eccs)
    incls = _ensure_tuple(incls)
    per0s = _ensure_tuple(per0s)
    long_ans = _ensure_tuple(long_ans)
    mean_anoms = _ensure_tuple(mean_anoms)

    # TODO: include vgamma!!!!
    # print "*** bs.do_dynamics", masses, smas, eccs, incls, per0s, long_ans, mean_anoms, t0
    d = photodynam.do_dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms, t0, stepsize, orbiterror, ltte, return_roche_euler)
    # d is in the format: {'t': (...), 'x': ( (1,2,3), (1,2,3), ...), 'y': ..., 'z': ...}

    nobjects = len(masses)
    ntimes = len(times)

    # TODO: need to return euler angles... if that even makes sense?? Or maybe we
    # need to make a new place in orbit??

    au_to_solrad = (1*u.AU).to(u.solRad).value

    ts = np.array(d['t'])
    xs = [(-1*np.array([d['x'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    ys = [(-1*np.array([d['y'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    zs = [(np.array([d['z'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    vxs = [(-1*np.array([d['vx'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    vys = [(-1*np.array([d['vy'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    vzs = [(np.array([d['vz'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]

    if return_roche_euler:
        # raise NotImplementedError("euler angles for BS not currently supported")
        # a (sma), e (ecc), in (incl), o (per0?), ln (long_an?), m (mean_anom?)
        ds = [(np.array([d['kepl_a'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # TODO: fix this
        Fs = [(np.array([1.0 for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # TODO: check to make sure this is the right angle
        # TODO: need to add np.pi for secondary component?
        # true anomaly + periastron
        ethetas = [(np.array([d['kepl_o'][ti][oi]+d['kepl_m'][ti][oi]+np.pi/2 for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # elongans = [(np.array([d['kepl_ln'][ti][oi]+long_ans[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        elongans = [(np.array([d['kepl_ln'][ti][oi]+long_ans[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # eincls = [(np.array([d['kepl_in'][ti][oi]+incls[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        eincls = [(np.array([d['kepl_in'][ti][oi]+np.pi-incls[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]



        # d, solRad, solRad/d, rad
        return ts, xs, ys, zs, vxs, vys, vzs, ds, Fs, ethetas, elongans, eincls

    else:

        # d, solRad, solRad/d
        return ts, xs, ys, zs, vxs, vys, vzs
