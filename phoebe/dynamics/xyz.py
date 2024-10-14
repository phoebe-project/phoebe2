"""
"""

import numpy as np
from scipy.optimize import newton

from phoebe import u, c
from phoebe import conf
from phoebe.dynamics import geometry

try:
    import rebound
except ImportError:
    _is_rebound = False
else:
    _is_rebound = True

try:
    import reboundx
except ImportError:
    _is_reboundx = False
else:
    _is_reboundx = True

import logging
logger = logging.getLogger("DYNAMICS.NBODY")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

_geometry = None

def dynamics_from_bundle(b, times, compute=None, return_roche_euler=False, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    NOTE: you must either provide compute (a label) OR all relevant options
    as kwargs (integrator, stepsize, epsilon, ltte, gr, geometry).

    Args:
        b: (Bundle) the bundle with a set hierarchy
        times: (list or array) times at which to run the dynamics
        return_roche_euler: (bool) whether to return Roche parameters and Euler angles
        geometry: (string) which geometry to use ('hierarchical', 'twopairs')

    Returns:
        t: (numpy array) all computed times
        xs, ys, zs: (numpy arrays) time-dependent Cartesian positions [solRad]
        vxs, vys, vzs: (numpy arrays) time-dependent Cartesian velocities [solRad/day]
        ds, Fs: (numpy arrays) Roche parameters [1]
        theta, longan, incl: (numpy arrays) Euler angles [rad]

    """
    global _geometry

    b.run_delayed_constraints()

    hier = b.hierarchy

    computeps = b.get_compute(compute=compute, force_ps=True, **_skip_filter_checks)
    stepsize = computeps.get_value(qualifier='stepsize', stepsize=kwargs.get('stepsize', None), **_skip_filter_checks)
    ltte = computeps.get_value(qualifier='ltte', ltte=kwargs.get('ltte', None), **_skip_filter_checks)
    gr = computeps.get_value(qualifier='gr', gr=kwargs.get('gr', None), **_skip_filter_checks)
    integrator = computeps.get_value(qualifier='integrator', integrator=kwargs.get('integrator', None), **_skip_filter_checks)
    epsilon = computeps.get_value(qualifier='epsilon', epsilon=kwargs.get('epsilon', None), **_skip_filter_checks)
    _geometry = computeps.get_value(qualifier='geometry', geometry=kwargs.get('geometry', None), **_skip_filter_checks)

    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits()

    masses = [b.get_value(qualifier='mass', unit=u.solMass, component=component, context='component', **_skip_filter_checks) * c.G.to('AU3 / (Msun d2)').value for component in starrefs]  # GM
    smas = [b.get_value(qualifier='sma', unit=u.AU, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    eccs = [b.get_value(qualifier='ecc', component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    incls = [b.get_value(qualifier='incl', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    per0s = [b.get_value(qualifier='per0', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    long_ans = [b.get_value(qualifier='long_an', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    mean_anoms = [b.get_value(qualifier='mean_anom', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]

    if return_roche_euler:
        rotperiods = [b.get_value(qualifier='period', unit=u.d, component=component, context='component', **_skip_filter_checks) for component in starrefs]
    else:
        rotperiods = None

    vgamma = b.get_value(qualifier='vgamma', context='system', unit=u.AU/u.d, **_skip_filter_checks)
    t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

    j2 = computeps.get_value(qualifier='j2', j2=kwargs.get('j2', None), **_skip_filter_checks)
    j2s = [b.get_value(qualifier='j2', unit=u.dimensionless_unscaled, component=component, context='component', **_skip_filter_checks) for component in starrefs]
    requivs = [b.get_value(qualifier='requiv', unit=u.AU, component=component, context='component', **_skip_filter_checks) for component in starrefs]

    nbod = len(masses)
    elmts = []
    for j in range(0, nbod-1):
        elmts.append([smas[j], eccs[j], incls[j], long_ans[j], per0s[j], mean_anoms[j]])

    xi, yi, zi, vxi, vyi, vzi = geometry.geometry(masses, elmts, geometry=_geometry)

    return dynamics(times, masses, xi, yi, zi, vxi, vyi, vzi, \
               rotperiods=rotperiods, \
               t0=t0, \
               vgamma=vgamma, \
               integrator=integrator, \
               stepsize=stepsize, \
               epsilon=epsilon, \
               ltte=ltte, \
               gr=gr, \
               j2=j2, \
               j2s=j2s, \
               requivs=requivs, \
               return_roche_euler=return_roche_euler \
               )


def dynamics(times, masses, xi, yi, zi, vxi, vyi, vzi, \
        rotperiods=None, \
        t0=0.0, \
        vgamma=0.0, \
        integrator='ias15', \
        epsilon=1.0e-9, \
        stepsize=0.01, \
        ltte=False, \
        gr=False, \
        j2=False, \
        j2s=None, \
        requivs=None, \
        return_roche_euler=False \
        ):

    """
    Computes N-body dynamics for initial conditions in Cartesian coordinates ("xyz").

    See :func:`dynamics_from_bundle` for a wrapper around this function
    which automatically handles passing everything in the correct order
    and in the correct units.

    Note: orbits = stars - 1

    Args:
        times: (iterable) times when to compute positions and velocities [days]
        masses: (iterable) mass for each star [solMass]
        xi, yi, zi: (iterable) initial Cartesian positions for each star [AU]
        vxi, vyi, vzi: (iterable) initial Cartesian velocities for each star [AU/day]
        rotperiods: (iterable) rotation periods for each star [day]
        t0: (float) epoch at which elements were given [days]
        vgamma: (float) gamma velocity [AU/day]
        integrator: (string) which integrator from the Rebound package
        stepsize: (float) step size [day]
        epsilon: (float) relative error controlling the step size [1]
        ltte: (bool) whether to account for light travel time effects
        gr: (bool) whether to account for general relativity effects
        j2: (bool) whether to account for J2 = -C20 or oblateness
        j2s: (iterable) J2 = -C20 or oblateness for each star [1]
        requivs: (iterable) equatorial radius for each star [AU]
        return_roche_euler: (bool) whether to return Roche parameters and Euler angles

    Returns:
        t: (numpy array) all computed times
        xs, ys, zs: (numpy arrays) time-dependent Cartesian positions [solRad]
        vxs, vys, vzs: (numpy arrays) time-dependent Cartesian velocities [solRad/day]
        ds, Fs: (numpy arrays) Roche parameters [1]
        theta, longan, incl: (numpy arrays) Euler angles [rad]

    """

    global _geometry

    if not _is_rebound:
        raise ImportError("rebound is not installed")

    if gr and not _is_reboundx:
        raise ImportError("reboundx is not installed (required for gr effects)")

    if j2 and not _is_reboundx:
        raise ImportError("reboundx is not installed (required for j2 effects)")

    times = np.asarray(times)

    nbod = len(masses)
    xs = np.zeros((nbod, len(times)))
    ys = np.zeros((nbod, len(times)))
    zs = np.zeros((nbod, len(times)))
    vxs = np.zeros((nbod, len(times)))
    vys = np.zeros((nbod, len(times)))
    vzs = np.zeros((nbod, len(times)))

    if return_roche_euler:
        ds = np.zeros((nbod, len(times)))
        Fs = np.zeros((nbod, len(times)))
        ethetas = np.zeros((nbod, len(times)))
        elongans = np.zeros((nbod, len(times)))
        eincls = np.zeros((nbod, len(times)))

    sim = rebound.Simulation()

    sim.integrator = integrator
    sim.dt = stepsize
    sim.ri_ias15.epsilon = epsilon
    sim.ri_whfast.corrector = 17
    sim.ri_whfast.safe_mode = 0;
    sim.G = 1.0

    if conf.devel:
        sim.status()

    for j in range(0, nbod):
        sim.add(primary=None, m=masses[j], x=xi[j], y=yi[j], z=zi[j], vx=vxi[j], vy=vyi[j], vz=vzi[j])

    sim.move_to_com()

    for particle in sim.particles:
        particle.vz -= vgamma

    if _is_reboundx:
        rebx = reboundx.Extras(sim)

    if gr:
        logger.info("enabling 'gr_full' in reboundx")
        gr = rebx.load_force("gr_full")
        gr.params["c"] = c.c.to("AU/d").value
        rebx.add_force(gr)

    if j2:
        logger.info("enabling 'j2' in reboundx")
        rebx = reboundx.Extras(sim)
        gh = rebx.load_force("j2")
        rebx.add_force(gh)

        for j in range(0, nbod):
            sim.particles[j].params["J2"] = j2s[j]
            sim.particles[j].params["R_eq"] = requivs[j]

    rb = np.zeros((nbod, 3))
    vb = np.zeros((nbod, 3))

    for i,time in enumerate(times):

        sim.integrate(time, exact_finish_time=True)

        for j in range(0, nbod):

            if ltte:
                particle = _ltte(sim, j, time)
            else:
                particle = sim.particles[j]

            rb[j][0] = particle.x
            rb[j][1] = particle.y
            rb[j][2] = particle.z
            vb[j][0] = particle.vx
            vb[j][1] = particle.vy
            vb[j][2] = particle.vz

        if return_roche_euler:

            elmts, euler, roche = geometry.invgeometry(masses, rb, vb, geometry=_geometry)

        fac = (1*u.AU).to(u.solRad).value

        xs[:,i] = fac * rb[:,0]
        ys[:,i] = fac * rb[:,1]
        zs[:,i] = fac * rb[:,2]
        vxs[:,i] = fac * vb[:,0]
        vys[:,i] = fac * vb[:,1]
        vzs[:,i] = fac * vb[:,2]

        if return_roche_euler:
            ds[:,i] = roche[:,0]
            Fs[:,i] = roche[:,1]/rotperiods[:]
            ethetas[:,i] = euler[:,0]
            elongans[:,i] = euler[:,1]
            eincls[:,i] = euler[:,2]

    if return_roche_euler:
        return times, xs, ys, zs, vxs, vys, vzs, ds, Fs, ethetas, elongans, eincls
    else:
        return times, xs, ys, zs, vxs, vys, vzs


def _ltte(sim, j, time):
    """
    Light-time effect.

    """
    scale_factor = (u.AU/c.c).to(u.d).value

    def residual(t):
        if sim.t != t:
            sim.integrate(t, exact_finish_time=True)
        z = sim.particles[j].z
        return t - z*scale_factor - time

    propertime = newton(residual, time)

    if sim.t != propertime:
        sim.integrate(propertime)

    return sim.particles[j]


