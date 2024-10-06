"""
"""

import numpy as np
from scipy.optimize import newton

from phoebe import u, c
from phoebe import conf

import rebound
from phoebe.dynamics import geometry
from phoebe.dynamics import invgeometry

import logging
logger = logging.getLogger("DYNAMICS.NBODY")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

_geometry = None

def dynamics_from_bundle(b, times, compute=None, return_roche_euler=False, **kwargs):
    """
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

    nbod = len(masses)
    elmts = []
    for j in range(0, nbod-1):
        elmts.append([smas[j], eccs[j], incls[j], long_ans[j], per0s[j], mean_anoms[j]])

    xi, yi, zi, vxi, vyi, vzi = geometry.geometry(masses, elmts, geometry=_geometry)

    return dynamics(times, masses, xi, yi, zi, vxi, vyi, vzi, \
                    rotperiods, t0, vgamma, stepsize, ltte, gr, \
                    integrator, return_roche_euler=return_roche_euler, \
                    epsilon=epsilon)


def dynamics(times, masses, xi, yi, zi, vxi, vyi, vzi,
        rotperiods=None, t0=0.0, vgamma=0.0, stepsize=0.01, ltte=False, gr=False,
        integrator='ias15', return_roche_euler=False,
        epsilon=1.0e-9):

    global _geometry

    def particle_ltte(sim, j, time):

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

    rb = np.zeros((nbod, 3))
    vb = np.zeros((nbod, 3))

    for i,time in enumerate(times):

        sim.integrate(time, exact_finish_time=True)

        for j in range(len(masses)):

            if ltte:
                particle = particle_ltte(sim, j, time)
            else:
                particle = sim.particles[j]

            rb[j][0] = particle.x
            rb[j][1] = particle.y
            rb[j][2] = particle.z
            vb[j][0] = particle.vx
            vb[j][1] = particle.vy
            vb[j][2] = particle.vz

        if return_roche_euler:

            elmts, euler, roche = invgeometry.invgeometry(masses, rb, vb, geometry=_geometry)

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


