#!/usr/bin/env python3

import numpy as np
from numpy import pi,sqrt,cos,sin,tan,arctan
from scipy.optimize import newton

from phoebe import u, c

from phoebe.dynamics import coord_j2b
from phoebe.dynamics import orbel_el2xv
from phoebe.dynamics import orbel_ehie

import logging
logger = logging.getLogger("DYNAMICS.KEPLERIAN")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def dynamics_from_bundle(b, times, compute=None, return_euler=False, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    NOTE: you must either provide compute (the label) OR all relevant options
    as kwargs (ltte)

    Args:
        b: (Bundle) the bundle with a set hierarchy
        times: (list or array) times at which to run the dynamics
        return_euler: (bool, default=False) whether to include euler angles
            in the return

    Returns:
        t, xs, ys, zs, vxs, vys, vzs [, theta, longan, incl].
        t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.
        Euler angles (theta, longan, incl) are only returned if return_euler is
        set to True.

    """

    b.run_delayed_constraints()

    computeps = b.get_compute(compute=compute, force_ps=True, **_skip_filter_checks)
    if len(computeps.computes) == 1:
        ltte = computeps.get_value(qualifier='ltte', ltte=kwargs.get('ltte', None), default=False, **_skip_filter_checks)
    else:
        ltte = False

    times = np.array(times)

    hier = b.hierarchy
    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits()

    G = c.G.to('solRad3 / (Msun d2)').value
    masses = [b.get_value(qualifier='mass', unit=u.solMass, component=component, context='component', **_skip_filter_checks) * G for component in starrefs]
    smas = [b.get_value(qualifier='sma', unit=u.solRad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    eccs = [b.get_value(qualifier='ecc', component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    incls = [b.get_value(qualifier='incl', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    per0s = [b.get_value(qualifier='per0', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    long_ans = [b.get_value(qualifier='long_an', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    t0_perpasses = [b.get_value(qualifier='t0_perpass', unit=u.d, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    periods = [b.get_value(qualifier='period', unit=u.d, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]
    mean_anoms = [b.get_value(qualifier='mean_anom', unit=u.rad, component=component, context='component', **_skip_filter_checks) for component in orbitrefs]

    vgamma = b.get_value(qualifier='vgamma', context='system', unit=u.solRad/u.d, **_skip_filter_checks)
    t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

    return dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms, \
        t0, vgamma, ltte, \
        return_euler=return_euler)


def dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms, \
    t0=0.0, vgamma=0.0, ltte=False, \
    return_euler=False):

    nbod = len(masses)
    rj = np.array(nbod*[[0.0, 0.0, 0.0]])
    vj = np.array(nbod*[[0.0, 0.0, 0.0]])

    xs = np.zeros((nbod, len(times)))
    ys = np.zeros((nbod, len(times)))
    zs = np.zeros((nbod, len(times)))
    vxs = np.zeros((nbod, len(times)))
    vys = np.zeros((nbod, len(times)))
    vzs = np.zeros((nbod, len(times)))

    if return_euler:
        ethetas  = np.zeros((nbod, len(times)))
        elongans = np.zeros((nbod, len(times)))
        eincls   = np.zeros((nbod, len(times)))

    for i, t in enumerate(times):

        # compute Jacobi coordinates
        msum = masses[0]
        for j in range(1,nbod):
            msum += masses[j]
            ialpha = -1
            a = smas[j-1]
            n = sqrt(msum/a**3)
            P = 2.0*np.pi/n
            M = mean_anoms[j-1] + n*(t-t0)
            elmts = [a, eccs[j-1], incls[j-1], long_ans[j-1], per0s[j-1], M]
 
            rj[j], vj[j] = orbel_el2xv.orbel_el2xv(msum, ialpha, elmts)

            if return_euler:
                euler = orbel_el2xv.get_euler(elmts) 

                ethetas[j][i] = euler[0]
                elongans[j][i] = euler[1]
                eincls[j][i] = euler[2]

                # need to copy the primary component
                # need to add np.pi for the secondary component
                if j==1:
                    ethetas[0][i] = euler[0]
                    elongans[0][i] = euler[1]
                    eincls[0][i] = euler[2]

                    ethetas[j][i] += np.pi

        # convert to barycentric frame
        rb, vb = coord_j2b.coord_j2b(masses, rj, vj)

        rb[:,2] -= vgamma*(t-t0)
        vb[:,2] -= vgamma

        xs[:,i] = -rb[:,0]
        ys[:,i] = -rb[:,1]
        zs[:,i] = rb[:,2]
        vxs[:,i] = -vb[:,0]
        vys[:,i] = -vb[:,1]
        vzs[:,i] = vb[:,2]

    if return_euler:
        return times, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls
    else:
        return times, xs, ys, zs, vxs, vys, vzs

if __name__ == "__main__":

    # Sun-Earth
    times = np.array([0.0])
    masses = np.array([1.0, 0.0])
    smas = np.array([1.0])
    eccs = np.array([0.0])
    incls = np.array([0.0])
    per0s = np.array([0.0])
    long_ans = np.array([0.0])
    mean_anoms = np.array([0.0])

    G = c.G.to('AU3 / (Msun d2)').value
    masses *= G

    times, xs, ys, zs, vxs, vys, vzs = dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms)

    print("xs = ", xs)
    print("ys = ", ys)
    print("zs = ", zs)
    print("vxs = ", vxs)
    print("vys = ", vys)
    print("vzs = ", vzs)

    print("v_of_Earth = ", vys[1][0]*(u.au/u.d).to('m s^-1'), " m/s")

    # 3-body problem
    times = np.array([0.0])
    masses = np.array([1.0, 1.0, 1.0])
    smas = np.array([1.0, 10.0])
    eccs = np.array([0.0, 0.0])
    incls = np.array([0.0, 0.0])
    per0s = np.array([0.0, 0.0])
    long_ans = np.array([0.0, 0.0])
    mean_anoms = np.array([0.0, 0.0])

    times, xs, ys, zs, vxs, vys, vzs = dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms)

    print("")
    print("xs = ", xs)
    print("ys = ", ys)
    print("zs = ", zs)
    print("vxs = ", vxs)
    print("vys = ", vys)
    print("vzs = ", vzs)


