

import numpy as np
from numpy import pi,sqrt,cos,sin,tan,arctan
from scipy.optimize import newton

from phoebe import u, c
from phoebe import conf

import logging
logger = logging.getLogger("DYNAMICS.KEPLERIAN")
logger.addHandler(logging.NullHandler())

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

    computeps = b.get_compute(compute, check_visible=False, force_ps=True)
    ltte = computeps.get_value('ltte', check_visible=False, **kwargs)

    # make sure times is an array and not a list
    times = np.array(times)

    vgamma = b.get_value('vgamma', context='system', unit=u.solRad/u.d)
    t0 = b.get_value('t0', context='system', unit=u.d)

    hier = b.hierarchy
    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits()
    s = b.filter(context='component')

    periods, eccs, smas, t0_perpasses, per0s, long_ans, incls, dpdts, \
    deccdts, dperdts, components = [],[],[],[],[],[],[],[],[],[],[]


    for component in starrefs:

        # we need to build a list of all orbitlabels underwhich this component
        # belongs.  For a simple binary this is just the parent, but for hierarchical
        # systems we need to get the labels of the outer-orbits as well
        ancestororbits = []
        comp = component
        while hier.get_parent_of(comp) in orbitrefs:
            comp = hier.get_parent_of(comp)
            ancestororbits.append(comp)

        #print "***", component, ancestororbits

        periods.append([s.get_value('period', u.d, component=orbit) for orbit in ancestororbits])
        eccs.append([s.get_value('ecc', component=orbit) for orbit in ancestororbits])
        t0_perpasses.append([s.get_value('t0_perpass', u.d, component=orbit) for orbit in ancestororbits])
        per0s.append([s.get_value('per0', u.rad, component=orbit) for orbit in ancestororbits])
        long_ans.append([s.get_value('long_an', u.rad, component=orbit) for orbit in ancestororbits])
        incls.append([s.get_value('incl', u.rad, component=orbit) for orbit in ancestororbits])
        dpdts.append([s.get_value('dpdt', u.d/u.d, component=orbit) for orbit in ancestororbits])
        if conf.devel:
            deccdts.append([s.get_value('deccdt', u.dimensionless_unscaled/u.d, component=orbit) for orbit in ancestororbits])
        else:
            deccdts.append([0.0 for orbit in ancestororbits])
        dperdts.append([s.get_value('dperdt', u.rad/u.d, component=orbit) for orbit in ancestororbits])

        # sma needs to be the COMPONENT sma.  This is stored in the bundle for stars, but is NOT
        # for orbits in orbits, so we'll need to recompute those from the mass-ratio and sma of
        # the parent orbit.

        smas_this = []
        for comp in [component]+ancestororbits[:-1]:
            if comp in starrefs:
                smas_this.append(s.get_value('sma', u.solRad, component=comp))
            else:
                q = s.get_value('q', component=hier.get_parent_of(comp))
                comp_comp = hier.get_primary_or_secondary(comp)

                # NOTE: similar logic is also in constraints.comp_sma
                # If changing any of the logic here, it should be changed there as well.
                if comp_comp == 'primary':
                    qthing = (1. + 1./q)
                else:
                    qthing = (1. + q)

                smas_this.append(s.get_value('sma', u.solRad, component=hier.get_parent_of(comp)) / qthing)

        smas.append(smas_this)

        # components is whether an entry is the primary or secondary in its parent orbit, so here we want
        # to start with component and end one level short of the top-level orbit
        components.append([hier.get_primary_or_secondary(component=comp) for comp in [component]+ancestororbits[:-1]])


    return  dynamics(times, periods, eccs, smas, t0_perpasses, per0s, \
                    long_ans, incls, dpdts, deccdts, dperdts, \
                    components, t0, vgamma, \
                    mass_conservation=True, ltte=ltte, return_euler=return_euler)




def dynamics(times, periods, eccs, smas, t0_perpasses, per0s, long_ans, incls,
            dpdts, deccdts, dperdts, components, t0=0.0, vgamma=0.0,
            mass_conservation=True, ltte=False, return_euler=False):
    """
    Compute the positions and velocites of each star in their nested
    Keplerian orbits at a given list of times.

    See :func:`dynamics_from_bundle` for a wrapper around this function
    which automatically handles passing everything in the correct order
    and in the correct units.

    Args:
        times: (iterable) times at which to compute positions and
            velocities for each star
        periods: (iterable) period of the parent orbit for each star
            [days]
        eccs: (iterable) eccentricity of the parent orbit for each star
        smas: (iterable) semi-major axis of the parent orbit for each
            star [solRad]
        t0_perpasses: (iterable) t0_perpass of the parent orbit for each
            star [days]
        per0s: (iterable) longitudes of periastron of the parent orbit
            for each star [rad]
        long_ans: (iterable) longitudes of the ascending node of the
            parent orbit for each star [rad]
        incls: (iterable) inclination of the parent orbit for each
            star [rad]
        dpdts: (iterable) change in period with respect to time of the
            parent orbit for each star [days/day]
        deccdts: (iterable) change in eccentricity with respect to time
            of the parent orbit for each star [1/day]
        dperdts: (iterable) change in periastron with respect to time
            of the parent orbit for each star [rad/d]
        components: (iterable) component ('primary' or 'secondary') of
            each star within its parent orbit [string]
        t0: (float, default=0) time at which all initial values (ie period, per0)
            are given [days]
        mass_conservation: (bool, optional) whether to require mass
            conservation if any of the derivatives (dpdt, dperdt, etc)
            are non-zero [default: True]
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
    # TODO: NOTE: smas must be per-component, not per-orbit
    # TODO: steal some documentation from 2.0a:keplerorbit.py:get_orbit
    # TODO: deal with component number more smartly

    def binary_dynamics(times, period, ecc, sma, t0_perpass, per0, long_an,
                        incl, dpdt, deccdt, dperdt, component='primary',
                        t0=0.0, vgamma=0.0, mass_conservation=True,
                        com_pos=(0.,0.,0.), com_vel=(0.,0.,0.), com_euler=(0.,0.,0.)):
        """
        """
        # TODO: steal some documentation from 2.0a:keplerorbit.py:get_orbit


        #-- if dpdt is non-zero, the period is actually an array, and the semi-
        #   major axis changes to match Kepler's third law (unless
        #   `mass_conservation` is set to False)
        if dpdt!=0:
            period_ = period
            period = dpdt*(times-t0) + period_
            if mass_conservation and not np.isscalar(period):
                 sma = sma/period[0]**2*period**2
            elif mass_conservation:
                 sma = sma/period_**2*period**2

        #-- if dperdt is non-zero, the argument of periastron is actually an
        #   array
        if dperdt!=0.:
            per0 = dperdt*(times-t0) + per0
        #-- if deccdt is non-zero, the eccentricity is actually an array
        if deccdt!=0:
            ecc = deccdt*(times-t0) + ecc


        #-- compute orbit
        n = 2*pi/period
        ma = n*(times-t0_perpass)
        E,theta = _true_anomaly(ma,ecc)
        r = sma*(1-ecc*cos(E))
        PR = r*sin(theta)
        #-- compute rdot and thetadot
        l = r*(1+ecc*cos(theta))#-omega))
        L = 2*pi*sma**2/period*sqrt(1-ecc**2)
        rdot = L/l*ecc*sin(theta)#-omega)
        thetadot = L/r**2
        #-- the secondary is half an orbit further than the primary
        if 'sec' in component.lower():
            theta += pi
        #-- take care of Euler angles
        theta_ = theta+per0
        #-- convert to the right coordinate frame
        #-- create some shortcuts
        sin_theta_ = sin(theta_)
        cos_theta_ = cos(theta_)
        sin_longan = sin(long_an)
        cos_longan = cos(long_an)
        #-- spherical coordinates to cartesian coordinates. Note that we actually
        #   set incl=-incl (but of course it doesn't show up in the cosine). We
        #   do this to match the convention that superior conjunction (primary
        #   eclipsed) happens at periastron passage when per0=90 deg.
        x = r*(cos_longan*cos_theta_ - sin_longan*sin_theta_*cos(incl))
        y = r*(sin_longan*cos_theta_ + cos_longan*sin_theta_*cos(incl))
        z = r*(sin_theta_*sin(-incl))
        #-- spherical vectors to cartesian vectors, and then rotated for
        #   the Euler angles Omega and i.
        vx_ = cos_theta_*rdot - sin_theta_*r*thetadot
        vy_ = sin_theta_*rdot + cos_theta_*r*thetadot
        vx = cos_longan*vx_ - sin_longan*vy_*cos(incl)
        vy = sin_longan*vx_ + cos_longan*vy_*cos(incl)
        vz = sin(-incl)*vy_
        #-- that's it!

        # correct by vgamma (only z-direction)
        vz += vgamma
        z += vgamma * (times-t0)

        return (x+com_pos[0],y+com_pos[1],z+com_pos[2]),\
                (vx+com_vel[0],vy+com_vel[1],vz+com_vel[2]),\
                (theta_,long_an,incl)
                # (theta_+com_euler[0],long_an+com_euler[1],incl+com_euler[2])

    def binary_dynamics_nested(times, periods, eccs, smas, \
                            t0_perpasses, per0s, long_ans, incls, dpdts, deccdts, \
                            dperdts, components, t0, vgamma, \
                            mass_conservation):

        """
        compute the (possibly nested) positions of a single component (ltte should be
        handle externally)
        """


        if not hasattr(periods, '__iter__'):
            # then we don't have to worry about nesting, and each item should
            # be a single value ready to pass to binary_dynamics
            pos, vel, euler = binary_dynamics(times, periods, eccs, smas, t0_perpasses, \
                                per0s, long_ans, incls, dpdts, deccdts, dperdts, components, \
                                t0, vgamma, mass_conservation)

        else:
            # Handle nesting - if this star is not in the top-level orbit, then
            # all values should actually be lists.  We want to sort by period to handle
            # the outer orbits first and then apply those offsets to the inner-orbit(s)

            # let's convert to arrays so we can use argsort easily
            periods = np.array(periods)
            eccs = np.array(eccs)
            smas = np.array(smas)
            t0_perpasses = np.array(t0_perpasses)
            per0s = np.array(per0s)
            long_ans = np.array(long_ans)
            incls = np.array(incls)
            dpdts = np.array(dpdts)
            deccdts = np.array(deccdts)
            dperdts = np.array(dperdts)
            components = np.array(components)

            si = periods.argsort()[::-1]

            #print "***", periods, si

            pos = (0.0, 0.0, 0.0)
            vel = (0.0, 0.0, 0.0)
            euler = (0.0, 0.0, 0.0)

            for period, ecc, sma, t0_perpass, per0, long_an, incl, dpdt, \
                deccdt, dperdt, component in zip(periods[si], eccs[si], \
                smas[si], t0_perpasses[si], per0s[si], long_ans[si], \
                incls[si], dpdts[si], deccdts[si], dperdts[si], components[si]):

                pos, vel, euler = binary_dynamics(times, period, ecc, sma, t0_perpass, \
                                    per0, long_an, incl, dpdt, deccdt, dperdt, component, \
                                    t0, vgamma, mass_conservation,
                                    com_pos=pos, com_vel=vel, com_euler=euler)



        return pos, vel, euler




    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []

    if return_euler:
        ethetas, elongans, eincls = [], [], []

    for period, ecc, sma, t0_perpass, per0, long_an, incl, dpdt, deccdt, \
            dperdt, component in zip(periods, eccs, smas, t0_perpasses, per0s, long_ans, \
            incls, dpdts, deccdts, dperdts, components):

        # We now have the orbital parameters for a single star/component.

        if ltte:
            #scale_factor = 1.0/c.c.value * c.R_sun.value/(24*3600.)
            scale_factor = (c.R_sun/c.c).to(u.d).value

            def propertime_barytime_residual(t):
                pos, vel, euler = binary_dynamics_nested(time, period, ecc, sma, \
                                t0_perpass, per0, long_an, incl, dpdt, deccdt, \
                                dperdt, components=component, t0=t0, vgamma=vgamma, \
                                mass_conservation=mass_conservation)
                z = pos[2]
                return t - z*scale_factor - time

            # Finding that right time is easy with a Newton optimizer:
            propertimes = [newton(propertime_barytime_residual, time) for \
               time in times]
            propertimes = np.array(propertimes).ravel()

            pos, vel, euler = binary_dynamics_nested(propertimes, period, ecc, sma, \
                            t0_perpass, per0, long_an, incl, dpdt, deccdt, \
                            dperdt, components=component, t0=t0, vgamma=vgamma, \
                            mass_conservation=mass_conservation)


        else:
            pos, vel, euler = binary_dynamics_nested(times, period, ecc, sma, \
                    t0_perpass, per0, long_an, incl, dpdt, deccdt, \
                    dperdt, components=component, t0=t0, vgamma=vgamma, \
                    mass_conservation=mass_conservation)


        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])
        vxs.append(vel[0])
        vys.append(vel[1])
        vzs.append(vel[2])
        if return_euler:
            ethetas.append(euler[0])
            elongans.append([euler[1]]*len(euler[0]))
            eincls.append([euler[2]]*len(euler[0]))


    # if return_euler:
    #     return times, \
    #         xs*u.solRad, ys*u.solRad, zs*u.solRad, \
    #         vxs*u.solRad/u.d, vys*u.solRad/u.d, vzs*u.solRad/u.d, \
    #         ethetas*u.rad, elongans*u.rad, eincls*u.rad
    # else:
    #     return times, \
    #         xs*u.solRad, ys*u.solRad, zs*u.solRad, \
    #         vxs*u.solRad/u.d, vys*u.solRad/u.d, vzs*u.solRad/u.d    if return_euler:

    # d, solRad, solRad/d, rad
    if return_euler:
        return times, \
            xs, ys, zs, \
            vxs, vys, vzs, \
            ethetas, elongans, eincls
    else:
        return times, \
            xs, ys, zs, \
            vxs, vys, vzs




def _true_anomaly(M,ecc,itermax=8):
    r"""
    Calculation of true and eccentric anomaly in Kepler orbits.

    ``M`` is the phase of the star, ``ecc`` is the eccentricity

    See p.39 of Hilditch, 'An Introduction To Close Binary Stars':

    Kepler's equation:

    .. math::

        E - e\sin E = \frac{2\pi}{P}(t-T)

    with :math:`E` the eccentric anomaly. The right hand size denotes the
    observed phase :math:`M`. This function returns the true anomaly, which is
    the position angle of the star in the orbit (:math:`\theta` in Hilditch'
    book). The relationship between the eccentric and true anomaly is as
    follows:

    .. math::

        \tan(\theta/2) = \sqrt{\frac{1+e}{1-e}} \tan(E/2)

    @parameter M: phase
    @type M: float
    @parameter ecc: eccentricity
    @type ecc: float
    @keyword itermax: maximum number of iterations
    @type itermax: integer
    @return: eccentric anomaly (E), true anomaly (theta)
    @rtype: float,float
    """
    # Initial value
    Fn = M + ecc*sin(M) + ecc**2/2.*sin(2*M)

    # Iterative solving of the transcendent Kepler's equation
    for i in range(itermax):
        F = Fn
        Mn = F-ecc*sin(F)
        Fn = F+(M-Mn)/(1.-ecc*cos(F))
        keep = F!=0 # take care of zerodivision
        if hasattr(F,'__iter__'):
            if np.all(abs((Fn-F)[keep]/F[keep])<0.00001):
                break
        elif (abs((Fn-F)/F)<0.00001):
            break

    # relationship between true anomaly (theta) and eccentric anomaly (Fn)
    true_an = 2.*arctan(sqrt((1.+ecc)/(1.-ecc))*tan(Fn/2.))

    return Fn,true_an
