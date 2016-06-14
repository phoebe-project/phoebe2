#
# Timing the eclipsing in Phoebe beta
#

import phoebe2
import numpy as np
import time


# period
# careful with t0_perpass if you change the period
period = 3.0

# syncpar
F = 1.0
period_rot = period/F
freq_rot = 2*np.pi/period_rot

# potentials (omega)
Phi = 9.5

# masses
q = 1.0

m_primary = 0.38
m_secondary = m_primary * q
masses = [m_primary, m_secondary]

# semi-major axis (solar radii)
sma = 8.0

# ecc (d = 1-e at periastron)
ecc = 0.0

# things that aren't really important for eclipse detection:
abun = 0.0
teff = 6000
gravb_bol = 0.2
gravb_law = 'zeipel'

Q = phoebe2.backend.mesh

# build the meshes and the system
primarymesh = Q.Star( F, Phi, masses, sma,
                      ecc, freq_rot, teff,
                      gravb_bol, gravb_law,
                      abun,
                      delta=0.1,
                      maxpoints=1e6,
                      ind_self=0, ind_sibling=1,
                      comp_no=1)

secondarymesh = Q.Star( F, Phi, masses, sma,
                        ecc, freq_rot, teff,
                        gravb_bol, gravb_law,
                        abun,
                        delta=0.1,
                        maxpoints=1e6,
                        ind_self=1,
                        ind_sibling=0,
                        comp_no=2)

system = Q.System({'primary': primarymesh,
                   'secondary': secondarymesh})


# place the meshes in their orbits
periods = np.array([period, period])
eccs = np.array([ecc, ecc])
smas = np.array([sma, sma])
t0_perpasses = np.array([-0.75, -0.75])
per0s = np.array([0, 0])
long_ans = np.array([0, 0])
incls = np.array([np.pi/2, np.pi/2])
dpdts = np.array([0, 0])
deccdts = np.array([0, 0])
dperdts = np.array([0, 0])
components = ['primary', 'secondary']

times = np.array([0.])
ts, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls =\
    phoebe2.dynamics.keplerian.dynamics(times, periods, eccs,
                                        smas, t0_perpasses,
                                        per0s, long_ans, incls,
                                        dpdts, deccdts, dperdts,
                                        components,
                                        return_euler=True)

system.initialize_meshes()

system.update_positions(times[0], xs[:,0], ys[:,0], zs[:,0],
                        vxs[:,0], vys[:,0], vzs[:,0],
                        ethetas[:,0], elongans[:,0], eincls[:,0])

# detect eclipses
start = time.time()
system.handle_eclipses(eclipse_alg='graham', subdiv_num=1)
end = time.time()

# let's see if we were expecting to eclipse
max_rs = [body.max_r for body in system.bodies]
proj_sep_sq = sum([(c[1]-c[0])**2 for c in system.xs, system.ys])
ecl = proj_sep_sq < sum(max_rs)**2
print "should have eclipsed: ", ecl

# let's make sure it worked
visibilities = system.meshes.get_column('visibility')
print "primary visibility:", visibilities['primary'].sum() / len(visibilities['primary'])
print "secondary visibility:", visibilities['secondary'].sum() / len(visibilities['secondary'])


print "time[ms]=", 1000*(end-start)

