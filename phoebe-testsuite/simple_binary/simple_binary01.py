"""
Simple binary (from stars)
==========================

In this example we'll create a simple binary, starting from two stars.
"""

# Import necessary modules and set up a logger.
import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.get_basic_logger()

# Parameter preparation
#-----------------------

# Define two stars, and set some of their parameters to be not the default ones.

star1 = phoebe.ParameterSet('star', mass=1.2, radius=1.1, teff=6321.,
                            atm='kurucz_p00', ld_coeffs='kurucz_p00', ld_func='claret')
star2 = phoebe.ParameterSet('star', mass=0.8, radius=0.9, teff=4123.,
                            atm='kurucz_p00', ld_coeffs='kurucz_p00', ld_func='claret')

# Derive component parameters (i.e. Roche potentials, synchronicity parameters)
# and orbit parameters (semi-major axis) from the stellar parameters. We need
# set of course the period of the system. Let's take 10 days.

comp1, comp2, orbit = phoebe.create.binary_from_stars(star1, star2, period=(10.,'d'))

# But we'll override the synchronicity parameters to be according to synchronous
# rotation

comp1['syncpar'] = 1.0
comp2['syncpar'] = 1.0

# To compute the objects, we need to define a mesh. We'll take the same one for
# the two components.

mesh = phoebe.ParameterSet('mesh:marching', delta=0.07)

# Next, we want to observe the system in the KEPLER passband, and compute some
# radial velocities. We only set the reference to one of them explicitly, to
# show the different approaches to refering to the parameterSets later.

lcdep = phoebe.ParameterSet('lcdep', passband='KEPLER.MEAN', atm='kurucz_p00',
                            ld_func='claret', ld_coeffs='kurucz_p00', ref='my kepler lc')
rvdep1 = phoebe.ParameterSet('rvdep', passband='JOHNSON.V', atm='kurucz_p00',
                            ld_func='claret', ld_coeffs='kurucz_p00')
rvdep2 = phoebe.ParameterSet('rvdep', passband='JOHNSON.V', atm='kurucz_p00',
                            ld_func='claret', ld_coeffs='kurucz_p00')


# Create observations
times = np.linspace(0, orbit['period'],100)
lcobs = phoebe.LCDataSet(time=times, ref=lcdep['ref'])
rvobs1 = phoebe.RVDataSet(time=times, ref=rvdep1['ref'])
rvobs2 = phoebe.RVDataSet(time=times, ref=rvdep2['ref'])

# Body setup
#--------------

# We need two BinaryRocheStars and put them in an orbit:
orbit['ecc'] = 0.1
star1 = phoebe.BinaryRocheStar(comp1, mesh=mesh, pbdep=[lcdep,rvdep1], obs=[rvobs1], orbit=orbit)
star2 = phoebe.BinaryRocheStar(comp2, mesh=mesh, pbdep=[lcdep,rvdep2], obs=[rvobs2], orbit=orbit)

system = phoebe.BodyBag([star1, star2], obs=[lcobs])


# Computation of observables
#-------------------------------

# Let us compute a whole orbit, sampled with 100 points in time, and compute
# the light curve and radial velocity curve. We use the binary eclipse algorithm
# to make computations go faster.

phoebe.observatory.compute(system, eclipse_alg='binary')

# Analysis of results:
#-----------------------

# Retrieve the results of the Kepler light curve computations, and convert all
# arrays in the parameterSet to arrays (instead of lists), for easy computation.

syn = system.get_synthetic(category='lc', ref='my kepler lc').asarray()

plt.figure()
plt.plot(syn['time'], syn['flux'], 'ko-')
plt.savefig('simple_binary_lc.png')


syn1 = system[0].get_synthetic(category='rv').asarray()
syn2 = system[1].get_synthetic(category='rv').asarray()

plt.figure()
plt.plot(syn1['time'], syn1['rv'], 'ko-')
plt.plot(syn2['time'], syn2['rv'], 'ro-')
plt.savefig('simple_binary_rv.png')


"""

+--------------------------------------------+--------------------------------------------+
| .. image:: images_tut/simple_binary_lc.png | .. image:: images_tut/simple_binary_rv.png |
|    :width: 233px                           |    :width: 233px                           |
+--------------------------------------------+--------------------------------------------+

"""

# Let us save these computations as observations for a next tutorial:

flux = syn['flux']/syn['flux'].mean()
flux += np.random.normal(size=len(flux), scale=0.02)
rv1 = syn1['rv'] + np.random.normal(size=len(syn1), scale=1.)
rv2 = syn2['rv'] + np.random.normal(size=len(syn2), scale=1.)

np.savetxt('lc.obs', np.column_stack([syn['time'], flux , 0.02*np.ones(len(syn))]))
np.savetxt('rv1.obs', np.column_stack([syn1['time'],rv1 , np.ones(len(syn1))]))
np.savetxt('rv2.obs', np.column_stack([syn2['time'], rv2, np.ones(len(syn2))]))
plt.show()
