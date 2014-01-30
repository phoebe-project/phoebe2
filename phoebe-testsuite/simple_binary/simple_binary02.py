"""
Simple binary (fitting)
==========================

In this example we'll create a simple binary, starting from two stars, and
fit it. We'll use the results from the other simple binary tutorial as input
observations.
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
                            atm='kurucz', ld_coeffs='kurucz', ld_func='claret')
star2 = phoebe.ParameterSet('star', mass=0.8, radius=0.9, teff=4123.,
                            atm='kurucz', ld_coeffs='kurucz', ld_func='claret')

# Derive component parameters (i.e. Roche potentials, synchronicity parameters)
# and orbit parameters (semi-major axis) from the stellar parameters. We need
# set of course the period of the system. Let's take 10 days.

comp1, comp2, orbit = phoebe.create.binary_from_stars(star1, star2, period=(10.,'d'))

# But we'll override the synchronicity parameters to be according to synchronous
# rotation

comp1['syncpar'] = 1.0
comp2['syncpar'] = 1.0
orbit['ecc'] = 0.1

# To compute the objects, we need to define a mesh. We'll take the same one for
# the two components.

mesh = phoebe.ParameterSet('mesh:marching', delta=0.07)

# Next, we want to observe the system in the KEPLER passband, and compute some
# radial velocities. We only set the reference to one of them explicitly, to
# show the different approaches to refering to the parameterSets later.

lcdep = phoebe.ParameterSet('lcdep', passband='KEPLER.V', atm='kurucz',
                            ld_func='claret', ld_coeffs='kurucz', ref='my kepler lc')
rvdep = phoebe.ParameterSet('rvdep', passband='JOHNSON.V', atm='kurucz',
                            ld_func='claret', ld_coeffs='kurucz')

# Loading observations
#-----------------------

# The observations come in a simple form of a multicolumn text file, so we
# can easily parse them. We need to tell the code which observation belongs to
# which lcdep, so we set the label explicitly:


input_lc, lcdep_ = phoebe.parse_lc('lc.obs')
input_lc['ref'] = 'my kepler lc'

# Fitting setup
#---------------

"""
First, we observe that the normalisation of the observations is not the same
as for the model computations. The observations are normalised with the mean
flux, whereas the model is in absolute fluxes. We can easily scale the
observations to the model via :py:func:`a linear fit <phoebe.universe.backend.compute_pblum_or_l3>`. The slope is then interpreted as
the passband luminosity, the offset as third light. In this example, we'll fit
them both but note that third light should be zero if everything goes well.
"""


input_lc.get_parameter('pblum').set_adjust(True)
input_lc.get_parameter('l3').set_adjust(True)


# Other parameters we want to set (arbitrarily here) are the inclination, the
# eccentricity and the argument of periastron. Light curves typically constrain
# :math:`e\cos\omega` and :math:`e\sin\omega` better than :math:`e` and
# :math:`\omega`, so we'll add those parameters first:


phoebe.parameters.tools.add_esinw_ecosw(orbit)

# Then we can set the parameters to be adjustable, set their priors and set
# an initial value

orbit['incl'] = 88.
incl = orbit.get_parameter('incl')
incl.set_adjust(True)
incl.set_prior(distribution='uniform', lower=0.0, upper=180.)

orbit['ecosw'] = 0.05
ecosw = orbit.get_parameter('ecosw')
ecosw.set_adjust(True)
ecosw.set_prior(distribution='uniform', lower=-1, upper=1)

orbit['esinw'] = 0.05
esinw = orbit.get_parameter('esinw')
esinw.set_adjust(True)
esinw.set_prior(distribution='uniform', lower=-1, upper=1)

# Let's use the Nelder-Mead simplex method to fit:

fitparams = phoebe.ParameterSet('fitting:lmfit', method='nelder')

# Body setup
#--------------

# We need two BinaryRocheStars and put them in an orbit:

star1 = phoebe.BinaryRocheStar(comp1, mesh=mesh, pbdep=[lcdep,rvdep])
star2 = phoebe.BinaryRocheStar(comp2, mesh=mesh, pbdep=[lcdep,rvdep])

system = phoebe.BinaryBag([star1, star2], orbit=orbit, obs=[input_lc])


# Fitting parameters
#-------------------------------

# Run the fitter, after specifying that we need to  use the fast eclipse
# detection algorithm.

params = phoebe.ParameterSet('compute', eclipse_alg='binary')
feedback = phoebe.fitting.run(system, params, fitparams=fitparams, mpi=None, accept=True)

# Save the results to a file:
system.save('mymodel.phoebe')
feedback.save('myfeedback.phoebe')

# Analysis of results:
#-----------------------

# Retrieve the system and feedback after saving it to a file. In this script,
# this is of course not necessary since we continue all the way from the beginning,
# but if you would want to analyse the results later, that is probably the way
# you'd want to do it.

system = phoebe.load_body('mymodel.phoebe')
feedback = phoebe.load_ps('myfeedback.phoebe')

syn = system.get_synthetic(category='lc', ref='my kepler lc').asarray()

plt.figure()
plt.plot(syn['time'], syn['flux'], 'ko-')

syn1 = system[0].get_synthetic(category='rv').asarray()
syn2 = system[1].get_synthetic(category='rv').asarray()

plt.figure()
plt.plot(syn1['time'], syn1['rv'], 'ko-')
plt.plot(syn2['time'], syn2['rv'], 'ro-')

plt.show()
