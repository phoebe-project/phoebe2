"""
Accretion disk
==============

Put a Roche lobe filling star in an orbit around an accretion disk.

For details on the implementation of the accretion disk, browse the source code. Also see the papers of `Copperwheat et al. (2010) <http://adsabs.harvard.edu/abs/2010MNRAS.402.1824C>`_ and `Wood et al. (1992) <http://adsabs.harvard.edu/abs/1992ApJ...393..729W>`_.

Parameters are wildly speculative and certainly not realistic.

Initialisation
--------------

"""
# First, import the necessary modules and set up a logger:
import phoebe
from phoebe.algorithms import eclipse
from phoebe.atmospheres import roche
logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------

# We initiate a small and thin accretion disk. The mass-loss rate is the one
# of the host star.

diskpars = phoebe.ParameterSet(context='accretion_disk')
diskpars['Rout'] = 11.,'Rsol'
diskpars['height'] = 2e-2,'Rsol'
diskpars['dmdt'] = 1e-6,'Msol/yr'
diskpars['label'] = 'accretion_disk'

# We need to compute a light curve, so we initiate a parameterSet defining the
# passband etc:
lcdep = phoebe.ParameterSet(context='lcdep')

# The system will be created as follows: first, a normal binary system will
# be constructed assuming a subgiant of near-solar mass and a heavier
# companion. We will put the subgiant actually in the system, and it will be
# almost filling it's entire roche lobe. We omit the secondary but put the
# accretion disk in its place.
# 
# So first, we create two sun like stars to put in a binary, and slightly change
# their parameters so that the primary is a subgiant and the secondary a little
# bit heavier:
star1 = phoebe.create.from_library('sun')
star1['mass'] = 0.9
star1['radius'] = 7.95,'Rsol'
star1['label'] = 'subgiant'
star2 = phoebe.create.from_library('sun')
star2['mass'] = 1.4,'Msol'

# The two objects should be placed in a binary, with a semi-major axis of around
# 25 solar radii, and we incline it so that is almost edge-on:
comp1,comp2,orbit = phoebe.create.binary_from_stars(star1,star2,sma=(25.,'Rsol'))
orbit['t0'] = 0.
orbit['incl'] = 75.,'deg'

# If you're interested in knowing what the actual critical radius of the primary
# is, you can obtain it with this simple command:
print(roche.calculate_critical_radius(orbit['q'],sma=orbit['sma']))

# Of course we also need a mesh:
mesh = phoebe.ParameterSet(context='mesh:marching',delta=0.1)

# Body setup
# ----------

# After creating the parameterSets, we need to construct the actual objects.
# We have one BinaryRocheStar and one AccretionDisk, and simply put them all in
# a BinaryBag:
mystar = phoebe.BinaryRocheStar(comp1,orbit,mesh,pbdep=[lcdep.copy()])
mydisk = phoebe.AccretionDisk(diskpars,pbdep=[lcdep])
system = phoebe.BinaryBag([mystar,mydisk],orbit=orbit,solve_problems=True)

# Computation of observables
# --------------------------

# We only want to compute everything at one time point, se we'll set the time
# of the system, detect the eclipses and the horizon, and make some figures:
system.set_time(0.255*orbit['period'])
#eclipse.detect_eclipse_horizon(system)
eclipse.horizon_via_normal(system)

# Analysis of results
# -------------------

system.plot2D(select='teff',savefig='accretion_disk_teff')
system.plot2D(select='rv',savefig='accretion_disk_rv')
system.plot2D(select='mu',savefig='accretion_disk_mu')
system.plot2D(savefig='accretion_disk_image')


"""

+------------------------------------------------------+----------------------------------------------------+
| Effective temperature                                | Limb angles                                        |
+------------------------------------------------------+----------------------------------------------------+
| .. image:: images_tut/accretion_disk_teff.png        | .. image:: images_tut/accretion_disk_mu.png        |
|    :scale: 100 %                                     |    :scale: 100 %                                   |
|    :height: 233px                                    |    :height: 233px                                  |
|    :width: 600px                                     |    :width: 600px                                   |
+------------------------------------------------------+----------------------------------------------------+
| Radial velocity                                      | Image                                              |
+------------------------------------------------------+----------------------------------------------------+
| .. image:: images_tut/accretion_disk_rv.png          | .. image:: images_tut/accretion_disk_image.png     |
|    :scale: 100 %                                     |    :scale: 100 %                                   |
|    :height: 233px                                    |    :height: 233px                                  |
|    :width: 600px                                     |    :width: 600px                                   |
+------------------------------------------------------+----------------------------------------------------+

"""