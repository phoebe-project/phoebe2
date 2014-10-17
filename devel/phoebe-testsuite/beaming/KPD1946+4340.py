"""
KPD1946+4340 (boosting)
==================================

Compute the light curve of a boosting binary, with and without boosting.

This example is only a crude approximation, we use no dedicated WD or sdB
atmosphere grids but approximate with a blackbody.

Inspired on [Bloemen2012]_.

Initialisation
--------------

# First, import necessary modules
"""
import matplotlib.pyplot as plt
import numpy as np
import phoebe
from phoebe.backend import office
from phoebe.parameters import create
import time

logger = phoebe.get_basic_logger()

# Parameter preparation and Body setup
# ------------------------------------

# Create the system: we'll use black bodies to compute the light curves just
# in case the Kurucz grid does not cover a wide enough parameter space.
# First the case with doppler boosting:
comp1, comp2, orbit = create.from_library('KPD1946+4340_bis')
mesh = phoebe.PS('mesh:marching')
lcdep1 = phoebe.PS('lcdep', atm=comp1['atm'], ld_coeffs=comp1['ld_coeffs'],
                  ld_func=comp1['ld_func'], passband='KEPLER.V', boosting=True)
lcdep2 = phoebe.PS('lcdep', atm=comp2['atm'], ld_coeffs=comp2['ld_coeffs'],
                  ld_func=comp2['ld_func'], passband='KEPLER.V', boosting=True)

star1 = phoebe.BinaryRocheStar(comp1, mesh=mesh, orbit=orbit, pbdep=[lcdep1])
star2 = phoebe.BinaryRocheStar(comp2, mesh=mesh, orbit=orbit, pbdep=[lcdep2])
system1 = phoebe.BodyBag([star1, star2])

# Then without doppler boosting:
comp1, comp2, orbit = create.from_library('KPD1946+4340_bis')
lcdep1 = phoebe.PS('lcdep', atm=comp1['atm'], ld_coeffs=comp1['ld_coeffs'],
                  ld_func=comp1['ld_func'], passband='KEPLER.V', boosting=False)
lcdep2 = phoebe.PS('lcdep', atm=comp2['atm'], ld_coeffs=comp2['ld_coeffs'],
                  ld_func=comp2['ld_func'], passband='KEPLER.V', boosting=False)
star1 = phoebe.BinaryRocheStar(comp1, mesh=mesh, orbit=orbit, pbdep=[lcdep1])
star2 = phoebe.BinaryRocheStar(comp2, mesh=mesh, orbit=orbit, pbdep=[lcdep2])
system2 = phoebe.BodyBag([star1, star2])

# Computation of observables
# --------------------------
# Generate a time series to simulate the light curve and images on, and compute
# them:
period = system1[0].params['orbit']['period']
t0 = system1[0].params['orbit']['t0']
times = np.linspace(t0,t0+period,150)

phoebe.observe(system1,times,lc=True,heating=False,refl=False,
               eclipse_alg='binary',animate=office.Animation1(system1, select='teff'))#,mpi=mpi)
phoebe.observe(system2,times,lc=True,heating=False,refl=False,eclipse_alg='binary')


# Analysis of results
# -------------------
lc1 = system1.get_synthetic(category='lc',cumulative=True)
lc2 = system2.get_synthetic(category='lc',cumulative=True)

times1 = np.array(lc1['time'])
times2 = np.array(lc2['time'])
flux1 = np.array(lc1['flux'])
flux2 = np.array(lc2['flux'])

plt.figure()
norm = flux1.mean()
plt.plot(times1,flux1/norm,'r-',lw=2,label='With boosting')
plt.plot(times2,flux2/norm,'k-',lw=2,label='Without boosting')
plt.xlabel('Time [JD]')
plt.ylabel('Flux [erg/s/cm2/AA]')
plt.legend(loc='best')
plt.grid()
plt.savefig('KPD1946_boosting.png')
plt.gcf().frameon = False
plt.savefig('KPD1946_boosting.pdf')

"""

.. figure:: images_tut/KPD1946_boosting.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Light curve of KPD1946+4340 with and without doppler boosting.
   
"""
