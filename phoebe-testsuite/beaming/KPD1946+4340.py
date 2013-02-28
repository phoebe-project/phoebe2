"""
KPD1946+4340 - a beaming binary
===============================

Compute the light curve of a beaming binary, with and without beaming.

This example is only a crude approximation, we use no dedicated WD or sdB
atmosphere grids but approximate with a blackbody.

Initialisation
--------------

# First, import necessary modules
"""
import matplotlib.pyplot as plt
import numpy as np
import phoebe
from phoebe.parameters import create

logger = phoebe.get_basic_logger()

# Parameter preparation and Body setup
# ------------------------------------

# Create the system: we'll use black bodies to compute the light curves just
# in case the Kurucz grid does not cover a wide enough parameter space.
# First the case with doppler beaming:
system1 = create.from_library('KPD1946+4340',create_body=True,pbdep=['lcdep'],
         atm='true_blackbody',ld_func='linear',ld_coeffs=[0.5],beaming=True)
system1[0].params['mesh']['delta'] = 0.1
system1[1].params['mesh']['delta'] = 0.1

# Then without doppler beaming:
system2 = create.from_library('KPD1946+4340',create_body=True,pbdep=['lcdep'],
         atm='true_blackbody',ld_func='linear',ld_coeffs=[0.5],beaming=False)
system2[0].params['mesh']['delta'] = 0.1
system2[1].params['mesh']['delta'] = 0.1

# Computation of observables
# --------------------------
# Generate a time series to simulate the light curve and images on, and compute
# them:
period = system1[0].params['orbit']['period']
t0 = system1[0].params['orbit']['t0']
times = np.linspace(t0,t0+0.95*period,150)

mpi = phoebe.ParameterSet(context='mpi',np=8)
phoebe.observe(system1,times,lc=True,heating=False,refl=False,mpi=mpi)
phoebe.observe(system2,times,lc=True,heating=False,refl=False)

# Analysis of results
# -------------------
lc1 = system1.get_synthetic(type='lcsyn',cumulative=True)
lc2 = system2.get_synthetic(type='lcsyn',cumulative=True)

times1 = np.array(lc1['time'])
times2 = np.array(lc2['time'])
flux1 = np.array(lc1['flux'])
flux2 = np.array(lc2['flux'])

plt.figure()
norm = flux1.mean()
plt.plot(times1,flux1/norm,'r-',lw=2,label='With beaming')
plt.plot(times2,flux2/norm,'k-',lw=2,label='Without beaming')
plt.xlabel('Time [JD]')
plt.ylabel('Flux [erg/s/cm2/AA]')
plt.legend(loc='best')
plt.grid()
plt.savefig('KPD1946_beaming.png')
plt.gcf().frameon = False
plt.savefig('KPD1946_beaming.pdf')


"""
.. figure:: KPD1946_beaming.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Light curve of KPD1946+4340 with and without doppler beaming.
"""