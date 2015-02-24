"""
Pulsating binary (eclipse mapping)
==================================

Compute the light curve of a pulsating star in a binary with a rapidly rotating
star.

Initialisation
--------------

"""
# First, import the necessary modules:
import time
from matplotlib import pyplot as plt
import numpy as np
import phoebe

logger = phoebe.get_basic_logger(clevel='DEBUG')
#c0 = time.time()

# Parameter preparation
# ---------------------
incl = 75.,'deg'
long = 30.,'deg'
    
# Create a ParameterSet representing a pulsating star
pstar = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True,
			    atm='kurucz_p00', ld_func='claret', ld_coeffs='kurucz_p00')
pstar['teff'] = 20000.
pstar['incl'] = incl
pstar['long'] = long
pstar['radius'] = 4.5,'Rsol'
pstar['mass'] = 3.0,'Msol'
pstar['rotperiod'] = 10.,'d'
pstar['shape'] = 'sphere'
pstar['atm'] = 'kurucz'
pstar['ld_coeffs'] = 'kurucz'
pstar['ld_func'] = 'claret'
pstar['label'] = 'Pulsating Star'
mesh1 = phoebe.ParameterSet(context='mesh:marching')

# Create a parameterSet representing a rapidly rotating star
rstar = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True,
			    atm='kurucz_p00', ld_func='claret', ld_coeffs='kurucz_p00')
rstar['rotperiod'] = 12.4,'h'
rstar['incl'] = incl
rstar['long'] = long
rstar['teff'] = 20000.,'K'
rstar['mass'] = 2.3,'Msol'
rstar['gravb'] = 1.0
rstar['radius'] = 2.26,'Rsol'
rstar['label'] = 'Rotating Star'

mesh2 = phoebe.ParameterSet(context='mesh:marching')

# Create a ParameterSet with parameters of the binary system
orbit = phoebe.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,c1label=pstar['label'],c2label=rstar['label'])
orbit['sma'] = 15.,'Rsol'
orbit['q'] = rstar['mass']/pstar['mass']
orbit['t0'] = 0.
orbit['period'] = 10.,'d'
orbit['incl'] = incl
orbit['long_an'] = long

star1 = phoebe.BinaryStar(pstar,  orbit=orbit, label='star1')
star2  = phoebe.BinaryStar(rstar, orbit=orbit, label='star2')

# Body setup
# ----------
# Build the two stars, and make each of them a binary BodyBag.
system = phoebe.BinaryBag([star1, star2], orbit=orbit, label='system')
bundle = phoebe.Bundle(system)

# Computation of observables
# add data
p = orbit['period']
time = np.linspace(0, 4*p , 1000)
bundle.lc_fromarrays(time=time)

#runs the computation
for eclipse_alg in ['binary']:#, 'graham', 'convex', 'full']:

  bundle['eclipse_alg@preview@compute']=eclipse_alg
  
  bundle['refl@preview@compute'] = False
  bundle['heating@preview@compute'] = False
  bundle.run_compute('preview')
  bundle.plot_syn('lc01', label='nothing')
  
  bundle['heating@preview@compute'] = True
  bundle.run_compute('preview')
  bundle.plot_syn('lc01', label='heating')
  
  bundle['refl@preview@compute'] = True
  bundle['refl_num@preview@compute'] = 3
  bundle.run_compute('preview')
  bundle.plot_syn('lc01', label='reflection')
  
  plt.legend(loc=3)
  plt.savefig("eclipse_"+eclipse_alg+".png", dpi=200)
  plt.close()



    