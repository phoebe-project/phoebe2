import phoebe
import numpy as np;
import matplotlib.pyplot as plt

logger = phoebe.utils.utils.get_basic_logger(clevel='DEBUG')

pstar1		= phoebe.ParameterSet(context="star", mass=1, radius=1, teff=30000, label='star1', atm='kurucz_p00', ld_func='claret', ld_coeffs='kurucz_p00')
pstar2		= phoebe.ParameterSet(context="star", mass=1, radius=1, teff=30000, label='star2', atm='kurucz_p00', ld_func='claret', ld_coeffs='kurucz_p00')
pposition	= phoebe.ParameterSet(context="position")			       
preddening	= phoebe.ParameterSet(context="reddening:interstellar")
pmesh		= phoebe.ParameterSet(context='mesh:marching')

comp1, comp2, orbit = phoebe.create.binary_from_stars(pstar1, pstar2, sma=(3, 'Rsol'));

star1 = phoebe.BinaryRocheStar(comp1, orbit=orbit);
star2 = phoebe.BinaryRocheStar(comp2, orbit=orbit);
system = phoebe.BinaryBag([star1, star2], orbit=orbit, reddening=preddening, position=pposition)

bundle = phoebe.Bundle(system)

p = orbit['period']
time = np.linspace(p, 4*p, 500)
bundle.lc_fromarrays(time=time)

for eclipse_alg in ['binary', 'graham', 'convex']:
  bundle['refl@preview@compute'] = False
  bundle['heating@preview@compute'] = False
  bundle['eclipse_alg@preview@compute']=eclipse_alg
  
  
  bundle.run_compute('preview')
  bundle.plot_syn('lc01', label='nothing')
  
  bundle['heating@preview@compute'] = True
  bundle.run_compute('preview')
  bundle.plot_syn('lc01', label='heating')
  
  bundle['refl@preview@compute'] = True
  bundle['refl_num@preview@compute'] = 3
  bundle.run_compute('preview')
  bundle.plot_syn('lc01', label='reflection')
  plt.legend()
  plt.savefig("eclipse_"+eclipse_alg+".png", dpi=200)
  plt.close()
#plt.show()