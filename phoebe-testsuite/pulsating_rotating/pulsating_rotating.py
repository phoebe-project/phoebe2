"""
Pulsating rotating star (Ledoux splitting)
==========================================

Last updated: ***time***
    
Initialisation
--------------

"""
# First, import necessary modules
import time
import numpy as np
import matplotlib.pyplot as plt
import phoebe

logger = phoebe.get_basic_logger()

c0 = time.time()

# Parameter preparation
# ---------------------

# Create a ParameterSet with parameters matching 12 Lac.
lac = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
lac['teff'] = 25600.
lac['incl'] = 45.,'deg'
lac['radius'] = 8.8,'Rsol'
lac['mass'] = 14.4,'Msol'
lac['rotperiod'] = 2.23,'d'
lac['ld_func'] = 'claret'
lac['atm'] = 'kurucz_p00'
lac['ld_coeffs'] = 'kurucz_p00'
lac['shape'] = 'sphere'
lac['label'] = '12Lac'

mesh = phoebe.ParameterSet(context='mesh:marching')
mesh['delta'] = 0.2

# Create a ParameterSet with Parameters for the pulsation mode
freq_pars1 = phoebe.ParameterSet(frame='phoebe',context='puls',add_constraints=True)
freq_pars1['freq'] = 1.178964,'cy/d'
freq_pars1['ampl'] = 0.02
freq_pars1['l'] = 1
freq_pars1['m'] = 0
freq_pars1['k'] = 0.
freq_pars1['amplteff'] = 0.2
freq_pars1['scheme'] = 'coriolis'

freq_pars2 = freq_pars1.copy()
freq_pars2['m'] = 1
freq_pars2['ampl'] = 0.05

freq_pars3 = freq_pars2.copy()
freq_pars3['m'] = -1

# Create a ParameterSet with parameters for the light curve
lcdep1 = phoebe.ParameterSet(frame='phoebe',context='lcdep',ref='lc')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz_p00'
lcdep1['atm'] = 'kurucz_p00'

# Body setup
# ----------

star = phoebe.Star(lac,mesh,puls=[freq_pars1,freq_pars2,freq_pars3],pbdep=[lcdep1])

# Computation of observables
# --------------------------

n = 400        
times = np.linspace(0,17,n)*lac['rotperiod']
rotperiods = np.ones(n)
rotperiods[:n/2] = np.inf
rotperiods[n/2:] = lac['rotperiod']

for i,itime in enumerate(times):
    if i==0: continue
    print(i)
    star.reset()
    star.params['star']['rotperiod'] = rotperiods[i]
    star.set_time(itime,ref='lc')
    star.lc(time=itime)
    phoebe.image(star,select='teff',ref=0,context='lcdep',savefig='image_{:05d}.png'.format(i))
star.save('pulsating_rotating.phoebe')

"""

+------------------------------------------------------+
| Ledoux splitting                                     |
+------------------------------------------------------+
| .. image:: images_tut/movie_pulsating_rotating.gif   |
|    :scale: 100 %                                     |
+------------------------------------------------------+

"""