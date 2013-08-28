"""
Rotating pulsating star in the traditional approximation
========================================================

Last update: ***time***

Make a movie of a star pulsating with one mode in the traditional
approximation. 

We create a star similar to 12 Lac, let it rotate at 50 of the critical
rotation frequency, and assume it pulsates in an l=3,m=-1 mode. The values
for coefficients of the linear combination of the modes (the so called B-
vector) are calculated with the rotational nonadiabatic code MADROT (Dupret
et al. 2003) for a random stellar model (i.e. the coefficients are not
calculated for this stellar model), they only serve as an example.

The parameters for the mode are also greatly exaggerated for the purpose of
generating nice images rather than representing any physical reality. In
particular, the temperature amplitude is gigantic and only chosen such that
the intensity varies visibly over the surface.

Initialisation
--------------

"""
# Import necessary modules
import numpy as np
from matplotlib import pyplot as plt
import phoebe
from phoebe.utils import pergrams

logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------
# Create a ParameterSet with parameters matching 12 Lac.
lac = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
lac['teff'] = 25600.
lac['incl'] = 90.,'deg'
lac['radius'] = 8.8,'Rsol'
lac['mass'] = 14.4,'Msol'
lac['rotperiod'] = 2.925,'d'
lac['ld_func'] = 'claret'
lac['atm'] = 'kurucz'
lac['ld_coeffs'] = 'kurucz'
lac['shape'] = 'sphere'

mesh = phoebe.ParameterSet(context='mesh:marching',alg='c')
mesh['delta'] = 0.1

# Read in the bvector from a file. The file contains two columns, the degree
# l of the terms in the linear combinations, and their coefficients. Phoebe
# requires coefficients for all degrees, so we set the coefficients to zero
# if a certain degree is not present in the data file.
ls,Bs = np.loadtxt('bvector_rot50.dat').T
bvector = np.zeros(int(ls.max())+1)
bvector[np.array(ls,int)] = Bs

# Create a ParameterSet with Parameters for the pulsation mode
freq_pars = phoebe.ParameterSet(frame='phoebe',context='puls',add_constraints=True)
freq_pars['freq'] = 2.32464159,'cy/d'
freq_pars['ampl'] = 0.01
freq_pars['l'] =  3
freq_pars['m'] = -1
freq_pars['k'] =  0.
freq_pars['trad_coeffs'] = bvector
freq_pars['amplteff'] = 0.2
freq_pars['scheme'] = 'traditional approximation'
print freq_pars
raise SystemExit
# Create a ParameterSet with parameters for the light curve
lcdep1 = phoebe.ParameterSet(frame='phoebe',context='lcdep')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz'
lcdep1['atm'] = 'kurucz'

# Body setup
# ----------

star = phoebe.Star(lac,mesh,puls=[freq_pars],pbdep=[lcdep1])

# Computation of observables
# --------------------------
        
tn = 1./freq_pars['freq']
times = np.linspace(0,tn,100)
phoebe.observe(star,times,subdiv_num=0,lc=True, extra_func=[phoebe.observatory.ef_image],
               extra_func_kwargs=[dict(select='teff',ref=0)])

# Analysis of results
# -------------------
# Collect the times of observations and calculated flux.
print star
times = np.array(star.params['syn']['lcsyn'].values()[0]['time'])
flux = np.array(star.params['syn']['lcsyn'].values()[0]['flux'])

freqs,ft = pergrams.DFTpower(times,flux-flux.mean())

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(times,flux,'ko-')
plt.xlabel('Time [d]')
plt.ylabel('Amplitude')
plt.xlim(times[0],times[-1])
plt.subplot(122)
plt.plot(freqs,np.sqrt(ft),'k-')
plt.vlines([freq_pars['freq']],0,(flux-flux.mean()).max(),color='r',lw=2)
plt.ylim(0,0.7e-09)
plt.xlabel('Frequency [cy/d]')
plt.ylabel('Amplitude')
plt.savefig('puls_tradappr_freqspec')

"""
.. figure:: images_tut/tradapprox.gif
   :scale: 50 %
   :width: 266 px
   :height: 266 px
   :align: center
   :alt: map to buried treasure

   Light curve and frequency spectrum.
"""

"""
.. figure:: images_tut/puls_tradappr_freqspec.png
   :scale: 90 %
   :align: center
   :alt: map to buried treasure

   Light curve and frequency spectrum.
"""


subprocess.call('convert -loop 0 compute*.png tradapprox.gif',shell=True)
plt.show()