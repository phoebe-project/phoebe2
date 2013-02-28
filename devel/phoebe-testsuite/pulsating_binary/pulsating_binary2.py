"""
Pulsating binary 2
==================

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
from phoebe.backend import observatory
from phoebe.utils import pergrams

logger = phoebe.get_basic_logger()
c0 = time.time()

# Parameter preparation
# ---------------------
incl = 75.,'deg'
long = 30.,'deg'

# Create a ParameterSet representing a pulsating star
pstar = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
pstar['teff'] = 10000.
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
mesh1['delta'] = 0.2

# Create a ParameterSet with Parameters for the pulsation mode
freq_pars1 = phoebe.ParameterSet(frame='phoebe',context='puls',add_constraints=True)
freq_pars1['freq'] = 21.2,'cy/d' # so that K0=0.1
freq_pars1['ampl'] = 0.015
freq_pars1['l'] = 4
freq_pars1['m'] = -3
freq_pars1['deltateff'] = 0.2
freq_pars1['ledoux_coeff'] = 0.5
freq_pars1['scheme'] = 'coriolis'

# Create a ParameterSet with parameters for the light curve
lcdep1 = phoebe.ParameterSet(frame='phoebe',context='lcdep',ref='Light curve')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz'
lcdep1['atm'] = 'kurucz'

# Create a ParameterSet with parameters of the binary system
orbit = phoebe.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,c1label=pstar['label'])
orbit['sma'] = 500.,'Rsol'
orbit['q'] = 1.0
orbit['ecc'] = 0.5
orbit['t0'] = 0.
orbit['period'] = 1.,'d'
orbit['incl'] = incl
orbit['long_an'] = long
orbit['per0'] = 10.,'deg'

# Body setup
# ----------

system = phoebe.BinaryStar(pstar,orbit,mesh1,puls=[freq_pars1],pbdep=[lcdep1])

# Computation of observables
# --------------------------
# Now calculate the light curves numerically and analytically:
    
P = orbit['period']
intimes = np.linspace(0,3*P,500)
mpi = phoebe.ParameterSet(context='mpi',np=8)
phoebe.observe(system,intimes,subdiv_num=0,lc=True,ltt=True,mpi=mpi)

# Analysis of results
# -------------------
c1 = time.time()
print system

#-- retrieve data: it consists of timestamps and flux calculations
times = np.array(system.params['syn']['lcsyn'].values()[0]['time'])
flux = np.array(system.params['syn']['lcsyn'].values()[0]['flux'])
    
#-- plot light curve: I ommit comments because frankly, it's just plotting
plt.figure()
plt.plot(times,flux,'k-',label='LTT')
plt.plot(intimes,flux,'r-',label='No LTT')
plt.xlim(times[0],times[-1])
plt.xlabel('Time [d]')
plt.ylabel("Flux [erg/s/cm2/AA]")
plt.grid()
plt.legend(loc='best').get_frame().set_alpha(0.5)
plt.savefig('pulsbin2_lc.png')
plt.gcf().frameon = False
plt.savefig('pulsbin2_lc.pdf')

plt.figure()
freqs1,ft1 = pergrams.DFTpower(times,(flux/flux.mean()-1)*100)
freqs2,ft2 = pergrams.DFTpower(intimes,(flux/flux.mean()-1)*100)
plt.plot(freqs1,np.sqrt(ft1),'k-',lw=2,label='LTT')
plt.plot(freqs2,np.sqrt(ft2),'r-',lw=2,label='No LTT')
plt.grid()
plt.xlim(16,26)
plt.ylabel("Amplitude [%]")
plt.xlabel("Frequency [d$^{-1}$]")
plt.legend(loc='best').get_frame().set_alpha(0.5)
plt.savefig('pulsbin2_freq.png')
plt.gcf().frameon = False
plt.savefig('pulsbin2_freq.pdf')


c2 = time.time()
print "Finished!"
print "Computation time:    %10.3f min"%((c1-c0)/60.)
print "Analysis of results: %10.3f sec"%((c2-c1))
print "-----------------------------------"
print "Total time:          %10.3f min"%((c2-c0)/60.)

plt.figure()
plt.plot(times-intimes,'ko-')
plt.show()