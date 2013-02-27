"""
Pulsating binary
================

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
mesh1['delta'] = 0.1

# Create a ParameterSet with Parameters for the pulsation mode
freq_pars1 = phoebe.ParameterSet(frame='phoebe',context='puls',add_constraints=True)
freq_pars1['freq'] = 1.5,'cy/d' # so that K0=0.1
freq_pars1['ampl'] = 0.01
freq_pars1['l'] = 4
freq_pars1['m'] = 3
freq_pars1['deltateff'] = 0.1
freq_pars1['ledoux_coeff'] = 0.5
freq_pars1['scheme'] = 'coriolis'

# Create a ParameterSet with parameters for the light curve
lcdep1 = phoebe.ParameterSet(frame='phoebe',context='lcdep',ref='Light curve')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz'
lcdep1['atm'] = 'kurucz'

# Create a parameterSet representing a rapidly rotating star
rstar = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
rstar['rotperiod'] = 12.4,'h'
rstar['incl'] = incl
rstar['long'] = long
rstar['teff'] = 9500.,'K'
rstar['mass'] = 2.3,'Msol'
rstar['gravb'] = 1.0
rstar['radius'] = 2.26,'Rsol'
rstar['label'] = 'Rotating Star'

mesh2 = phoebe.ParameterSet(context='mesh:marching')
mesh2['delta'] = 0.1

# Create a ParameterSet with parameters of the binary system
orbit = phoebe.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,c1label=pstar['label'],c2label=rstar['label'])
orbit['sma'] = 15.,'Rsol'
orbit['q'] = rstar['mass']/pstar['mass']
orbit['t0'] = 0.
orbit['period'] = 10.,'d'
orbit['incl'] = incl
orbit['long_an'] = long


pulsating_star = phoebe.BinaryStar(pstar,orbit,mesh1,puls=[freq_pars1],pbdep=[lcdep1])
rotating_star  = phoebe.BinaryStar(rstar,orbit,mesh2,pbdep=[lcdep1])

# Body setup
# ----------
# Build the two stars, and make each of them a binary BodyBag.
system = phoebe.BodyBag([pulsating_star,rotating_star])

# Computation of observables
# --------------------------
# Now calculate the light curves numerically and analytically:
    
P = orbit['period']
times = np.linspace(0,P,100)

extra_funcs = [observatory.ef_binary_image]*4
extra_funcs_kwargs = [dict(select='teff',cmap=plt.cm.spectral,name='pulsbin_teff',ref='Light curve'),
                      dict(select='rv',name='pulsbin_rv',ref='Light curve'),
                      dict(select='proj',name='pulsbin_proj',ref='Light curve'),
                      dict(select='teff',cmap='eye',name='pulsbin_bb',ref='Light curve')]

phoebe.observe(system,times,subdiv_num=3,lc=True,extra_func=extra_funcs,extra_func_kwargs=extra_funcs_kwargs)

# Analysis of results
# -------------------
c1 = time.time()
print system

#-- retrieve data: it consists of timestamps and flux calculations
times = np.array(pulsating_star.params['syn']['lcsyn'].values()[0]['time'])
flux = np.array(pulsating_star.params['syn']['lcsyn'].values()[0]['flux'])+\
        np.array(rotating_star.params['syn']['lcsyn'].values()[0]['flux'])

    
#-- plot light curve: I ommit comments because frankly, it's just plotting
pl.figure()
pl.plot(times,flux,'ko-')
pl.xlabel('Time [d]')
pl.savefig('pulsbin_lc.png')
    

"""
+------------------------------+----------------------------+
| Primary light curve          | Secondary light curve      |
+------------------------------+----------------------------+
| .. image:: pulsbin_proj.gif  | .. image:: pulsbin_teff.gif|
|    :scale: 100 %             |    :scale: 100 %           |
|    :height: 233px            |    :height: 233px          |
|    :width: 450px             |    :width: 450px           |
+------------------------------+----------------------------+
| Combined light curve         | Radial velocity curve      |
+------------------------------+----------------------------+
| .. image:: pulsbin_rv.gif    | .. image:: pulsbin_bb.gif  |
|    :scale: 100 %             |    :scale: 100 %           |
|    :height: 233px            |    :height: 233px          |
|    :width: 450px             |    :width: 450px           |
+------------------------------+----------------------------+


.. image:: pulsbin_lc.png
   :scale: 75 %
"""            
                        
c2 = time.time()
print "Finished!"
print "Computation time:    %10.3f min"%((c1-c0)/60.)
print "Analysis of results: %10.3f sec"%((c2-c1))
print "-----------------------------------"
print "Total time:          %10.3f min"%((c2-c0)/60.)

subprocess.call('convert -loop 0 pulsbin_bb*.png pulsbin_bb.gif',shell=True)
subprocess.call('convert -loop 0 pulsbin_rv*.png pulsbin_rv.gif',shell=True)
subprocess.call('convert -loop 0 pulsbin_teff*.png pulsbin_teff.gif',shell=True)    
subprocess.call('convert -loop 0 pulsbin_proj*.png pulsbin_proj.gif',shell=True)    