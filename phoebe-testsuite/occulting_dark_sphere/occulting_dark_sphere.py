"""
Occulting dark sphere (analytical transits)
===========================================

Compute the light curve of a dark sphere occulting a star. We compare the values
of the total projected intensity, calculated numerically and analytically. Note
that, if we would not need the analytical computation of the light curve, the
``universe.BinaryStar`` class is redundant, and could be replaced by first
creating a single ``universe.Star``, and placing that star in a
``universe.BodyBag``,specifying the binary parameters in the latter class
instance upon initalisation.

Setup:

* Kurucz atmosphere
* Claret's limbdarkening law
* Spherical stellar surfaces (i.e. not rotationally deformed)
* Intensities in JOHNSON.V band

Inspired on [MandelAgol2002]_


Initialisation
--------------
    
"""
# First, import the necessary modules
import time
import matplotlib.pyplot as plt
import numpy as np
import phoebe

logger = phoebe.get_basic_logger()

c0 = time.time()

# Parameter preparation
#----------------------

# Create a ParameterSet with parameters of the binary system. We add common
# default constraints, so that more information is accessible (such as the
# individual masses). Most of the default parameters are kept, except the
# value for the semi-major axis ``sma``.


# Next, create the parameters of the uniform source. This includes a parameter
# that sets the density of the mesh, and also the radius of the secondary
# sphere, because we will need that in the computation of the analytical flux.
# The second sphere will have a radius half of the primary.    
sphere1 = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
sphere1['radius'] = 1.,'Rsol'
sphere1['teff'] = 10000,'K'
sphere1['atm'] = 'kurucz'
sphere1['ld_coeffs'] = 'kurucz'
sphere1['rotperiod'] = 0.5,'d'
sphere1['shape'] = 'sphere'
sphere1['incl'] = 0.
sphere1.add_constraint('{radius2} = 0.5*{radius}')

mesh1 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh1['delta'] = 0.1

# To compute light curves, the parameters for to compute the light curve have
# to be given. We compute the light curve analytically and numerically, so two
# sets are necessary.
lcdep1a = phoebe.ParameterSet(frame='phoebe',context='lcdep')
lcdep1a['ld_func'] = 'claret'
lcdep1a['ld_coeffs'] = 'kurucz'
lcdep1a['atm'] = 'kurucz'
lcdep1a['ref'] = lcdep1a['method']
lcdep1b = lcdep1a.copy()
lcdep1b['method'] = 'analytical'
lcdep1b['ref'] = lcdep1b['method']

# The secondary sphere is a dark sphere. We set its temperature equal to zero
# Kelvin.
sphere2 = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
sphere2['radius'] = sphere1.get_constraint('radius2','Rsol'),'Rsol'
sphere2['teff'] = 0,'K'
sphere2['atm'] = 'blackbody'
sphere2['incl'] = 0.
sphere2['rotperiod'] = np.inf
sphere2.add_constraint('{radius2} = %.6g'%(sphere1.get_value('radius','m')))

mesh2 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh2['delta'] = 0.2

ps = phoebe.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,c1label=sphere1['label'],c2label=sphere2['label'])
ps['sma'] = 5.

# Again, we need parameters to compute the light curve.
lcdep2a = phoebe.ParameterSet(frame='phoebe',context='lcdep')
lcdep2a['ref'] = lcdep2a['method']
lcdep2b = lcdep2a.copy()
lcdep2b['method'] = 'analytical'
lcdep2b['ref'] = lcdep2b['method']

# Body setup
# ----------
# Initialise the two stars as a ``BinaryStar``.
star1 = phoebe.BinaryStar(sphere1,ps,mesh1,pbdep=[lcdep1a,lcdep1b])
star2 = phoebe.BinaryStar(sphere2,ps,mesh2,pbdep=[lcdep2a,lcdep2b])

# Computation of observables
# --------------------------
# Now calculate the light curves numerically and analytically. To do this, first
# define a time array, and then make the observables.
P = ps['period']
times = np.linspace(ps['t0']-0.075*P,ps['t0']+0.075*P,40)# was 40

bbag = phoebe.BodyBag([star1,star2])
phoebe.observe(bbag,times,subdiv_num=2,lc=True)

# Analysis of the results
# -----------------------
print(star1)

times = np.array(list(star1.params['syn']['lcsyn'].values())[0]['time'])
flux1 = np.array(list(star1.params['syn']['lcsyn'].values())[0]['flux'])
flux2 = np.array(list(star1.params['syn']['lcsyn'].values())[1]['flux'])

diffr = (flux1-flux2)
plt.figure()
plt.axes([0.1,0.3,0.85,0.65])
plt.plot(times,flux1,'ko-')
plt.plot(times,flux2,'r-',lw=2)
plt.xlabel("Time [JD]")
plt.ylabel("Flux [erg/s/cm2]")
plt.axes([0.1,0.1,0.85,0.2])
plt.plot(times,diffr,'ko-')
plt.xlabel("Time [JD]")
diffr = diffr/flux2*100.
plt.annotate(r'$\bar{f} = %.6f \pm %.6f\%%$'%(diffr.mean(),diffr.std()/np.sqrt(len(times))),(0.15,0.2),xycoords='figure fraction')
plt.annotate(r'$\epsilon(f) = %.6f\%%$'%(diffr.std()),(0.15,0.15),xycoords='figure fraction')
plt.savefig('ocds_lc.png')

"""
.. figure:: images_tut/ocds_lc.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Computed light curve.
"""


print("Finished!")
print(("Total time: {0:.3f} min".format((time.time()-c0)/60.)))
