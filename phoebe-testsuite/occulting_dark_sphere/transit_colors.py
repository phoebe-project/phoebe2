"""
Occulting dark sphere (transit colors)
===========================================

Compute the light curve of a dark sphere occulting a star. Compare the shape
in the SDSS.GP and SDSS.RP band.

Setup:

* Kurucz atmosphere
* Claret's limbdarkening law
* Spherical stellar surfaces (i.e. not rotationally deformed)
* Intensities in SDSS.GP and SDSS.RP band


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
sphere1['teff'] = 5777.,'K'
sphere1['atm'] = 'kurucz'
sphere1['ld_func'] = 'claret'
sphere1['ld_coeffs'] = 'kurucz'
sphere1['rotperiod'] = 5.,'d'
sphere1['shape'] = 'equipot'
sphere1['incl'] = 0.

mesh1 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching',alg='c')
mesh1['delta'] = 0.05

# To compute light curves, the parameters for to compute the light curve have
# to be given. We compute the light curve analytically and numerically, so two
# sets are necessary.
lcdep1a = phoebe.ParameterSet(frame='phoebe',context='lcdep')
lcdep1a['ld_func'] = 'claret'
lcdep1a['ld_coeffs'] = 'kurucz'
lcdep1a['atm'] = 'kurucz'
lcdep1a['ref'] = 'SDSS.GP'
lcdep1a['passband'] = 'SDSS.GP'

lcdep1b = lcdep1a.copy()
lcdep1b['ref'] = 'SDSS.RP'
lcdep1b['passband'] = 'SDSS.RP'

# The secondary sphere is a dark sphere. We set its temperature equal to zero
# Kelvin.
sphere2 = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
sphere2['radius'] = 1,'Rjup'
sphere2['teff'] = 0,'K'
sphere2['atm'] = 'blackbody'
sphere2['incl'] = 0.
sphere2['rotperiod'] = np.inf
sphere2['ld_func'] = 'uniform'
sphere2['ld_coeffs'] = [1]
sphere2.add_constraint('{radius2} = %.6g'%(sphere1.get_value('radius','m')))

mesh2 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh2['delta'] = 0.1

ps = phoebe.ParameterSet(frame='phoebe',context='orbit',add_constraints=True,c1label=sphere1['label'],c2label=sphere2['label'])
ps['sma'] = 5.

# Body setup
# ----------
# Initialise the two stars as a ``BinaryStar``.
star1 = phoebe.BinaryStar(sphere1,ps,mesh1,pbdep=[lcdep1a,lcdep1b])
star2 = phoebe.BinaryStar(sphere2,ps,mesh2,pbdep=[lcdep1a,lcdep1b])

# Computation of observables
# --------------------------
# Now calculate the light curves numerically and analytically. To do this, first
# define a time array, and then make the observables.
P = ps['period']
times = np.linspace(ps['t0']-0.06*P,ps['t0']+0.06*P,40)# was 40

bbag = phoebe.BodyBag([star1,star2],solve_problems=True)
phoebe.observe(bbag,times,subdiv_num=2,lc=True)

bbag.save('transit_colors.phoebe')
# Analysis of the results
# -----------------------
print(star1)

times = np.array(list(star1.params['syn']['lcsyn'].values())[0]['time'])
flux1 = np.array(list(star1.params['syn']['lcsyn'].values())[0]['flux'])
flux2 = np.array(list(star1.params['syn']['lcsyn'].values())[1]['flux'])

diffr = (flux1/flux1[0]-flux2/flux2[0])
plt.figure()
plt.axes([0.1,0.3,0.85,0.65])
plt.plot(times,flux1/flux1[0],'go-',label='SDSS.G')
plt.plot(times,flux2/flux2[0],'ro-',label='SDSS.R')
plt.grid()
plt.legend().get_frame().set_alpha(0.5)
plt.xlabel("Time [JD]")
plt.ylabel("Relative flux")
plt.ylim(0.985,1.0)
plt.axes([0.1,0.1,0.85,0.2])
plt.plot(times,diffr*100,'ko-')
plt.xlabel("Time [JD]")
plt.ylabel("$\Delta$ [%]")
plt.savefig('transit_colors.png')

"""
.. figure:: images_tut/transit_colors.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Computed light curve.
"""


print("Finished!")
print(("Total time: {0:.3f} min".format((time.time()-c0)/60.)))
