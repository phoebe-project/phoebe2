"""
Vega (interferometry)
======================

Last updated: ***time***

Compute Vega's bolometric luminosity numerically, and compare with the observed
value (see [Aufdenberg2006]_.

Compare the "true" Vega model with:

* the fast-rotating limbdarkened disk model from [Aufdenberg2006]_
* a non-rotating Claret-limbdarkening model
* a non-rotating uniform disk model

Input parameters:

* Kurucz atmospheres
* Roche deformed surface

We compare the values of the total projected intensity (calculated
numerically and the observed values). We make an image and a plot of the radial
velocities of all surface elements.

Inspired upon [Aufdenberg2006]_.
    
Initialisation
--------------

"""
# First, import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import phoebe

logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------

# Create a ParameterSet with parameters matching Vega. The bolometric atmosphere
# and limb darkening coefficients are set the Kurucz and Claret models.
vega = phoebe.ParameterSet(frame='phoebe', context='star', label='True Vega',
                           add_constraints=True)
vega['teff'] = 8900, 'K'
vega['mass'] = 2.3, 'Msol'
vega['radius'] = 2.26, 'Rsol'
vega['rotperiod'] = 12.5, 'h'
vega['gravb'] = 1.
vega['incl'] = 4.7, 'deg'
vega['ld_func'] = 'claret'
vega['atm'] = 'kurucz'
vega['ld_coeffs'] = 'kurucz'

globs = phoebe.ParameterSet('position')
globs['distance'] = 7.756, 'pc'

# We want to compute interferometric visibilities in the K band:
ifdep1 = phoebe.ParameterSet(frame='phoebe', context='ifdep', ref='Fast rotator')
ifdep1['ld_func'] = 'claret'
ifdep1['ld_coeffs'] = 'kurucz'
ifdep1['atm']= 'kurucz'
ifdep1['passband'] = '2MASS.KS'

mesh = phoebe.ParameterSet(frame='phoebe', context='mesh:marching', alg='c')
mesh['delta'] = 0.1

# Next, wee'll make a Star similar to Vega but that is not a rapid rotator:
# We'll make uniform disk model and power-law limbdarkened disk on a
# non-rotating star. Note that we scale the radius of the stars according to the
# fitted angular diameter of Aufdenberg. That is, we add a constraint on the
# parameter 'angdiam', so that the radius will change if we change the angular
# diameter. Then we set the angular diameter to the fitted one, and the radius
# will be adjusted accordingly.
vega2 = vega.copy()
vega2['rotperiod'] = np.inf
vega2['teff'] = 9700, 'K'
vega2['label'] = 'Non-rotating Vega (uniform disk)'

# 
theta2 = phoebe.Parameter(qualifier='angdiam', description='Angular diameter',
                          unit='mas', value=3.209)
vega2.add(theta2)
vega2.add_constraint('{{radius}}=0.5*{{angdiam}}*{}'.format(globs.get_value('distance','m')))

# 
ifdep2 = ifdep1.copy()
ifdep2['ld_func'] = 'linear'
ifdep2['ld_coeffs'] = [0.0]
ifdep2['ref'] = 'uniform LD'

# The next nonrotating model has an angular diameter of 3.345 mas, and a power-
# law limbdarkening.
vega3 = vega2.copy()
vega3['angdiam'] = 3.345, 'mas'
vega3['label'] = "Non-rotating Vega (power law)"
ifdep3 = ifdep1.copy()
ifdep3['ld_func'] = 'power'
ifdep3['ld_coeffs'] = [0.341]
ifdep3['ref'] = 'power-law LD'
ifdep3['atm'] = 'blackbody'

# The final nonrotating model has an angular diameter of 3.259 and normal
#  Claret limbdarkening.
vega4 = vega3.copy()
vega4['angdiam'] =3.259, 'mas'
vega4['label'] = 'Non-rotating Vega (Claret)'
ifdep4 = ifdep1.copy()
ifdep4['ld_func'] = 'claret'
ifdep4['ld_coeffs'] = 'kurucz'
ifdep4['ref'] = 'Claret LD'


# There is some literature data available, which we will use to compare the
# models with.
data_baselines = np.array([101.606,127.859,141.062,148.558,105.770,107.698,111.609,
                           115.384,118.799,141.448,142.615,144.744,167.221,226.407,
                           161.569,192.469,227.080,236.716,249.095,260.851,268.673,
                           258.773,268.295,273.279,210.699])
data_vis_square = np.array([21.1531, 6.2229, 2.6265, 1.3567,18.2301,16.7627,14.4223,
                            12.2229,10.3873, 2.6399, 2.3968, 2.0041, 0.1040, 1.2148,
                             0.2426, 0.5913, 1.1066, 1.1361, 0.9120, 0.6047, 0.5079,
                             0.5911, 0.4518, 0.3788, 0.9303])
data_evs_square = np.array([0.8846,0.2019,0.0742,0.0417,0.1976,0.1710,0.1493,
                            0.1336,0.1168,0.0741,0.0676,0.0591,0.0059,0.0521,
                            0.0194,0.0314,0.0670,0.0414,0.0344,0.0259,0.0238,
                            0.0427,0.0241,0.0199,0.0682])

ifobs_simul1 = phoebe.IFDataSet(time=np.zeros(100),
                          baseline=np.linspace(0,300,100))

# We need to copy the ifobs because we need to set the reference for each of
# them differently, since the ifdeps have different references.

ifobs_simul2 = ifobs_simul1.copy()
ifobs_simul3 = ifobs_simul1.copy()
ifobs_simul4 = ifobs_simul1.copy()
ifobs_simul1['ref'] = ifdep1['ref']
ifobs_simul2['ref'] = ifdep2['ref']
ifobs_simul3['ref'] = ifdep3['ref']
ifobs_simul4['ref'] = ifdep4['ref']

# Body setup
# ----------

# Build the Star Bodies
star1 = phoebe.Star(vega,  mesh, pbdep=ifdep1, obs=ifobs_simul1, position=globs)
star2 = phoebe.Star(vega2, mesh, pbdep=ifdep2, obs=ifobs_simul2, position=globs)
star3 = phoebe.Star(vega3, mesh, pbdep=ifdep3, obs=ifobs_simul3, position=globs)
star4 = phoebe.Star(vega4, mesh, pbdep=ifdep4, obs=ifobs_simul4, position=globs)

# For convenience, we put everything in a BodyBag. Then we can set the time of
# all bodies simultaneously and don't have to cycle through them.
system = phoebe.BodyBag([star1, star2, star3, star4])

# Computation of observables
# --------------------------

system.compute(eclipse_alg='only_horizon')

# Analysis of results
# -------------------
# Make some check images
phoebe.image(star1,savefig='vega_image1.png',ref=0,context='ifdep')
phoebe.image(star2,savefig='vega_image2.png',ref=0,context='ifdep')
phoebe.image(star3,savefig='vega_image3.png',ref=0,context='ifdep')
phoebe.image(star4,savefig='vega_image4.png',ref=0,context='ifdep')

"""
+-----------------------------------------+---------------------------------------+-----------------------------------------+---------------------------------------+
| Fast rotator (Claret)                   | Fast rotator (Aufdenberg)             | Slow rotator (uniform disk)             | Slow rotator (Claret)                 |
+-----------------------------------------+---------------------------------------+-----------------------------------------+---------------------------------------+
| .. image:: images_tut/vega_image1.png   | .. image:: images_tut/vega_image3.png | .. image:: images_tut/vega_image2.png   | .. image:: images_tut/vega_image4.png |
|    :width: 233px                        |    :width: 233px                      |    :width: 233px                        |    :width: 233px                      |                          
|    :height: 233px                       |    :height: 233px                     |    :height: 233px                       |    :height: 233px                     |                          
+-----------------------------------------+---------------------------------------+-----------------------------------------+---------------------------------------+

"""

#-- get extra observables that we can derive from the computed ones (equatorial
#   rotation velocity, surface gravitation difference between pole and equator,
#   temperature difference between pole and equator etc.)
for mesh in system.bodies:
    parameters = mesh.get_parameters()
    print parameters
    radi = np.sqrt((mesh.mesh['center'][:,0]**2+\
                    mesh.mesh['center'][:,1]**2+\
                    mesh.mesh['center'][:,2]**2))
    velo = np.sqrt((mesh.mesh['velo___bol_'][:,0]**2+\
                    mesh.mesh['velo___bol_'][:,1]**2+\
                    mesh.mesh['velo___bol_'][:,2]**2))
    print "\nRadii",radi.min(),radi.max()
    print "v_eq = %.3f km"%(velo.max()*8.04986111111111)
    print 'g_pole - g_eq = %.3f - %.3f [cm/s2]'%(mesh.mesh['logg'].max(),mesh.mesh['logg'].min())
    print 'T_pole - T_eq = %.3f - %.3fK\n\n'%(mesh.mesh['teff'].max(),mesh.mesh['teff'].min())


plt.figure(figsize=(10,6))

# Make the basic plot of the visibilities versus baseline

plt.subplot(121)
phoebe.plotting.plot_ifsyn(star1, '-', lw=2, x='baseline', y='vis2')
phoebe.plotting.plot_ifsyn(star2, '-', lw=2, x='baseline', y='vis2')
phoebe.plotting.plot_ifsyn(star3, '-', lw=2, x='baseline', y='vis2')
phoebe.plotting.plot_ifsyn(star4, '-', lw=2, x='baseline', y='vis2')
plt.errorbar(data_baselines, data_vis_square/100., yerr=data_evs_square/100.,
             fmt='ko')

# Take care of some figure decorations

plt.gca().set_yscale('log', nonposy='clip')
plt.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)
plt.xlabel('meter')
plt.ylabel('Squared Visibility')
plt.axis([0, 300, 0.0009, 1.01])

# Then plot the phases versus baseline

plt.subplot(122)
phoebe.plotting.plot_ifsyn(star1, 'o-', x='baseline', y='phase')
phoebe.plotting.plot_ifsyn(star2, 'o-', x='baseline', y='phase')
phoebe.plotting.plot_ifsyn(star3, 'o-', x='baseline', y='phase')
phoebe.plotting.plot_ifsyn(star4, 'o-', x='baseline', y='phase')

# And the figure decorations

plt.xlabel('meter')
plt.ylabel('Phase [rad]')
plt.axis([0, 300, 0, 2*np.pi])

plt.savefig('vega_details')

"""
.. figure:: images_tut/vega_details.png
   :align: center
   :width: 700px
   :alt: map to buried treasure

   Interferometric quantities of the Vega models.

"""

print 'Finished!'

