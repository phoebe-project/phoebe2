'''
:download:`Download this page as a python script <../../devel/phoebe-testsuite/solar_calibration/solar_calibration.py>`

Solar calibration
=================

Last updated: 2015-02-24 10:18:44.147562

In this example, we evaluate the :index:`precision` and :index:`accuracy`
of the code, via
computation of the :index:`Solar bolometric luminosity <single: Sun; bolometric luminosity>`.
We not only compute the value of the total bolometric luminosity, but also
the :index:`projected intensity <single: Sun; projected intensity>`.
Finally, we make an image of the :index:`Sun <single: Sun; image>`, and make a
3D plot of the radial velocity of all surface elements. The precision and
accuracy are evaluated in the following way:

Accuracy
    The bolometric luminosity is computed *numerically*, and compared to the
    *analytically* computed value, but using the same treatment of the Solar
    atmosphere.

Precision
    The *numerically* computed value is compared to the *true* value (which
    we know because the Sun is so freaking close).
    
The Sun can be modelled in a kind-of realistic way assuming:

* It is a single star
* We can use Kurucz atmospheres
* The limb darkening is parametrizable via Claret's 4-parameter law
* It has a spherical surface (i.e. not rotationally deformed) 

Initialisation
--------------

First, we need to import the Phoebe namespace and create a basic logger
to log information to the screen.
'''

import numpy as np
import matplotlib.pyplot as plt
import phoebe
from phoebe.atmospheres.limbdark import sphere_intensity

logger = phoebe.get_basic_logger()
    
'''
Parameter preparation
---------------------
Starting with PHOEBE 2.0, the implementation of a shiney new front-end interface makes system set up much easier
than with previous iterations of PHOEBE. The front-end provides many pre-set Bundles from :py:mod:`<phoebe.frontend.bundle>`
instead of having to build a :ref:`Star <parlabel-phoebe-star>` Body from an empty ParameterSet. Since we are working with
the Sun in this example, we can use the preset bundle for the Sun and tweak a few parameters.
'''

sun = phoebe.Bundle('sun')

"""
With this, we've set up a single star body with a full attached ParamaterSet. To view the parameters we simply issue:
"""

print sun['star']

"""	
	Output:
		teff 5807.644175                	K - Effective temperature         
		radius 1.275827                 	Rsol - Radius                        
		mass 1.215673                  		Msol - Stellar mass                  
		atm kurucz                         	--   Bolometric Atmosphere model   
		rotperiod 5.377479                 	d - Polar rotation period         
		diffrot 0.0                        	d - (Eq - Polar) rotation period  
							    (<0 is solar-like)            
		gravb 1.0                         	-- - Bolometric gravity brightening
		gravblaw zeipel                   	--   Gravity brightening law       
		incl 90.0                         	deg - Inclination angle             
		long 0.0                          	deg - Orientation on the sky (East of North)                     
		t0 0.0                            	JD - Origin of time                
		shape equipot                     	--   Shape of surface              
		alb 0.0                           	-- - Bolometric albedo (1-alb      
							     heating, alb reflected)       
		redist 0.0                              -- - Global redist par (1-redist)  
							     local heating, redist global  
							     heating                       
		redisth 0.0                     	-- - Horizontal redist par         
							     (redisth/redist) horizontally 
							     spread                        
		irradiator True                 	--   Treat body as irradiator of   
							     other objects                 
		abun 0.0                        	--   Metallicity                   
		label G0V_95cacade-             	--   Name of the body              
		ld_func claret                  	--   Bolometric limb darkening     
							     model                         
		ld_coeffs kurucz                	--   Bolometric limb darkening     
							     coefficients   
	
To edit any of these parameters, we call the parameter as a python dictionary. For this instance, we would like to change 
the ``shape`` parameter to 'sphere' and the label to 'sun' for easy book keeping.
"""	

sun['shape'] = 'sphere'
#sun['label'] = 'sun'
	
"""
In addition to physical parameters, the Bundle also initiates numerical parameters used for building
the model grid. Since we wish to use the marching method to generate the grid we don't have to change anything
as it is set to as the default for ``mesh``.
"""

print sun['mesh:marching@Sun']
	
"""
	Output:
		delta 0.1     		--   Stepsize for mesh generation via marching method        
		maxpoints 100000	--   Maximum number of triangles for marching method         
		alg c      		--   Select type of algorithm 

We don't only need parameters for the Star body itself, but also information
on how we want to compute the observed quantities. In this example, we want
to compute an image. Thus, we need the same information as we would need to
compute a light curve (:ref:`lcdep <parlabel-phoebe-lcdep>`). We start by initializing 
two new lightcurves using :py:func:`lc_fromarrays <phoebe.frontend>`.
"""

sun.lc_fromarrays(time=np.linspace(0,10,201))
sun.lc_fromarrays(time=np.linspace(0,10,201))

"""
Now each of the new lightcurves we created have :ref:`lcdep <parlabel-phoebe-lcdep>`, :ref:`lcobs <parlabel-phoebe-lcobs>`, 
and :ref:`lcsyn <parlabel-phoebe-lcsyn>`.
"""

print sun.summary()

"""	
	Output:
		sun (Star)
		star
		mesh
		lcdep: lc01, lc02
		lcobs: lc01, lc02
		lcsyn: lc01, lc02

		* Compute: detailed, preview, legacy

:ref:`lcdep <parlabel-phoebe-lcdep>` contains the parameters used to compute the the observables for the lightcurve.
Again, most of the default parameters are just what we need, however we want to change ``passband`` to ``OPEN.BOL`` 
since we don't want to use a filter and we want to scale ``pblum`` to 4pi. 
"""

sun['passband@lc01@lcdep'] = 'OPEN.BOL'
sun['pblum@lc01@lcdep']    = 4*np.pi

"""
The same parameters have to be used for the analytical lightcurve, but of
course the ``method`` keyword has to be changed from its default value.
"""

sun['method@lc02@lcdep']   = 'analytical'
sun['pblum@lc02@lcdep']    = 4*np.pi
sun['passband@lc02@lcdep'] = 'OPEN.BOL'

"""
Great! Now our bodies are fully set up and tuned to our example, and ready to be passed to 
:py:func:`run_compute() <phoebe.backend.universe.Star.run_compute>`.

Computation of observables
--------------------------
There are three labels under which we can run compute ``preview``, ``detailed``, ``legacy``. For this tutorial, we will be using ``preview``, but 
before we run anything, we need to set :ref:`eclipse_alg <parlabel-phoebe-compute>`to None since we're dealing with a single object and not a binary. 
"""

sun['eclipse_alg@preview@compute'] = 'none'

"""
Now compute the observables.
"""

sun.run_compute('preview')

"""    
And finally make some check images using :py:class:`plot_mesh <phoebe.frontend.plotting>`:

* a simulated image of the Sun

* a "true color" image of the Sun

* a radial velocity map of the Sun
"""

sun.plot_mesh(dataref='lc01',select='proj2')
plt.savefig('sun_sun.png')
sun.plot_mesh(dataref='lc01',select='teff')
plt.savefig('sun_sun_true_color.png')
sun.plot_mesh(dataref='lc01',select='rv')
plt.savefig('sun_rv.png')

"""
+-----------------------------------+----------------------------------------------+-----------------------------------+
| Image                             | True Color Image                             | Radial velocity map               |
+-----------------------------------+----------------------------------------------+-----------------------------------+ 
| .. image:: images_tut/sun_sun.png | .. image:: images_tut/sun_sun_true_color.png | .. image:: images_tut/sun_rv.png  |
|    :width: 233px                  |    :width: 233px                             |    :width: 233px                  |
+-----------------------------------+----------------------------------------------+-----------------------------------+

Analysis of results
-------------------
"""


nflux = sun.get_value('flux@lc01@lcsyn')[0]
aflux = sun.get_value('flux@lc02@lcsyn')[0]

mupos = sun.get_mesh()['mu']>0.
area  = sun.get_system().area()

size  = sun.get_mesh()['size']	
lum2  = sun.get_system().get_parameters().get_value('luminosity','W') 

print("Analytical = %.6e W/m2"%(aflux))
print("Numerical  = %.6e W/m2"%(nflux)) 
print("Real       = %.6e W/m2"%(1368.000)) 
print("Numerical error on projected area:%.6f%%"% (np.abs(np.pi-((size*sun.get_mesh()['mu'])[mupos]).sum())/np.pi*100))
print("Numerical error on projected flux: %.6f%%"%(np.abs(nflux-aflux)/aflux*100))
print("Error on real projected flux (measure for accuracy):%.6f%%"%(np.abs(1368.000-aflux)/aflux*100))

print("\nTotal flux:")
lum1=sphere_intensity(sun['star'],sun.get_system().params['pbdep']['lcdep'].values()[0])[0]
lum2=lum2
lumsn = phoebe.constants.Lsol#_cgs 
print("Analytical = %.6e W"%(lum1)) 
print("Numerical = %.6e W"%(lum2))
print("Real = %.6e W"%(lumsn)) 
print("Numerical error on total area: %.6f%%"%(np.abs(4*np.pi-area)/4*np.pi*100))
print("Numerical error on total flux: %.6f%%"%(np.abs(lum1-lum2)/lum1*100)) 
print("Error on real total flux (measure for accuracy): %.6f%%"%(np.abs(lum1-lumsn)/lumsn*100))
print('Finished!')

"""
Code output

    Projected flux:
    Analytical = 1.364600e+06 erg/s/cm2
    Numerical  = 1.363938e+06 erg/s/cm2
    Real       = 1.368000e+06 erg/s/cm2
    Numerical error on projected area: 0.047775%
    Numerical error on projected flux: 0.048540%
    Error on real projected flux (measure for accuracy): 0.249138%
    
    Total flux:
    Analytical = 3.837657e+33 erg/s
    Numerical  = 3.839170e+33 erg/s
    Real       = 3.846000e+33 erg/s
    Numerical error on total area: 0.474328%
    Numerical error on total flux: 0.039424%
    Error on real total flux (measure for accuracy): 0.216930%
"""

