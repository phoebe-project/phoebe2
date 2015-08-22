'''
Fast rotating star (rotational convolution)
===========================================

Last updated: ***time***

The following example shows how to compute the total intensity and the
broadening of a spectral line due to the rotation. We compare this with direct
convolution of a spectral line with a rotational broadening kernel. Thus, we
do **not** take gravity darkening or Roche deformation into account. Though
rotating, the star is assumed to be still spherically symmetric. The parameters
of the star are conveniently chosen to approximate the Sun.

The following setup is used:

* spherical surface (i.e. not Roche-deformed)
* linear limbdarkening law
* Black body atmosphere

Initialisation
--------------

First, import the necessary modules and create a logger to show messages
to the screen:
'''

import numpy as np
from matplotlib import pyplot as plt
import phoebe
    
logger = phoebe.get_basic_logger()
    

'''
Parameter preparation
---------------------
Create a Bundle (**ADD HyperRef Later**) with parameters
matching the Sun, but make a fine-enough mesh. Also, the rotation period is
set to almost 90% of the Sun's critical rotation period.
'''

star = phoebe.Bundle('G0V')
star['atm'] = 'blackbody'
star['ld_func'] = 'linear'
star['ld_coeffs'] = [0.6]
star['rotperiod'] = 0.24,'d'
star['teff'] = 10000.
star['radius'] = 1.0,'Rsol'

star['delta@mesh'] = 0.05

'''
These are the parameters for the calculation of the :ref:`spectrum <parlabel-phoebe-spdep>`. We assume the
spectral line is in the visual region, and use a linear limb darkening law and initialize it 
with sp_fromarrays (**ADD HyperRef Later**) such that we now have a set of wavelengths 
that we can compute the spectrum for.
'''

wvl  = np.linspace(399.7,401.3,1000)
star.sp_fromarrays(wavelength=wvl,time=[0])
star.sp_fromarrays(wavelength=wvl,time=[0])

    
'''    
It is worth while to note that while we set the :ref:`atmosphere <parlabel-phoebe-atm>` model at the Bundle
level, it is also possible to set the :ref:`atmosphere <parlabel-phoebe-atm>` model, :ref:`ld_func <parlabel-phoebe-ld_func>`,
:ref:`ld_coeffs <parlabel-phoebe-ld_coeffs>` at level of each individual data set as follows:
'''

star['atm@sp01@spdep'] = 'blackbody'
star['ld_coeffs@sp01@spdep'] = [0.6]

star['atm@sp02@spdep'] = 'blackbody'
star['ld_coeffs@sp02@spdep'] = [0.6]

'''
For this tutorial, we want to compare the numerical computation with analytical computation.
'''


star['method@sp02@spdep'] = 'analytical'

'''
Computation of observables
--------------------------
With the two different data sets ready to be computed, we only have to run the computations. The Bundle has three computation settings:
**Preview**, **Detailed**, **Legacy**, but for our purposes we only need **Preview**. Before running the computations, we need to 
set the eclipsing algorithm to 'none' else we confuse our computations.
'''

star['eclipse_alg@preview@compute']='none'
star.run_compute('preview')
    
'''
But if you now want to make an image, you need to specify that you want it
to be with intensities computed for the spectra; since no light curves were
added.
'''

plt.figure(1)
star.plot_mesh(dataref='sp01',select='teff')
plt.savefig('fast_rotator_image.png')

plt.figure(10)
star.plot_mesh(dataref='sp01',select='rv')
plt.savefig('fast_rotator_rv.png')

'''
+----------------------------------------------+-------------------------------------------+
| Image                                        | Radial velocity map                       |
+----------------------------------------------+-------------------------------------------+
| .. image:: images_tut/fast_rotator_image.png | .. image:: images_tut/fast_rotator_rv.png |
|    :align: center                            |    :align: center                         |
|    :width: 233px                             |    :width: 233px                          |
+----------------------------------------------+-------------------------------------------+

Analysis of the results
-----------------------
'''
plt.figure(100)
star.plot_syn(dataref='sp01',time=0.0,fmt='k-',label='Numerical')
star.plot_syn(dataref='sp02',time=0.0,fmt='r--',label='Via convolution')
   
plt.xlabel('Wavelength [A]')
plt.ylabel('Normalized flux')
plt.legend(loc='best')
plt.savefig('fast_rotator_spectrum.png')
plt.show()
    
'''
.. figure:: images_tut/fast_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral line.
'''

print 'Finished!'
