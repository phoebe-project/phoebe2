"""
:download:`Download this page as a python script <../../devel/phoebe-testsuite/differential_rotation/differential_rotation.frontend.py>`

Differential rotator
====================

Last updated: 2015-04-12 22:39:51.871761

In this example, we set a Vega-like star to have an equatorial rotation period
equal to about half the solid-body break-up velocity. Then, we set the polar
rotation period to be equal to around 100%, 75%, 50%, 25% and 15% of the
solid-body break-up period. We thus cover polar acceleration and polar
deceleration, as well as solid-body rotation. For each of these examples,
we compute an image, an effective temperature map and a radial velocity map.
Finally, the spectroscopic line profile is computed for each of them.

Initialisation
--------------

First, import necessary modules
"""

import numpy as np
from matplotlib import pyplot as plt
import phoebe

logger = phoebe.get_basic_logger()
    
"""
Parameter preparation
---------------------
Create a ParameterSet with parameters closely matching Vega. We use a
Kurucz atmosphere and set the rotation period to about 50% of the
critical rotation period of Vega.
"""

star = phoebe.Bundle('A2V')

star['rotperiod']  = 22.8,'h'
star['teff']       = 8900.,'K'
star['mass']       = 2.3,'Msol'
star['gravb']      = 1.0
star['radius']     = 2.26,'Rsol'
star['incl']       = 90.,'deg'
star['atm']        = 'kurucz'
star['ld_coeffs']  = 'kurucz'
star['ld_func']    = 'claret'
star['label@star'] = 'A2V'
    
"""
Remember that Phoebe expects a polar rotation period and a differential
rotation period, such that ``diffrot=equatorial-polar``. In this example,
we want to do the reverse and fix the equatorial rotation period and change
the polar one. For our convenience, we add an extra parameter to the
parameterSet and add a constraint.
"""

star.get_ps('star').add(dict(qualifier='eqrotperiod', value=22.8, unit='h', description='Equatorial rotation period'))
star.get_ps('star').add_constraint('{diffrot} = {eqrotperiod} - {rotperiod}')

"""
For the mesh, we use a low-density grid. We also must set the eclipse algorithm to ``none`` since
we are dealing with a single object.
"""

star['delta@mesh:marching'] = 0.02
star['eclipse_alg@detailed@compute']='none'
    
"""
In the spectrum that we calculate, we set the parameters for the atmosphere
and limb-darkening coefficients as realistic as possible (i.e., Claret
limb darkening and Kurucz atmospheres). However, for the calculation of the
spectrum, we require no knowledge of the the true line profile, but use a
generic Gaussian line.
"""

star.sp_fromarrays(wavelength=np.linspace(-3,3,1000)+4508.3,time=[0],dataref='Gaussian')

star['passband@A2V@spdep'] = 'JOHNSON.V'
star['method@A2V@spdep'] = 'numerical'

"""
Computation of observables and analysis of results
-----------------------------------------------------
Cycle over different polar periods, compute the spectra, make a few images
and plot the spectra
"""

polar_periods = [11.5,15.,22.8,45.,80.]
for i,polar_period in enumerate(polar_periods):
        
    # compute the spectra
    star['rotperiod'] = polar_period,'h'
    star.run_compute('detailed')
   
    # and make an image of the star and a map of its effective temperature and
    # radial velocity
    #star.plot_mesh(dataref='Gaussian',time=0.0,select='proj')
    #plt.savefig('diff_rotator_image_{}.png'.format(i))
    
    plt.figure(num=100)
    star.plot_mesh(dataref='Gaussian',time=0.0,select='teff',vmin=8000,vmax=9400,background='k')
    plt.savefig('diff_rotator_teff_{}.png'.format(i))
    #plt.show()

    plt.figure(110)
    star.plot_mesh(dataref='Gaussian',time=0.0,select='rv',vmin=-100,vmax=100,background='k')
    plt.savefig('diff_rotator_rv_{}.png'.format(i))

    #plt.show()

    # and plot the spectra
    plt.figure(120)
    star.plot_syn(dataref='Gaussian', time=0.0, fmt='-', lw=2)#, label='$P_\mathrm{{pol}} : P_\mathrm{{eq}} = {:.2f}$'.format(polar_periods[i]/22.8))
    #plt.show()

plt.figure(120)
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel("Normalized flux")
plt.grid()

leg = plt.legend(loc='lower right')
leg.get_frame().set_alpha(0.5)
plt.savefig('diff_rotator_spectrum.png')
plt.gcf().frameon = False
plt.savefig('diff_rotator_spectrum.pdf')
    

"""
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| Image                                             | Effective temperature map                        | Radial velocity map                              |
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| .. image:: images_tut/diff_rotator_image_0.png    | .. image:: images_tut/diff_rotator_teff_0.png    | .. image:: images_tut/diff_rotator_rv_0.png      |
|    :width: 266px                                  |    :width: 266px                                 |    :width: 266px                                 |
|    :height: 266px                                 |    :height: 266px                                |    :height: 266px                                |
|    :scale:  80 %                                  |    :scale: 80 %                                  |    :scale: 80 %                                  |
|    :align: center                                 |                                                  |                                                  |
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| .. image:: images_tut/diff_rotator_image_1.png    | .. image:: images_tut/diff_rotator_teff_1.png    | .. image:: images_tut/diff_rotator_rv_1.png      |
|    :width: 266px                                  |    :width: 266px                                 |    :width: 266px                                 |
|    :height: 266px                                 |    :height: 266px                                |    :height: 266px                                |
|    :scale:  80 %                                  |    :scale: 80 %                                  |    :scale: 80 %                                  |
|    :align: center                                 |                                                  |                                                  |
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| .. image:: images_tut/diff_rotator_image_2.png    | .. image:: images_tut/diff_rotator_teff_2.png    | .. image:: images_tut/diff_rotator_rv_2.png      |
|    :width: 266px                                  |    :width: 266px                                 |    :width: 266px                                 |
|    :height: 266px                                 |    :height: 266px                                |    :height: 266px                                |
|    :scale:  80 %                                  |    :scale: 80 %                                  |    :scale: 80 %                                  |
|    :align: center                                 |                                                  |                                                  |
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| .. image:: images_tut/diff_rotator_image_3.png    | .. image:: images_tut/diff_rotator_teff_3.png    | .. image:: images_tut/diff_rotator_rv_3.png      |
|    :width: 266px                                  |    :width: 266px                                 |    :width: 266px                                 |
|    :height: 266px                                 |    :height: 266px                                |    :height: 266px                                |
|    :scale:  80 %                                  |    :scale: 80 %                                  |    :scale: 80 %                                  |
|    :align: center                                 |                                                  |                                                  |
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| .. image:: images_tut/diff_rotator_image_4.png    | .. image:: images_tut/diff_rotator_teff_4.png    | .. image:: images_tut/diff_rotator_rv_4.png      |
|    :width: 266px                                  |    :width: 266px                                 |    :width: 266px                                 |
|    :height: 266px                                 |    :height: 266px                                |    :height: 266px                                |
|    :scale:  80 %                                  |    :scale: 80 %                                  |    :scale: 80 %                                  |
|    :align: center                                 |                                                  |                                                  |
+---------------------------------------------------+--------------------------------------------------+--------------------------------------------------+


.. figure:: images_tut/diff_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral lines.
"""


print("Finished!")
