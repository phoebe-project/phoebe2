"""
Differential rotator
====================

Last updated: ***time***

In this example, we set a Vega-like star to have an equatorial rotation period
equal to about half the solid-body break-up velocity. Then, we set the polar
rotation period to be equal to around 100%, 75%, 50%, 25% and 15% of the
solid-body break-up period. We thus cover polar acceleration and polar
deceleration, as well as solid-body rotation. For each of these examples,
we compute an image, an effective temperature map and a radial velocity map.
Finally, the spectroscopic line profile is computed for each of them.

Initialisation
--------------

"""
# First, import necessary modules
import time
import numpy as np
from matplotlib import pyplot as plt
import phoebe

c0 = time.time()
logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------

# Create a ParameterSet with parameters closely matching Vega. We use a
# Kurucz atmosphere and set the rotation period to about 50% of the
# critical rotation period of Vega.
star = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
star['rotperiod'] = 22.8,'h'
star['teff'] = 8900.,'K'
star['mass'] = 2.3,'Msol'
star['gravb'] = 1.0
star['radius'] = 2.26,'Rsol'
star['incl'] = 90.,'deg'
star['atm'] = 'kurucz'
star['ld_coeffs'] = 'kurucz'
star['ld_func'] = 'claret'

# Remember that Phoebe expects a polar rotation period and a differential
# rotation period, such that ``diffrot=equatorial-polar``. In this example,
# we want to do the reverse and fix the equatorial rotation period and change
# the polar one. For our convenience, we add an extra parameter to the
# parameterSet and add a constraint.
star.add(dict(qualifier='eqrotperiod',value=22.8,unit='h',\
              description='Equatorial rotation period'))
star.add_constraint('{diffrot}={eqrotperiod}-{rotperiod}')

# For the mesh, we take a low-density grid, sampled using the marching method.
star_mesh = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
star_mesh['delta'] = 0.1

# In the spectrum that we calculate, we set the parameters for the atmosphere
# and limb-darkening coefficients as realistic as possible (i.e., Claret
# limbdarkening and Kurucz atmospheres). However, for the calculation of the
# spectrum, we require no knowledge of the the true line profile, but use a
# generic Gaussian line.
spdep = phoebe.ParameterSet(frame='phoebe',context='spdep')
spdep['ld_func'] = 'claret'
spdep['atm']= 'kurucz'
spdep['ld_coeffs'] = 'kurucz'
spdep['clambda'] = 4508.3
spdep['passband'] = 'JOHNSON.V'
spdep['method'] = 'numerical'
spdep['ref'] = 'Gaussian profile (numerical)'

# Body setup
# ----------
# Build the Star body.
mesh1 = phoebe.Star(star,star_mesh,pbdep=[spdep])
print mesh1

# Computation of observables
# --------------------------
polar_periods = [11.5,15.,22.8,45.,80.]
for i,polar_period in enumerate(polar_periods):
    star['rotperiod'] = polar_period,'h'
    mesh1.reset()
    mesh1.set_time(0,ref='all')
    
    # Compute the spectrum
    mesh1.sp()

    # and make an image of the star and a map of it's effective temperature.
    phoebe.image(mesh1)
    plt.xlim(-2.8,2.8)
    plt.ylim(-2.8,2.8)
    plt.savefig('diff_rotator_image_{}.png'.format(i))
    phoebe.image(mesh1,select='teff',savefig='diff_rotator_teff_{}.png'.format(i),vmin=8800,vmax=9400)
    phoebe.image(mesh1,select='rv',savefig='diff_rotator_rv_{}.png'.format(i),vmin=-100/8.05,vmax=100/8.05)

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

"""

# Analysis of the results
# -----------------------

# Retrieve all the spectra. We're too lazy to repeat the labels we gave above,
# but we want to use them to label the plot. We want the first, second and third
# spectrum calculated.

spectrum = mesh1.get_synthetic(type='spsyn',ref=0)

# A plot is then easily made:
waves = np.array(spectrum['wavelength'])
specs = np.array(spectrum['flux'])
conts = np.array(spectrum['continuum'])

plt.figure()
for i,(wave,spec,cont) in enumerate(zip(waves,specs,conts)):
    plt.plot(wave,spec/cont,'-',lw=2,label='P$_{{pl}}$={:.1f} h'.format(polar_periods[i]))
plt.xlabel('Wavelength [A]')
plt.ylabel("Normalised flux")
plt.grid()
leg = plt.legend()
leg.get_frame().set_alpha(0.5)
plt.savefig('diff_rotator_spectrum.png')
plt.gcf().frameon = False
plt.savefig('diff_rotator_spectrum.pdf')

"""
.. figure:: images_tut/diff_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral lines.

"""

print("Finished!")
print("Total time:                 {:10.3g} min".format((time.time()-c0)/60.))