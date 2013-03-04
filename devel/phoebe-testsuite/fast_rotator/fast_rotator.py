"""
Fast rotating star
==================

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

Total execution time (Dual Core Genuine Intel(R) CPU, T2600, 2.16GHz, 32-bit):

+-----------+------------+
|Total time |   22.8 sec |
+-----------+------------+

Initialisation
--------------

"""
# First, import the necessary modules
import time
import numpy as np
import pylab as pl
import phoebe

logger = phoebe.get_basic_logger()

c0 = time.time()

# Parameter preparation
# ---------------------

# Create a ParameterSet with parameters matching the Sun, but make a fine-enough
# mesh. Also, the rotation period is set to almost 90% of the Sun's critical
# rotation period.
star = phoebe.ParameterSet(frame='phoebe',context='star',add_constraints=True)
star['atm'] = 'blackbody'
star['ld_func'] = 'linear'
star['ld_coeffs'] = [0.6]
star['rotperiod'] = 0.24,'d'
star['shape'] = 'sphere'
star['teff'] = 10000.
star['radius'] = 1.0,'Rsol'

mesh = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
mesh['delta'] = 0.05

# These are the parameters for the calculation of the spectrum. We assume the
# spectral line is in the visual region, and use a linear limb darkening law.
spdep1 = phoebe.ParameterSet(frame='phoebe',context='spdep')
spdep1['ld_func'] = 'linear'
spdep1['atm'] = 'blackbody'
spdep1['ld_coeffs'] = [0.6]
spdep1['passband'] = 'JOHNSON.V'
spdep1['method'] = 'numerical'
spdep1['ref'] = 'Numerical'

# We compare the numerical computation with analytical computation.
spdep2 = spdep1.copy()
spdep2['method'] = 'analytical'
spdep2['ref'] = 'Via convolution'

# Body setup
# ----------
# Build the Star body.
mesh1 = phoebe.Star(star,mesh,pbdep=[spdep1,spdep2])
print mesh1

# Computation of observables
# --------------------------
# Prepare to store the results and set the time in the universe.
mesh1.set_time(0)

# Compute the spectrum, and make an image of the star. Also, make a plot of
# the radial velocity of the surface elements.
mesh1.sp()
mesh1.plot2D(savefig='fast_rotator_image.png')
mesh1.plot2D(select='rv',savefig='fast_rotator_rv.png')
"""

+----------------------------------------------+-------------------------------------------+
| Image                                        | Radial velocity map                       |
+----------------------------------------------+-------------------------------------------+
| .. image:: images_tut/fast_rotator_image.png | .. image:: images_tut/fast_rotator_rv.png |
|    :align: center                            |    :align: center                         |
|    :width: 233px                             |    :width: 233px                          |
+----------------------------------------------+-------------------------------------------+

"""

# Analysis of the results
# -----------------------

result1 = mesh1.get_synthetic(type='spsyn',ref=0)
result2 = mesh1.get_synthetic(type='spsyn',ref=1)

pl.figure()
pl.plot(result1['wavelength'][0],
        np.array(result1['flux'][0])/np.array(result1['continuum'][0]),
        'k-',label=result1['ref'])
pl.plot(result2['wavelength'][0],
        np.array(result2['flux'][0])/np.array(result2['continuum'][0]),'r--',lw=2,label=result2['ref'])
pl.xlabel('Wavelength [A]')
pl.ylabel('Normalized flux')
pl.legend(loc='best')
pl.savefig('fast_rotator_spectrum')

"""
.. figure:: images_tut/fast_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral line.

"""

print 'Finished!'
print 'Total execution time: %.3g sec'%((time.time()-c0))