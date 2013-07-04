"""
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

"""
# First, import the necessary modules and create a logger to show messages
# to the screen:
import time
import numpy as np
from matplotlib import pyplot as plt
import phoebe

logger = phoebe.get_basic_logger()

c0 = time.time()

# Parameter preparation
# ---------------------

# Create a :ref:`star ParameterSet <parlabel-phoebe-star>` with parameters
# matching the Sun, but make a fine-enough mesh. Also, the rotation period is
# set to almost 90% of the Sun's critical rotation period.
star = phoebe.ParameterSet(context='star',add_constraints=True)
star['atm'] = 'blackbody'
star['ld_func'] = 'linear'
star['ld_coeffs'] = [0.6]
star['rotperiod'] = 0.24,'d'
star['shape'] = 'sphere'
star['teff'] = 10000.
star['radius'] = 1.0,'Rsol'

mesh = phoebe.ParameterSet(context='mesh:marching',alg='c')
mesh['delta'] = 0.05

# These are the parameters for the calculation of the :ref:`spectrum <parlabel-phoebe-spdep>`. We assume the
# spectral line is in the visual region, and use a linear limb darkening law.
spdep1 = phoebe.ParameterSet(context='spdep')
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

# Fake some observations, so that we know onto which wavelengths we need to
# compute the spectrum
wavelengths = np.linspace(399.7, 400.3, 1000)
spobs1 = phoebe.ParameterSet(context='spobs', ref=spdep1['ref'], wavelength=wavelengths)
spobs2 = phoebe.ParameterSet(context='spobs', ref=spdep2['ref'], wavelength=wavelengths)

# Body setup
# ----------
# Build the :py:class:`Star <phoebe.backend.universe.Star>` body.
mesh1 = phoebe.Star(star, mesh, pbdep=[spdep1, spdep2])
print mesh1

# Computation of observables
# --------------------------
# Prepare to store the results and set the time in the universe.
mesh1.set_time(0)

# Compute the spectrum, and make an image of the star. Also, make a plot of
# the radial velocity of the surface elements.
mesh1.sp(obs=spobs1)
mesh1.sp(obs=spobs2)

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

result1 = mesh1.get_synthetic(category='sp',ref=0)
result2 = mesh1.get_synthetic(category='sp',ref=1)

plt.figure()
plt.plot(result1['wavelength'][0],
        np.array(result1['flux'][0])/np.array(result1['continuum'][0]),
        'k-',label=result1['ref'])
plt.plot(result2['wavelength'][0],
        np.array(result2['flux'][0])/np.array(result2['continuum'][0]),'r--',lw=2,label=result2['ref'])
plt.xlabel('Wavelength [A]')
plt.ylabel('Normalized flux')
plt.legend(loc='best')
plt.savefig('fast_rotator_spectrum')

"""
.. figure:: images_tut/fast_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral line.

"""

print 'Finished!'
print 'Total execution time: %.3g sec'%((time.time()-c0))