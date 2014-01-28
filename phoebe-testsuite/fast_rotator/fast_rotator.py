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
star = phoebe.ParameterSet(context='star')
star['atm'] = 'blackbody'
star['ld_func'] = 'linear'
star['ld_coeffs'] = [0.6]
star['rotperiod'] = 0.24,'d'
star['shape'] = 'sphere'
star['teff'] = 10000.
star['radius'] = 1.0,'Rsol'

mesh = phoebe.ParameterSet(context='mesh:marching')
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
spobs1 = phoebe.SPDataSet(ref=spdep1['ref'], wavelength=wavelengths, time=[0])
spobs2 = phoebe.SPDataSet(ref=spdep2['ref'], wavelength=wavelengths, time=[0])

# Body setup
# ----------
# Build the :py:class:`Star <phoebe.backend.universe.Star>` body.
body = phoebe.Star(star, mesh, pbdep=[spdep1, spdep2], obs=[spobs1, spobs2])

# Computation of observables
# --------------------------

# Since observations are added, the system is smart enough to know what to compute:

body.compute()

# But if you now want to make an image, you need to specify that you want it
# to be with intensities computed for the spectra; since no light curves were
# added.

body.plot2D(ref='Numerical', context='spdep', savefig='fast_rotator_image.png')
body.plot2D(select='rv',savefig='fast_rotator_rv.png')
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

phoebe.plotting.plot_spsyn_as_profile(body, 'k-', ref='Numerical')
phoebe.plotting.plot_spsyn_as_profile(body, 'r--', ref='Via convolution')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Normalized flux')
plt.legend(loc='best')
plt.savefig('fast_rotator_spectrum')
plt.show()

"""
.. figure:: images_tut/fast_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral line.

"""

print 'Finished!'
print 'Total execution time: %.3g sec'%((time.time()-c0))