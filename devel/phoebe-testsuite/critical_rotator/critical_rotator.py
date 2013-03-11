"""
Critical rotator
================

Last updated: ***time***

We calculate the total intensity and generate a spectrum of a critically
rotating (Roche deformed) star. The spectrum is compared to that obtained with
numerical convolution of a reference spectrum with a rotational broadening
kernel, and a linear limb darkening law. These profiles are then compared to a
simulation of a line profile using an grid of synthetic spectra. The influence
of neighbouring lines is then also taken into account.

The conclusion we can derive from this is that assuming a constant Gaussian line
profile is a very rude approximation. The shape of the line profile can be
quite different when the influence of neighbouring lines is taken into account.
In particular, the continuum is much different, because the continuum of a
rotating star contains then also smeared out neighbouring lines.

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
# Kurucz atmosphere and set the rotation period to about 93% of the critical
# rotation period of Vega.
star = phoebe.ParameterSet(context='star')
star['rotperiod'] = 11.6,'h'#11.5,'h'
star['teff'] = 8900.,'K'
star['mass'] = 2.3,'Msol'
star['gravb'] = 1.0
star['radius'] = 2.26,'Rsol'
star['ld_func'] = 'claret'
star['atm']= 'kurucz'
star['ld_coeffs'] = 'kurucz'


# For the mesh, we take a low-density grid, sampled using the marching method.
star_mesh = phoebe.ParameterSet(frame='phoebe',context='mesh:marching')
star_mesh['delta'] = 0.06

# In the first spectrum that we calculate, we set the parameters for the
# atmosphere and limb-darkening coefficients as realistic as possible (i.e., 
# Claret limbdarkening and Kurucz atmospheres). However, for the calculation of
# the spectrum, we require not knowledge of the the true line profile, but use
# a generic Gaussian line.
spdep1 = phoebe.ParameterSet(frame='phoebe',context='spdep')
spdep1['ld_func'] = 'claret'
spdep1['atm']= 'kurucz'
spdep1['ld_coeffs'] = 'kurucz'
spdep1['clambda'] = 4508.3,'AA'
spdep1['passband'] = 'JOHNSON.V'
spdep1['method'] = 'numerical'
spdep1['ref'] = 'Gaussian profile (numerical)'

# We want to compare numerical calculation with analytical approximation. In
# the analytical approximation, we need a linear limb darkening coefficient,
# but we still use Kurucz atmospheres.
spdep2 = spdep1.copy()
spdep2['ld_func'] = 'linear'
spdep2['ld_coeffs'] = [0.6]
spdep2['clambda'] = 4508.3,'AA'
spdep2['method'] = 'analytical'
spdep2['ref'] = 'Gaussian profile (analytical)'

# Finally we compute a spectrum using profiles from a grid. This will include
# Contamination from neighbouring lines!
spdep3 = spdep1.copy()
spdep3['profile'] = 'atlas'
spdep3['clambda'] = 4508.3,'AA'
spdep3['ref'] = 'Synthesized profile'

# Body setup
# ----------
# Build the Star body.
mesh1 = phoebe.Star(star,star_mesh,pbdep=[spdep1,spdep2,spdep3])
print mesh1

# Computation of observables
# --------------------------
mesh1.set_time(0,ref='all')

# Compute the spectrum
mesh1.sp()

# and make an image of the star and a map of it's effective temperature.
phoebe.image(mesh1,savefig='critical_rotator_image.png')
phoebe.image(mesh1,select='teff',savefig='critical_rotator_teff.png')

# For fun, set the inclination to 60 degrees and again make an image and a map
# of the star's effective temperature. We need to recalculate the mesh, so
# we need to call the mesh's ``reset`` method.
mesh1.reset()
mesh1.params['star']['incl'] = 60.,'deg'
mesh1.set_time(0,ref='all')
phoebe.image(mesh1,savefig='critical_rotator_image2.png')
phoebe.image(mesh1,select='teff',savefig='critical_rotator_teff2.png')

"""

+---------------------------------------------------+--------------------------------------------------+---------------------------------------------------+--------------------------------------------------+
| Image                                             | Effective temperature map                        | Image                                             | Effective temperature map                        |
+---------------------------------------------------+--------------------------------------------------+---------------------------------------------------+--------------------------------------------------+
| .. image:: images_tut/critical_rotator_image.png  | .. image:: images_tut/critical_rotator_teff.png  | .. image:: images_tut/critical_rotator_image2.png | .. image:: images_tut/critical_rotator_teff2.png |
|    :width: 233px                                  |    :width: 233px                                 |    :width: 233px                                  |    :width: 233px                                 |
+---------------------------------------------------+--------------------------------------------------+---------------------------------------------------+--------------------------------------------------+


"""

# Analysis of the results
# -----------------------

# Retrieve all the spectra. We're too lazy to repeat the labels we gave above,
# but we want to use them to label the plot. We want the first, second and third
# spectrum calculated.

result1 = mesh1.get_synthetic(type='spsyn',ref=0)
result2 = mesh1.get_synthetic(type='spsyn',ref=1)
result3 = mesh1.get_synthetic(type='spsyn',ref=2)

# A plot is then easily made:
wave1 = np.array(result1['wavelength'])
spec1 = np.array(result1['flux'])
cont1 = np.array(result1['continuum'])
wave2 = np.array(result2['wavelength'])
spec2 = np.array(result2['flux'])
cont2 = np.array(result2['continuum'])
wave3 = np.array(result3['wavelength'])
spec3 = np.array(result3['flux'])
cont3 = np.array(result3['continuum'])


plt.figure()
plt.plot(wave1[0],spec1[0]/cont1[0],'k-',lw=2,label=result1['ref'])
plt.plot(wave2[0],spec2[0]/cont2[0],'r-',lw=2,label=result2['ref'])
plt.plot(wave3[0],spec3[0]/cont3[0],'g-',lw=2,label=result3['ref'])
plt.xlabel('Wavelength [A]')
plt.ylabel("Normalised flux")
plt.grid()
leg = plt.legend()
leg.get_frame().set_alpha(0.5)
plt.savefig('critical_rotator_spectrum.png')

"""
.. figure:: images_tut/critical_rotator_spectrum.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Generated spectral line.

"""
plt.show()
print("Finished!")
print("Total time:                 {:10.3g} min".format((time.time()-c0)/60.))