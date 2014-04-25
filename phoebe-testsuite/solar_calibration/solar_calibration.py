"""
Solar calibration
=================

Last updated: ***time***

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

"""
# First, we need to import the Phoebe namespace and create a basic logger
# to log information to the screen.
import time
import numpy as np
import phoebe
from phoebe.atmospheres import limbdark
from phoebe.parameters import tools

logger = phoebe.get_basic_logger()

c0 = time.time()

# Parameter preparation
# ---------------------

# Most objects we place in the :py:mod:`universe <phoebe.backend.universe>`, need some set of parameters. Therefore,
# we start with the default ParameterSet linked to a specific object. In the
# case of the Sun, which we want to model as a single star, we need the default
# ParamaterSet of the :ref:`Star <parlabel-phoebe-star>` body. We're lucky, since most of the default
# parameters are set to represent the Sun, so we don't need to set the
# effective temperature, radius etc.
# Extra parameters that we set here concern the shape of the surface (``shape``),
# the distance to the Sun from the Earth (``distance``), the type of atmosphere
# to use for the emergent fluxes (``atm``) and limb darkening coefficients.
sun = phoebe.ParameterSet(context='star')
sun['shape'] = 'sphere'
sun['atm'] = 'kurucz'
sun['ld_coeffs'] = 'kurucz'
sun['ld_func'] = 'claret'

globs = phoebe.ParameterSet(context='position')
globs['distance'] = 1.0, 'au'

# Aside from physical parameters, we also need numerical ones. We wish to use
# the marching method to generate the grid, with a reasonably fine marching step.
sun_mesh = phoebe.ParameterSet(context='mesh:marching')
sun_mesh['delta'] = 0.05

# We don't only need parameters for the Star body itself, but also information
# on how we want to compute the observed quantities. In this example, we want
# to compute an image. Thus, we need the same information as we would need to
# compute a light curve (:ref:`lcdep <parlabel-phoebe-lcdep>`). Again we start from a default set of parameters, but
# change the limbdarkening model (``ld_func``). Additionally, we want to the
# program to look up values for the limb darkening coefficients instead of
# setting them manually (``ld_coeffs``). We don't want to use a filter, so we set the
# ``passband`` to ``OPEN.BOL`` (bolometric).
lcdep1 = phoebe.ParameterSet(frame='phoebe', context='lcdep')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz'
lcdep1['atm'] = 'kurucz'
lcdep1['passband'] = 'OPEN.BOL'
lcdep1['ref'] = 'Bolometric (numerical)'

# The same parameters have to be used for the analytical lightcurve, but of
# course the ``method`` keyword has to be changed from its default value.
lcdep2 = lcdep1.copy()
lcdep2['method'] = 'analytical'
lcdep2['ref'] = 'Bolometric (analytical)'

# Body setup
# ----------
# Next, a Body needs to be created in the Universe. The Body best representing
# the Sun is of course a single :py:class:`Star <phoebe.backend.universe.Star>`.
# The parameters for the *pbdeps* need
# to be passed along. The parameters of the Body itself can always be accessed
# via ``the_sun.params['star']``, the parameters of the *pbdeps* can be
# accessed via ``the_sun.params['pbdep']``, which is an ordered dictionary.
# Thus, the light curves can be accessed via
# ``the_sun.params['pbdep']['lcdep'].values()[0]``, and the numerical light
# curve on its turn via ``the_sun.params['pbdep']['lcdep'].values()[1]``.

the_sun = phoebe.Star(sun, mesh=sun_mesh, pbdep=[lcdep1, lcdep2], position=globs)

# Computation of observables
# --------------------------
# Upon creation, the Body does not yet exist in the Universe. We need to set the
# time of the object via :py:func:`set_time() <phoebe.backend.universe.Star.set_time>`, so that it knows where and in what orientiation it needs
# to put itself. For a single, non-rotating star this is easy; it will be put
# in the origin of the Universe but inclined with respect to the line of sight.
# The value of the time is in this case unimportant, since it is
# time-independent.
the_sun.set_time(0)

# Now compute the observables.
the_sun.lc()

"""
And finally make some check images using :py:class:`plot2D() <phoebe.backend.universe.Body.plot2D>`:

* a simulated image of the Sun

* a "true color" image of the Sun

* a radial velocity map of the Sun
"""
the_sun.plot2D(savefig='sun_sun.png',ref=0)
the_sun.plot2D(savefig='sun_sun_true_color.png',select='teff',cmap='blackbody_proj')
the_sun.plot2D(savefig='sun_rv.png',select='rv')

"""

+-----------------------------------+----------------------------------------------+-----------------------------------+
| Image                             | True Color Image                             | Radial velocity map               |
+-----------------------------------+----------------------------------------------+-----------------------------------+ 
| .. image:: images_tut/sun_sun.png | .. image:: images_tut/sun_sun_true_color.png | .. image:: images_tut/sun_rv.png  |
|    :width: 233px                  |    :width: 233px                             |    :width: 233px                  |
+-----------------------------------+----------------------------------------------+-----------------------------------+

"""

# Analysis of results
# -------------------
print(the_sun)

params = the_sun.get_parameters()
print("\nProjected flux:")
nflux = the_sun.params['syn']['lcsyn']['Bolometric (numerical)']['flux'][0]
aflux = the_sun.params['syn']['lcsyn']['Bolometric (analytical)']['flux'][0]
mupos = the_sun.mesh['mu']>0

print("Analytical = %.6e W/m2"%(aflux))
print("Numerical  = %.6e W/m2"%(nflux))
print("Real       = %.6e W/m2"%(1368.000))
print("Numerical error on projected area: %.6f%%"%(np.abs(np.pi-((the_sun.mesh['size']*the_sun.mesh['mu'])[mupos]).sum())/np.pi*100))
print("Numerical error on projected flux: %.6f%%"%(np.abs(nflux-aflux)/aflux*100))
print("Error on real projected flux (measure for accuracy): %.6f%%"%(np.abs(1368.000-aflux)/aflux*100))

print("\nTotal flux:")
lumi1 = limbdark.sphere_intensity(the_sun.params['star'],the_sun.params['pbdep']['lcdep'].values()[0])[0]
lumi2 = params.get_value('luminosity','W')
lumsn = phoebe.constants.Lsol#_cgs
print("Analytical = %.6e W"%(lumi1))
print("Numerical  = %.6e W"%(lumi2))
print("Real       = %.6e W"%(lumsn))
print("Numerical error on total area: %.6f%%"%(np.abs(4*np.pi-the_sun.area())/4*np.pi*100))
print("Numerical error on total flux: %.6f%%"%(np.abs(lumi1-lumi2)/lumi1*100))
print("Error on real total flux (measure for accuracy): %.6f%%"%(np.abs(lumi1-lumsn)/lumsn*100))

print('Finished!')
print('Total execution time: %.3g min'%((time.time()-c0)/60.))

"""
Code output::
 
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