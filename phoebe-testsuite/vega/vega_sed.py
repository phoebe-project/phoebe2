"""
Vega (SED fit)
====================================

We'll fit a fully rotationally distorted stellar model to multicolour photometry.

The parameters that we allow to be fitted are:

    * surface gravity (``surfgrav``)
    * radius (``radius``)
    * fraction rotation frequency (``rotfreqcrit``)
    * polar effective temperature (``teffpolar``)
    * inclination angle (``incl``)
    
The values from [Yoon2010]_ are::

    surfgrav = 3.655-4.021 [cm/s2]
    radius = 2.362 Rsol
    rotfreqcrit = 0.875
    teffpolar = 10059 K
    incl = 4.975 deg
    
The values derived from our fit are::

    surfgrav = 3.92 +/- 0.06 [cm/s2] (between 3.58-3.921)
    radius = 2.36 +/- 0.07 Rsol
    rotfreqcrit =  0.86 +/- 0.08
    teffpolar = 10127 +/- 255 K
    incl = 13.6 +/- 13.9 deg
    
And if we don't use Zeipel gravity darkening but Espinosa gd::

    surfgrav        = 3.94995 +/- 0.0743 [cm/s2] (between 3.58-3.95)
    radius          = 2.30837 +/- 0.114 Rsol
    rotfreqcrit     = 0.88015 +/- 0.101 
    teffpolar       = 10039.8 +/- 280 K
    incl            = 1.85536 +/- 9.85 deg

        
Note that not all parameters are good parameters to fit. For example, the
angular diameter is not well defined for a rotationally distorted star,
partially due to its shape, and partially due to its wavelength dependency.

We'll try to match the physics of Vega as closely as possible, i.e. Kurucz
atmosphere, Claret LD etc..., and choose a rough estimate of Vega's parameters
as a starting point for a Levenberg-Marquard optimization.

Initialization
--------------

"""

# First, import necessary modules
import matplotlib.pyplot as plt
import phoebe
from phoebe.parameters import tools
from phoebe.atmospheres import passbands

logger = phoebe.get_basic_logger()

# Parameter preparation
# ----------------------

# We load the parameters of Vega from the library, but add a few new ones
# which we like to fit.

star = phoebe.create.from_library('Vega', gravblaw='espinosa')
mesh = phoebe.PS(context='mesh:marching',alg='c',delta=0.1)

# The parameters that we want to fit but are not available, are not very
# exotic so they can be added via the :py:mod:`parameters.tools <phoebe.parameters.tools>`
# module

tools.add_surfgrav(star, 4.0, derive='mass')
tools.add_rotfreqcrit(star, 0.5)
tools.add_teffpolar(star, 9000.)

# Next, we tell to code to consider these parameters to be fitted.
star.set_adjust(('teffpolar','radius','surfgrav','incl','rotfreqcrit'),True)

# To be sure the code doesn't go of the grid, we define priors, which will be
# interpreted as boundaries for the Levenberg-Marquardt algorithm.
star.get_parameter('teffpolar').set_prior(distribution='uniform',lower=4500,upper=15000.)
star.get_parameter('surfgrav').set_prior(distribution='uniform',lower=3.,upper=5.0)
star.get_parameter('radius').set_prior(distribution='uniform',lower=1.5,upper=4.0)
star.get_parameter('incl').set_prior(distribution='uniform',lower=0.,upper=90.)
star.get_parameter('rotfreqcrit').set_prior(distribution='uniform',lower=0.,upper=0.99)

# Next up is to define the fitting parameters. We choose not compute detailed
# confidence intervals, since our experience is that ``lmfit`` fails most of
# the time (plus it's really slow). We specify that we want the fit to be
# bounded, and that we start from the initial values we gave (``iters=0``).
# If you set ``iters>0``, the code will do that many iterations of the fitting
# process, each time starting from a new, random starting point drawn from the
# priors.
fitparams = phoebe.PS(context='fitting:lmfit',method='leastsq',label='mynonlin',
                      compute_ci=False,bounded=True,iters=1)

# The last thing we need before building the star are the actual observations.
# The phot files also contain the required information for the lcdeps, so
# we don't need to define any extra parameters. You can view the contents of
# the phot file :download:`here <../../devel/phoebe-testsuite/vega/Vega.phot>`.
lcdats,lcdeps = phoebe.parse_phot('Vega.phot')
ifobs, ifdep = phoebe.parse_vis2('Vega.vis2')
ifobs.set_enabled(False)

# Body setup
# ----------
# All is left, is build the star!
mystar = phoebe.Star(star,mesh,pbdep=lcdeps+[ifdep],obs=lcdats+[ifobs])

# Fitting process
# ---------------
# Now we can run the Levenberg-Marquardt fitter, and accept the results
# immediately. We put a strange if statement here, only so that  we're able
# to quickly load a previous fit instead of refitting if we're only interested
# in checking the parameters or the model.
if True:
    fitparams = phoebe.run(mystar,fitparams=fitparams,accept=True)
    mystar.params['obs']['ifobs'].values()[0].set_enabled(True)
    mystar.compute()
    #mystar.save('vega_sed_{}.phoebe'.format(star['gravblaw']))
    #fitparams.save('fitparams_{}.phoebe'.format(star['gravblaw']))
else:
    mystar = phoebe.load_body('vega_sed_{}.phoebe'.format(star['gravblaw']))
    fitparams = phoebe.load_ps('fitparams_{}.phoebe'.format(star['gravblaw']))
    
# Print out the fit results:
fb = fitparams['feedback']
for val,sigma,par in zip(fb['values'],fb['sigmas'],fb['parameters']):
    if par.has_unit():
        unit = par.get_unit()
    else:
        unit = ''
    print('{:15s} = {:3g} +/- {:.3g} {}'.format(par.get_qualifier(),val,sigma,unit))

# Analysis
# ----------

# Make an SED plot of the observations. Since the observations don't contain
# the passband, we need to retrieve that information from the lcdep. And
# since passbands don't contain wavelength information, we need to extract
# that info from the :py:mod:`atmospheres.passbands <phoebe.atmospheres.passbands>`
# library.
plt.figure()
for j,ref in enumerate([ps['ref'] for ps in lcdeps]):
    dep,ref = mystar.get_parset(type='pbdep',ref=ref)
    syn,ref = mystar.get_parset(type='syn',ref=ref)
    obs,ref = mystar.get_parset(type='obs',ref=ref)
    
    wave = passbands.get_info([dep['passband']])['eff_wave']
    wave = list(wave)*len(obs['flux'])
    
    label = 'Observed' if j==0 else '__nolegend__'
    plt.errorbar(wave,obs['flux'],yerr=obs['sigma'],fmt='ko',label=label)
    label = 'Computed' if j==0 else '__nolegend__'
    plt.plot(wave,syn['flux'],'rx',mew=2,mec='r',ms=10,label=label)
    plt.legend(loc='best').get_frame().set_alpha(0.5)
    plt.grid()
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.xlabel("Wavelength [\AA]")
plt.ylabel("Flux [erg/s/cm$^2$/$\AA$]")
plt.xlim(1000,25000)
plt.ylim(2e-11,1e-8)
#plt.savefig('vega_sed_bestfit1_{}.png'.format(star['gravblaw']))
plt.xlim(1200,10000)
plt.ylim(8e-10,8e-9)
#plt.savefig('vega_sed_bestfit2_{}.png'.format(star['gravblaw']))

plt.figure()
phoebe.plotting.plot_ifobs(mystar, fmt='ko')
phoebe.plotting.plot_ifsyn(mystar, 'rx', ms=10, mew=2)

plt.show()
"""

+----------------------------------------------+---------------------------------------------+
| SED from UV to near infrared                 | Zoom in the UV-optical                      |
+----------------------------------------------+---------------------------------------------+
| .. image:: images_tut/vega_sed_bestfit1.png  | .. image:: images_tut/vega_sed_bestfit2.png |
|    :align: center                            |    :align: center                           |
|    :width: 233px                             |    :width: 233px                            |
+----------------------------------------------+---------------------------------------------+

"""