"""
Misaligned binary
=================

In this example we show the influence of the misalignment parameter :math:`\theta`,
and compare light curves computed with different misalignment parameters. Also
the radial velocity map is shown, to show the direction of the motions.

Initialisation
--------------

"""

# First, import necessary modules and set up a logger.

import phoebe
from phoebe.parameters import tools
from phoebe.backend import plotting
from phoebe.utils import plotlib
import numpy as np
import matplotlib.pyplot as plt
import itertools

logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------

# Let us start from a massive and cool star, and create a binary from them with
# an orbital period of 1 day.

star1 = phoebe.create.star_from_spectral_type('B9V',atm='blackbody',ld_coeffs=[0.5],ld_func='linear')
star2 = phoebe.create.star_from_spectral_type('M1V',atm='blackbody',ld_coeffs=[0.5],ld_func='linear')
comp1,comp2,orbit = phoebe.create.binary_from_stars(star1,star2,period=(1.,'d'))

# To make the orbit misaligned, we need to add a few extra parameters: these are
# the parameters ``theta``, which acts like an inclination, the parameter
# ``\phi``, which acts like a phase angle, and the parameter ``precperiod``.
# If you set the latter to infinity (default), there is no precession. Positive
# precession means prograde precession, otherwise it is retrograde. Notice that
# although the geometry is the same for :math:`\theta=0` and :math:`\theta=180`,
# the first implies a prograde rotation, the second a retrograde.

tools.make_misaligned(orbit,theta=0.0,phi0=90.0,precperiod=np.inf)

# Define the parameters to compute light curve and the mesh

lcdep = phoebe.ParameterSet('lcdep',atm='blackbody',ld_coeffs=[0.5],ld_func='linear')
mesh = phoebe.ParameterSet('mesh:marching',delta=0.05,alg='python')

# We only consider the primary to be misaligned. For reference, we also make a
# version of the primary that *is* aligned.

star1a = phoebe.MisalignedBinaryRocheStar(comp1,mesh=mesh,orbit=orbit,pbdep=[lcdep])
star1b = phoebe.BinaryRocheStar(comp1,mesh=mesh,orbit=orbit,pbdep=[lcdep])
star2 = phoebe.BinaryRocheStar(comp2,mesh=mesh,orbit=orbit,pbdep=[lcdep])

# Thus we have two systems, the first with a misaligned primary, the second
# with two stars that are fully aligned.

system1 = phoebe.BodyBag([star1a,star2])
system2 = phoebe.BodyBag([star1b,star2])

# We want to observe the star during one full orbital period
times = np.linspace(0., orbit['period'], 100)[:-1]

# Observe the reference system, compute the light curve and make some check images.
phoebe.observe(system2,times,lc=True,extra_func=[phoebe.observatory.ef_binary_image,
                                                 phoebe.observatory.ef_binary_image],
                                     extra_func_kwargs=[dict(select='rv',name='rv00',),
                                                        dict(select='teff',name='teff00',vmin=10000.,vmax=11300.)])

# Keep track of the computed light curves in a nice figure:
colors = itertools.cycle([plt.cm.spectral(i) for i in np.linspace(0,1,5)])
plotting.plot_lcsyn(system2,'x-',color=colors.next(),ms=10,mew=2,scale=None,label='Aligned')
plt.xlabel("Time [d]")
plt.ylabel("Flux [erg/s/cm2/AA]")
plt.xlim(times[0],times[-1])
plt.savefig('light_curves')

# And make movies from the check images:
plotlib.make_movie('rv00*.png',fps=12,output='rv_aligned.gif',cleanup=True)
plotlib.make_movie('teff00*.png',fps=12,output='teff_aligned.gif',cleanup=True)

#Now repeat the previous procedure for various misalignment parameters.
for theta in np.arange(0,181,45):
    orbit['theta'] = theta
    system1.reset_and_clear()
    phoebe.observe(system1,times,lc=True,extra_func=[phoebe.observatory.ef_binary_image,
                                                     phoebe.observatory.ef_binary_image],
                                         extra_func_kwargs=[dict(select='rv',name='theta_{:03.0f}deg_rv'.format(theta)),
                                                            dict(select='teff',name='theta_{:03.0f}deg_teff'.format(theta),vmin=10000.,vmax=11300.)])
    
    plotting.plot_lcsyn(system1,'o-',color=colors.next(),label=r'$\theta$={:.0f}'.format(theta),scale=None)
    plt.legend(loc='best',prop=dict(size=10)).get_frame().set_alpha(0.5)
    plt.xlim(times[0],times[-1])
    plt.savefig('light_curves')
    
    code = 'theta_{:03.0f}deg_rv'.format(theta)
    plotlib.make_movie(code+'*.png',fps=12,output=code+'.gif',cleanup=True)
    code = 'theta_{:03.0f}deg_teff'.format(theta)
    plotlib.make_movie(code+'*.png',fps=12,output=code+'.gif',cleanup=True)

# Now repeat the previous procedure for a different phase angle, but first restart
# the figure:

plt.close()
plt.figure()
colors.next()
plt.xlabel("Time [d]")
plt.ylabel("Flux [erg/s/cm2/AA]")
plt.xlim(times[0],times[-1])


orbit['phi0'] = 0.
for theta in np.arange(0,181,45.):
    orbit['theta'] = theta
    system1.reset_and_clear()
    phoebe.observe(system1,times,lc=True,extra_func=[phoebe.observatory.ef_binary_image],
                                         extra_func_kwargs=[dict(select='teff',name='theta_{:03.0f}deg_phi_000deg_teff'.format(theta),vmin=10000.,vmax=11300.)])
    
    plotting.plot_lcsyn(system1,'o-',color=colors.next(),label=r'$\theta$={:.0f}'.format(theta),scale=None)
    plt.legend(loc='best',prop=dict(size=10)).get_frame().set_alpha(0.5)
    plt.xlim(times[0],times[-1])
    plt.savefig('light_curves_phi0_000')
    
    code = 'theta_{:03.0f}deg_phi_000deg_teff'.format(theta)
    plotlib.make_movie(code+'*.png',fps=12,output=code+'.gif',cleanup=True)
    
"""

+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| Radial velocity maps, :math:`\phi_0=90^\circ`   | Effective temperature maps, :math:`\phi_0=90^\circ`  | Effective temperature maps, :math:`\phi_0=0^\circ`          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/rv_aligned.gif            | .. image:: images_tut/teff_aligned.gif               | .. image:: images_tut/teff_aligned.gif                      |
|    :height: 150 px                              |    :height: 150 px                                   |    :height: 150 px                                          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/theta_000deg_rv.gif       | .. image:: images_tut/theta_000deg_teff.gif          | .. image:: images_tut/theta_000deg_phi_000deg_teff.gif      |
|    :height: 150 px                              |    :height: 150 px                                   |    :height: 150 px                                          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/theta_045deg_rv.gif       | .. image:: images_tut/theta_045deg_teff.gif          | .. image:: images_tut/theta_045deg_phi_000deg_teff.gif      |
|    :height: 150 px                              |    :height: 150 px                                   |    :height: 150 px                                          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/theta_090deg_rv.gif       | .. image:: images_tut/theta_090deg_teff.gif          | .. image:: images_tut/theta_090deg_phi_000deg_teff.gif      |
|    :height: 150 px                              |    :height: 150 px                                   |    :height: 150 px                                          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/theta_135deg_rv.gif       | .. image:: images_tut/theta_135deg_teff.gif          | .. image:: images_tut/theta_135deg_phi_000deg_teff.gif      |
|    :height: 150 px                              |    :height: 150 px                                   |    :height: 150 px                                          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/theta_180deg_rv.gif       | .. image:: images_tut/theta_180deg_teff.gif          | .. image:: images_tut/theta_180deg_phi_000deg_teff.gif      |
|    :height: 150 px                              |    :height: 150 px                                   |    :height: 150 px                                          |
+-------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------+


+----------------------------------------------------------------+----------------------------------------------------------------+
| .. image:: images_tut/light_curves.png                         | .. image:: images_tut/light_curves_phi0_000.png                |
|    :scale: 75 %                                                |    :scale: 75 %                                                |
+----------------------------------------------------------------+----------------------------------------------------------------+

"""


# Finally we make two more movies: for each phase angle explored above, we now
# also set the precession period to be equal to the orbital period.

orbit['phi0'] = 90,'deg'
orbit['theta'] = 45.,'deg'
orbit['precperiod'] = orbit['period']

system1.reset_and_clear()
phoebe.observe(system1,times,lc=True,extra_func=[phoebe.observatory.ef_binary_image],
                                     extra_func_kwargs=[dict(select='teff',name='theta_045deg_prec_phi_090deg_teff',vmin=10000.,vmax=11300.)])
plotlib.make_movie('theta_045deg_prec_phi_090deg_teff*.png',fps=12,output='theta_045deg_prec_phi_090deg_teff.gif',cleanup=True)

plt.close()
plt.figure()
plt.xlabel("Time [d]")
plt.ylabel("Flux [erg/s/cm2/AA]")    
plotting.plot_lcsyn(system2,'ko-',label=r'Aligned',scale=None)
plotting.plot_lcsyn(system1,'ro-',label=r'$\phi_0$=90',scale=None)

# Now for the second phase angle:

orbit['phi0'] = 0,'deg'
orbit['theta'] = 45.,'deg'
orbit['precperiod'] = orbit['period']

system1.reset_and_clear()
phoebe.observe(system1,times,lc=True,extra_func=[phoebe.observatory.ef_binary_image],
                                     extra_func_kwargs=[dict(select='teff',name='theta_045deg_prec_phi_000deg_teff',vmin=10000.,vmax=11300.)])
plotlib.make_movie('theta_045deg_prec_phi_000deg_teff*.png',fps=12,output='theta_045deg_prec_phi_000deg_teff.gif',cleanup=True)

plotting.plot_lcsyn(system1,'bo-',label=r'$\phi_0$=0',scale=None)

plt.legend(loc='best',prop=dict(size=10)).get_frame().set_alpha(0.5)
plt.xlim(times[0],times[-1])
plt.savefig('light_curves_prec')



"""

+---------------------------------------------------------------------------+-------------------------------------------------------------+
| Effective temperature map, :math:`\phi_0=90^\circ`                        | Effective temperature map, :math:`\phi_0=0^\circ`           |
+---------------------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images_tut/theta_045deg_prec_phi_090deg_teff.gif               | .. image:: images_tut/theta_045deg_prec_phi_000deg_teff.gif |
|    :height: 150 px                                                        |    :height: 150 px                                          |
+---------------------------------------------------------------------------+-------------------------------------------------------------+


.. figure:: images_tut/light_curves_prec.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Light curves with :math:`\theta=45^\circ`, different phase angles and a
   precession period equal to the orbital period.

"""
