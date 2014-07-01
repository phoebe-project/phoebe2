"""
Spotted star
============

Last updated: ***time***

Compute a light curve of a rotating star with one spot.

Initialisation
--------------

"""
# First, import the necessary modules.
import time
import numpy as np
from matplotlib import pyplot as plt
import phoebe
from phoebe.utils import plotlib

logger = phoebe.get_basic_logger()

c0 = time.time()

# Parameter preparation
# ---------------------
# Create a ParameterSet with parameters closely matching the Sun:
sun = phoebe.ParameterSet(context='star')
sun['atm'] = 'kurucz'
sun['incl'] = 90.,'deg'

mesh = phoebe.ParameterSet(context='mesh:marching')

# Definition of the spot:
spot = phoebe.ParameterSet(context='circ_spot')
spot['teffratio'] = 0.9
spot['long'] = 90.,'deg'
spot['colat'] = 45.,'deg'
spot['angrad'] = 10.

# To compute the light curve:
lcdep1 = phoebe.ParameterSet(context='lcdep')
lcdep1['ld_func'] = 'claret'
lcdep1['ld_coeffs'] = 'kurucz'
lcdep1['atm'] = 'kurucz'
lcdep1['ref'] = 'spot'

lcobs = phoebe.LCDataSet(time = np.linspace(0,sun['rotperiod'],50))

# Body setup
# ----------
# Build the Star body:
star = phoebe.Star(sun, mesh=mesh, pbdep=[lcdep1], circ_spot=spot, obs=[lcobs])
print(star)

# Computation of observables
# --------------------------

star.compute(extra_func=[phoebe.observatory.ef_image], extra_func_kwargs=[dict(ref='spot')])

"""
+----------------------------------------------------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
| .. image:: images_tut/spotted_star_0000003_142857_png.png      | .. image:: images_tut/spotted_star_0000005_387755_png.png    | .. image:: images_tut/spotted_star_0000008_979592_png.png           |
|    :width: 233px                                               |    :width: 233px                                             |    :width: 233px                                                    |
+----------------------------------------------------------------+--------------------------------------------------------------+---------------------------------------------------------------------+

.. image:: images_tut/spotted_star.gif
   :scale: 40%
   :width: 266 px
   :height: 266 px

"""            

# Analysis of results
# -------------------

plt.figure()
phoebe.plotting.plot_lcsyn(star, 'ko-')
plt.xlabel('Time [d]')
plt.ylabel('Relative flux')
plt.savefig('spotted_star_lc')

plotlib.make_movie('ef_image_????*.png',output='spotted_star.gif')
"""
.. figure:: images_tut/spotted_star_lc.png
   :scale: 75 %
   :align: center
   :alt: map to buried treasure

   Light curve

"""