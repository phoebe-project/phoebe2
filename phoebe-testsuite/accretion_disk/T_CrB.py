"""
T Coronae Borealis - The Blaze star
===================================

Compute the shadow of an accretion disk on a red giant irradiated by an extremely
hot white dwarf in the center of the disk.

Initialisation
--------------

"""

import matplotlib.pyplot as plt
from phoebe.parameters import create
from phoebe.backend import observatory
from phoebe.algorithms import reflection
import phoebe
import time

logger = phoebe.get_basic_logger()

c0 = time.time()

# Body setup
# ----------

# Load the system from the library. A BodyBag is returned, of which the first
# Body is the red giant, the second the white dwarf and the third the disk.
# We override some of the defaults concerning the atmosphere treatment just
# to make sure we are not going outside of the interpolated tables.
x = create.T_CrB(create_body=True,pbdep=['lcdep'],ld_coeffs=[1.0],ld_func='uniform',atm='blackbody')

# We improve the resolution of the giant a bit, to have a nicer shade effect
x[0].params['mesh']['delta'] = 0.07

# And raise the temperature of the white dwarf considerably to have a really
# obvous reflection effect
x[1].params['component']['teff'] = 10000000.,'K'

# Computation of observables
# --------------------------

# Set the time of the system to some nice phase amd compute the local intensities
# of all Bodies.
x.set_time(200.)
x.intensity(ref='all')

# Compute a "before" image of the system
observatory.image(x,select='teff',cmap='blackbody',savefig='T_CrB_before.png')

# Next, we need to compute the actual heating between all the bodies. By
# default, only the white dwarf is an irradiator, the red giant and accretion
# disk is irradiated by it. We only compute heating, no reflection (the albedo
# is unity anyway).
reflection.mutual_heating(*x.bodies,heating=True,reflection=False,third_bodies=x[2])
print "Done mutual heating"

# Make an image of the system, coloring the bodies with the blackbody
# temperatures
observatory.image(x,select='teff',cmap='blackbody',savefig='T_CrB_after.png')
print "Done image"

# The previous image looks cool, but physically unrealistic: we would expect
# convection to redistribute heat across the surface. Therefore, we set the
# global (and immediate) heat redistribution parameter of the the giant.
x.reset()
x[0].params['component']['redist'] = 0.9
x.set_time(200.)
x.intensity(ref='all')
reflection.mutual_heating(*x.bodies,heating=True,reflection=False,third_bodies=x[2])
observatory.image(x,select='teff',cmap='blackbody',savefig='T_CrB_redistr.png')

print time.time()-c0

"""
+----------------------------------------------+-----------------------------------------+-----------------------------------------+
| No reflection                                | With reflection                         | With heat redistribution                |
+----------------------------------------------+-----------------------------------------+-----------------------------------------+
| .. image:: images_tut/T_CrB_before.png       | .. image:: images_tut/T_CrB_after.png   | .. image:: images_tut/T_CrB_redistr.png |
|    :height: 233px                            |    :height: 233px                       |    :height: 233px                       |
|    :width: 233px                             |    :width: 233px                        |    :width: 233px                        |
+----------------------------------------------+-----------------------------------------+-----------------------------------------+
"""
