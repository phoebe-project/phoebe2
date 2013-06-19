"""
Venus (apparent magnitude)
==========================

Compute the visual magnitude of Venus taking into account the Sun's
reflection.

Initialisation
--------------

"""
# First, import necessary modules

import phoebe
from phoebe.parameters import create

logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------
#
# Load the Sun, Venus, and the orbit of Venus from the library. Make sure to
# set the Sun to be an irradiator, so that sunlight can be reflected of Venus'
# surface.

sun = create.from_library('sun')
sun['irradiator'] = True
venus,orbit = create.from_library('venus')
orbit['incl'] = 100.

# Make sure the mesh is reasonably fine, to make pretty pictures. We'll
# compute everything in the visual, so we also need to create light curve
# parameterSets.

mesh = phoebe.ParameterSet(context='mesh:marching', delta=0.05, alg='c')
lcdep1 = create.dep_from_object(sun,context='lcdep', passband='JOHNSON.V',
                                ref='Visual')
lcdep2 = create.dep_from_object(venus,context='lcdep', passband='JOHNSON.V',
                                ref='Visual')

# Body setup
# ----------

# Then the Sun and Venus are easily created as ``BinaryStars``.
bsun = phoebe.BinaryStar(sun, orbit, mesh, pbdep=lcdep1)
bvenus = phoebe.BinaryStar(venus, orbit, mesh, pbdep=lcdep2)
system = phoebe.BodyBag([bsun, bvenus])

# Computation of observables
# --------------------------

#observatory.compute(system,[0.35*orbit['period']],lc=True,reflection=True,heating=False,circular=False)
phoebe.observe(system,[0.41*orbit['period']],lc=True,refl=True,heating=False)

# Analysis of results
# -------------------

# We make some nice images of Venus and the SUn, and convert the computed
# intensity to Johnson magnitudes. For fun, we also check what the distance
# to the two bodies is.
bvenus.plot2D(ref='Visual',select='proj',savefig='venus_proj')
bvenus.plot2D(ref='Visual',select='teff',cmap='eye',savefig='venus_eye')
bsun.plot2D(ref='Visual',select='proj',savefig='sun_proj')
bsun.plot2D(ref='Visual',select='teff',cmap='eye',savefig='sun_eye')

osun = bsun.as_point_source()
ovenus = bvenus.as_point_source()
proj = osun['intensity']
print osun
print ovenus
vmag = phoebe.convert('erg/s/cm2/AA','mag',proj,passband='JOHNSON.V')
print("Apparent visual magnitude of the Sun = {:.3f}".format(vmag))
proj = ovenus['intensity']
vmag = phoebe.convert('erg/s/cm2/AA','mag',proj,passband='JOHNSON.V')
print("Apparent visual magnitude of Venus   = {:.3f}".format(vmag))
print("Distance to the Sun = {:.3f} au".format(osun.get_value('distance','au')))
print("Distance to Venus   = {:.3f} au".format(ovenus.get_value('distance','au')))


"""
This should give output like::
    
    Apparent visual magnitude of the Sun = -26.759
    Apparent visual magnitude of Venus   = -4.041   
    Distance to the Sun = 1.000 au
    Distance to Venus   = 1.547 au

+-----------------------------------------+---------------------------------------+-----------------------------------------+---------------------------------------+
| The Sun                                 | Venus                                 | The Sun                                 | Venus                                 |
+-----------------------------------------+---------------------------------------+-----------------------------------------+---------------------------------------+
| .. image:: images_tut/sun_eye.png       | .. image:: images_tut/venus_eye.png   | .. image:: images_tut/sun_proj.png      | .. image:: images_tut/venus_proj.png  |
|    :height: 233px                       |    :height: 233px                     |    :height: 233px                       |    :height: 233px                     |
|    :width: 233px                        |    :width: 233px                      |    :width: 233px                        |    :width: 233px                      |
+-----------------------------------------+---------------------------------------+-----------------------------------------+---------------------------------------+

"""
