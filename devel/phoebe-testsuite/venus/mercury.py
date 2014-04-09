"""
Mercury (heat transport)
==========================

In this example, Mercury is used as a starting point to show the effect of
various heat transport mechanisms on the heat distribution of a planet.

We'll let the Sun irradiate Mercury. We set it's albedo to 0.932, which means
that most of the light that enters the Mercurian 'atmosphere' is used to heat
the planet. The rest is reflected.

This 93.2% of the incoming radiation is distributed over the surface of the
planet in a variety of ways. First, half of that light is used to locally heat
the planet (``redist=0.5``), i.e. the hemisphere pointed towards the Sun. Next,
45% of the heating radiation (90% of 50%, ``redisth=0.9``) is distributed only
the longitudinal direction, effectively creating colatitudes of nearly constant
temperature. The remaining 5% is used to globally heat the entire planet.

To make the influence of the various processes more clear, we incline Mercury
slightly such that the pole is not directly heated by the Sun. Since longitudinal
heating cannot reach the pole, the polar temperature is therefore only set by the
global heat redistribution parameter.

We (fairly arbitrarily) set the temperature of Mercury to 100K. So if there would
be no heating, 100K would be temperature of the entire planet. If all radiation
would be globally distributed, Mercury would have a uniform temperature of about
400 K (that's its equilibrium temperature). If all radiation would be used to
locally heat the planet, the subsolar point would amount to about 700K. That
gives you an idea of the strength of the radiation.

Initialisation
-----------------

Let's get to work: First, we import the necessary modules, and set up a logger.

"""

import phoebe
import matplotlib.pyplot as plt
import numpy as np

logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------

# Next, we define the parameters of the Sun and Mercury.

sun = phoebe.create.from_library('sun', irradiator=True)


mercury = phoebe.PS('star', teff=100., atm='blackbody', radius=(0.3829,'Rearth'),
                mass=(0.055,'Mearth'), ld_func='uniform', shape='sphere',
                rotperiod=(58.646,'d'),alb=0.932, label='mercury', redist=0.5,
                redisth=0.9, incl=70., long=(-10.,'deg'))

orbit = phoebe.PS('orbit', period=87.9691, t0=0, t0type='superior conjunction',
                sma=(0.307499,'au'), q=mercury['mass']/sun['mass'], ecc=0.205630,
                c1label=sun['label'], c2label=mercury['label'],incl=86.62,
                per0=(29.124,'deg'))

# Technically speaking, we don't need light curve information. However we'll
# use it anyway, because then we can readily call the ``compute`` function,
# which will take care of calling ``set_time`` and the radiation budget 
# functionality in the right way (since these need to be computed to get the
# light curve anyway). We could do it manually, but this is easier.

lcdep1 = phoebe.PS('lcdep', atm='kurucz', ld_coeffs='kurucz', ld_func='claret', ref='apparent')
lcdep2 = phoebe.PS('lcdep', atm='blackbody', ld_coeffs=[0.7], ld_func='uniform', ref='apparent', alb=0.932)
obs = phoebe.LCDataSet(time=np.array([orbit['period']*0.25]), columns=['time'], ref=lcdep1['ref'])

# Choose a moderate mesh density for the Sun (not so important), but a finer
# grid for Mercury to resolve the latitudes better.

mesh1 = phoebe.PS('mesh:marching', delta=0.1)
mesh2 = phoebe.PS('mesh:marching', delta=0.03, maxpoints=40000)

# Put the system at about 1 AU:

globals = phoebe.PS('position', distance=(1,'au')) 

# Body setup
# ----------

sun = phoebe.BinaryStar(sun, mesh=mesh1, orbit=orbit, pbdep=[lcdep1])
mercury = phoebe.BodyBag(phoebe.Star(mercury, mesh=mesh2, pbdep=[lcdep2]),
                         orbit=orbit)

system = phoebe.BodyBag([sun, mercury], obs=[obs], globals=globals)


# Computation of observables
# --------------------------

system.compute(heating=True, refl=True, refl_num=1, eclipse_alg='graham')


# Analysis of results
# -------------------

# Let's make a temperature map, with the extremes being the temperature of the
# planet in absence of radiation, and the temperature of maximum heating.

out = system[1].plot2D(select='teff',vmin=100, vmax=700)
cbar = plt.colorbar(out[-1])
cbar.set_label('Effective temperature [K]')
plt.title('Heating of a Mercury-like planet')
plt.savefig('mercury_heating')
plt.show()

"""

.. image:: images_tut/mercury_heating.png
   :scale: 75 %
   :align: center

"""