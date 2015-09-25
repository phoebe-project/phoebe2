"""
The Reflection Effect in PHOEBE
=================================

Last updated: ***time***

In this tutorial, we'll walk through how PHOEBE handles the reflection effect in close binaries:
"""

import phoebe
import matplotlib.colors as colors
import matplotlib.cm as cm

eb = phoebe.Bundle()


norm = colors.Normalize(vmin=0, vmax=10)
cmap = cm.gnuplot 

m = cm.ScalarMappable(norm=norm, cmap=cmap)

"""
Generally speaking, the critera for the reflection effect to occur is that the radii of the irradiating star is at least 15-20% of the
distance between the two objects. 

Although it's called the reflection effect, what really occurs is that the irraddiated body is heated by the flux of the irradiating 
star, which is a bolomoetric process, and then the radiation is reprocessed and re-emitted by the irradiated body, which is a wavelength
dependent process.

That's why it's important to make the distinction between parameters at the component and lcdep levels in PHOEBE. 

Parameters at the component level deal with bolometric processes (i.e. heating) while parameters at the lcdep level are 
passband dependent.

For simplicities sake in this example, however, we'll set all values of those parameters which exist at both levels to the same value. 
"""

eb['value@teff@component@primary'] = 25000
eb['value@teff@component@secondary'] = 5000


eb['value@sma@orbit'] = 1.5, 'Rsol'
eb['value@period@orbit'] = 2.3,'d'
eb['value@incl@orbit'] = 85.

eb.set_value_all('ld_func','linear') 
eb.set_value_all('atm','blackbody') 

"""
The three parameters that control the Reflection Effect in PHOEBE are alb, redist, and redisth. 

To see how the parameters influence the reflection effect, let's start by creating a dataset for us to calculate a synthetic lightcurve.
"""

import numpy as np
import matplotlib.pyplot as plt
eb.lc_fromarrays(time=np.linspace(0,10,500))

eb.run_compute('detailed')

eb.plot_syn('lc01')
plt.show()

"""
Now let's adjust the albedo, or amount of light reflected, and see how it influences the reflection effect.

Keep in mind, that this is done with NO redistribution of heat, so it is all local. 
"""

albedos = np.arange(0,1.1,0.1)

plt.figure(100)
for ii,alb in enumerate(albedos):
    eb.set_value_all('alb',alb)
    eb.run_compute('detailed')
    eb.plot_syn('lc01',color=m.to_rgba(ii),label='%.1f'%alb)

plt.legend(loc='lower right')
plt.savefig('reflection_var_alb.png')
plt.show()

"""

+-----------------------------------------------+
| Image                                         |
+-----------------------------------------------+
| .. image:: images_tut/reflection_var_alb.png  |
|    :width: 233px                              |
+-----------------------------------------------+

"""



"""
For this next part, we'll vary the parameter redist, which has the constraint of: heating = 1-alb ; redist = heating / redist.

If the albedo is set to 1, then all of the light is reflected, and none of it goes into heating, so we're going to set alb = 0.5 for 
this part. 
"""

eb.set_value_all('alb',0.5)

redists = np.arange(0,1.1,0.1)

plt.figure(200)
for ii,redist in enumerate(redists):
    eb['value@redist@component@secondary'] = redist
    eb.run_compute('detailed')
    eb.plot_syn('lc01',color=m.to_rgba(ii),label='%.1f'%redist)
plt.legend(loc='lower right')
plt.savefig('reflection_var_redist.png')
plt.show()

"""

+--------------------------------------------------+
| Image                                            |
+--------------------------------------------------+
| .. image:: images_tut/reflection_var_redist.png  |
|    :width: 233px                                 |
+--------------------------------------------------+

"""


"""
Finally, we're going to allow the parameter redisth to vary. This is used in reflection computations such that the fraction 
redist*redisth is used to mimick horizontal winds aiding in the redistribution of heat.
"""

eb.set_value_all('alb',0.5)
eb['value@redist@component@secondary'] = 0.8

redisths = np.arange(0,1.1,0.1)

plt.figure(300)
for ii,redisth in enumerate(redisths):
    eb['value@redisth@component@secondary'] = redisth
    eb.run_compute('detailed')
    eb.plot_syn('lc01',color=m.to_rgba(ii),label='%.1f'%redisth)

plt.legend(loc='lower right')
plt.savefig('reflection_var_redisth.png')
plt.show()

"""
+---------------------------------------------------+
| Image                                             |
+---------------------------------------------------+
| .. image:: images_tut/reflection_var_redisth.png  |
|    :width: 233px                                  |
+---------------------------------------------------+

As we can see, redisth has little impact on the general shape of the lightcurve. 

Keep in mind that we have only edited the redist and redisth parameters for the secondary body, as there was only one reflection used
in computations. However, you can add as many reflections as you want by editing the refl_num parameter at each compute_label.
"""
