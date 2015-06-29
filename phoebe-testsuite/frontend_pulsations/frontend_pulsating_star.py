"""
Pulsating Star
========================

Last updated: ***time***

This example shows how to attach pulsations to a single star

inspired by Aerts 1993

Initialisation
--------------

"""
# First we'll import phoebe and create a logger



import phoebe
import numpy as np
import matplotlib.pyplot as plt
logger = phoebe.get_basic_logger()


# Bundle preparation
# ------------------

# we'll just use the default Sun system and then change the parameters

b = phoebe.Bundle('Sun')

b['teff'] = 25600.
b['incl'] = 80.
b['radius'] = 8.8
b['mass'] = 14.4
b['shape'] = 'sphere'
b['atm'] = 'kurucz'
b['ld_coeffs'] = 'kurucz'
b['ld_func'] = 'claret'
b['irradiator'] = True
b['delta@mesh'] = 0.08

# we want to give veq instead of rotperiod - so we have to create and
# attach a new parameter and then setup a constraint so phoebe knows
# how to convert to rotperiod

b['star'].add(phoebe.Parameter(qualifier='veq', unit='km/s', value=200))
b['star'].add_constraint('{rotperiod} = 2*np.pi*{radius}/{veq}')

# Add pulsations
# -----------------

# We need to attach a ParameterSet with context='puls' to the star

b.attach_ps(phoebe.PS(context='puls', label='modepulsations'), 'object@Sun')

# and set the Parameters

b['freq@puls'] = 3.97
b['ampl@puls'] = 0.01
b['k@puls'] = 0.0
b['l@puls'] = 4
b['m@puls'] = 4
b['ledoux_coeff@puls'] = 0.1585
b['scheme@puls'] = 'nonrotating'

# Compute model
# ---------------

times = np.linspace(0,1./(b['value@freq@puls']-4./b['value@rotperiod']),50)
b.lc_fromarrays(time=times, ld_func='claret', ld_coeffs='kurucz', atm='kurucz')

# since we didn't give dataref, it will automatically be assigned as lc01

b.sp_fromarrays(time=times, wavelength=np.linspace(4544, 4556, 200))

# since we didn't give dataref, it will automatically be assigned as sp01


b.run_compute()

# Plotting
# ---------

b.attach_plot_syn('lc01', axesloc=(2,2,1))

# since we didn't give figref, it will automatically be assigned as fig01

b.attach_plot_syn('sp01', ylim=(0.970,1.005), axesloc=(2,2,2))
b.attach_plot_mesh(dataref='__bol', objref='Sun', select='proj', axesloc=(2,2,3))
b.attach_plot_mesh(dataref='__bol', objref='Sun', select='rv', axesloc=(2,2,4))

b.draw('fig01', time=times, fps=5, fname='frontend_puls_mode.gif')

"""
.. image:: images_tut/frontend_puls_mode.gif
   :width: 750px    
"""
