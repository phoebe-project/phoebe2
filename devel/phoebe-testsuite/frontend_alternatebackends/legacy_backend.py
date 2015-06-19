"""
Using alternate computation backends: PHOEBE 1.0 Legacy [EXPERIMENTAL]
========================

Last updated: ***time***

This example shows PHOEBE's interface to using other supported alternative backends - in this case the original PHOEBE 1.0/0.3/Legacy.

In order for this script to work, you must have the latest version PHOEBE 1.0 installed (http://phoebe-project.org/1.0) and run
python setup.py build; sudo python setup.py install in the phoebe-py directory inside the PHOEBE 1.0 stable SVN.

Please note: support for alternative backends is currently experimental and not all features/parameters have direct translations available.

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

# Now let's create a default binary and attach a synthetic light curve and radial velocities.
# Nothing here is different than usual
b = phoebe.Bundle()
b.lc_fromarrays(time=np.linspace(0,1,100), dataref='lc01')
b.rv_fromarrays(time=np.linspace(0,1,40), objref=['primary', 'secondary'], dataref='rv01')

# Running computations
# ----------------------

# In order to use PHOEBE legacy as an alternative backend, we must add
# its own compute options ParameterSet to the bundle.
b.add_compute(context='compute:legacy', label='phoebe1')

# Printing this ParameterSet shows the options that can be adjusted - these
# can be accessed and modified as with anything else using twig/dictionary access in the bundle.
print(b['phoebe1'])

# Now just pass the label for these compute options to run_compute, and PHOEBE
# will pass all computation to PHOEBE Legacy and retrieve the output
b.run_compute('phoebe1')

# Plotting
# ----------------

# Now that computations are complete, we can use any of the PHOEBE 2 convenient
# plotting functions.
b.plot_syn('lc01')
plt.savefig('legacy_lc.png')
plt.cla()
b.plot_syn('rv01@primary', fmt='b-')
b.plot_syn('rv01@secondary', fmt='r-')
plt.savefig('legacy_rv.png')


"""

+--------------------------------------+----------------------------------------------+
| Synthetic Light Curve                | Synthetic Radial Velocities                  |
+--------------------------------------+----------------------------------------------+
| .. image:: images_tut/legacy_lc.png  | .. image:: images_tut/legacy_rv.png          |
|    :width: 233px                     |    :width: 233px                             |
+--------------------------------------+----------------------------------------------+

"""
