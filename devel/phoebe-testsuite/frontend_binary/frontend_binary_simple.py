"""
Simple Binary
========================

Last updated: ***time***

This example shows the very basics of changing parameter values and creating a quick synthetic light curve using the frontend interface

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

# Without any arguments, phoebe.Bundle() creates a default binary

b = phoebe.Bundle()

# We can print information about the binary in a variety of different formats

print b
print b.summary()
print b.list()

# Accessing or modifying the value of a Parameter can be done through dictionary acess

print b['incl']
b['incl'] = 90
print b['incl']

# Parameters that exist in multiple instances in the system require more information.  
# We call these complex dictionary keys 'twigs'.  See the frontend tutorials for a 
# more in-depth explanation

print b['teff@primary']

# Creating a synthetic light curve
# ---------------------------------
b.lc_fromarrays(time=np.linspace(0,3,100), dataref='mylc')
b.run_compute()

# Plotting
# -----------

b.plot_syn('mylc')
plt.show()
