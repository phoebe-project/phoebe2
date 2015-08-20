"""
Building my first binary from scratch with PHOEBE 2.0-alpha
===========================================================

Last updated: ***time***

This tutorial will cover the basic steps of creating the model of a binary star using the PHOEBE 2.0-alpha frontend. All frontend functionality is documented in the frontend Application Programming Interface (API) and you can refer to it for further details on any of the used commands in this tutorial. If you haven't done so already, you may want to read the :doc:`First steps with PHOEBE 2.0-alpha <tutorial.first_steps>` tutorial first -- it covers the details that we will assume you are familiar with.

Ok, enough warnings and disclaimers, let us build our first binary from scratch! As before, we will start by creating a bundle. So fire up python, import phoebe and initialize a new bundle:
"""

import phoebe
eb = phoebe.Bundle()

"""
This will create a new binary with default parameters. Take some time to get familiar with the structure of the newly created system:
"""

print eb.summary()

"""
You can get further information on any part of the structure by accessing it explicitly, e.g.:
"""

print eb['position']
print eb['primary']

"""
or, to print all information on the system:
"""

print eb.tree()

"""
You can change the values of the parameters using a dictionary-style or the method-style approach:
"""

eb['teff@primary'] = 5500.
eb.set_value('period', 10)

"""
For the pedantic among you, the units can be provided explicitly, in the following manner:
"""

eb['teff@primary'] = (5500., 'K')
eb.set_value('period', 10, 'd')

"""
To take a look at a few other things you can change issue:
"""
print 
print 
print 
print eb['primary']
print 
print 
print 


"""
where all recognized units and their string representations are listed in the `units package documentation <phoebe.units>`_. If 'teff@primary' confuses you, this is called a twig. Twigs are used to refer to all parameters in PHOEBE. They are explained in detail in the :doc:`First steps with PHOEBE 2.0-alpha <tutorial.first_steps>` tutorial, and a cool feature about them is that they are tab-completable. To save ourselves from excessive typing, we will be using the dictionary-style approach here, but you are of course encouraged to use your preferred way.

The default system has no data attached. You may choose to attach observations, or synthesize a theoretical curve. Under the hood PHOEBE makes little distinction, and the principle is always the same. In the following we will focus on adding a light curve, but the same applies to other types of curves, such as radial velocities (RVs) or spectra.

Observations (if available) are contained in a structure called :ref:`lcdep <parlabel-phoebe-lcdep>`. This structure holds the filename, all parsed columns with data (times, fluxes, sigmas, ...), etc. The data will be typically attached to the system (so that they correspond to the binary system), but in some circumstances they may also be attached to individual bodies, i.e. when photometry of individual components is possible, such as in the resolved binary system. Don't worry, we'll take you through all the steps on how to do that -- right now we are only covering the basics.

In order to compute a theoretical light curve, we need to provide PHOEBE with passband-dependent parameters. These are contained in a structure called :ref:`lcdep <parlabel-phoebe-lcdep>`. For example, you will find here the passband used, passband luminosity, limb darkening coefficients, third light, etc. PHOEBE needs these parameters to compute a light curve. All bodies in a system have their own lcdep structure, since each star has its own passband-dependent properties.

Once a theoretical light curve is computed, it is contained in a structure called lcsyn. This structure is always paired up with the lcobs structure so that PHOEBE can cross-reference computations with observations. From then on, we can plot the residuals and work our magic with the data.

So let us start by generating our first synthetic light curve. To do that, we will use the bundle method :py:func:`lc_fromarrays() <phoebe.frontend.bundle.Bundle.lc_fromarrays>`. Let us first run it and then explain what it did.
"""

import numpy as np
eb.lc_fromarrays(phase=np.linspace(-0.6, 0.6, 201))

"""
If you are familiar with Python, you probably know what is happening here. We imported `numpy <http://www.numpy.org>`_, a python numerical library (that also happens to be a requirement for PHOEBE, so we know you already have it) and used a method defined therein, linspace(), to generate an array between -0.6 and 0.6 of length 201. We passed that array as a phase array, and PHOEBE automatically attached all the required structures for the computation of the light curve. Let us inspect them:
"""

print eb.summary()

"""
You can see that, next to the structures of the original system, now we have system-level entries :ref:`lcobs <parlabel-phoebe-lcobs>` and :ref:`lcsyn <parlabel-phoebe-lcsyn>`, and component-level entries :ref:`lcdep <parlabel-phoebe-lcdep>`. They are all connected with the data reference ``lc01``: that is how PHOEBE knows what parameters go with what light curve. Take a minute to explore them:
"""

print eb['lc01@lcobs']
print eb['lc01@lcsyn']
print eb['lc01@primary']
print eb['lc01@secondary']

"""
There are several details worth pointing out.

* In the 'lc01@lcobs' structure you can see that most of the arrays are empty (such as time, flux, flag, weight, etc), and that the phase array is initialized to the array passed as the argument to :py:func:`lc_fromarrays() <phoebe.frontend.bundle.Bundle.lc_fromarrays>`. It is probably obvious now how you would pass any additional data to :py:func:`lc_fromarrays() <phoebe.frontend.bundle.Bundle.lc_fromarrays>` in order to have them appear in the :ref:`lcobs <parlabel-phoebe-lcobs>` structure;
* the 'sigma@lc01@lcobs' array is initialized, and each element is -1. This implies that standard deviations on data points are unknown and that they will not be plotted. However, if you ever wanted to run heuristic samplers such as MCMC to fit the data, that will not do, because the results will crucially depend on per-point standard deviations -- that is why they default to an unphysical value, to alert you should you neglect them. But let's not get ahead of ourselves -- fitting is a subject of another tutorial;
* notice that all arrays in 'lc01@lcsyn' are empty; the :py:func:`lc_fromarrays() <phoebe.frontend.bundle.Bundle.lc_fromarrays>` method only initialized this structure, but another method, :py:func:`run_compute() <phoebe.frontend.bundle.Bundle.run_compute>`, will actually fill them in;
* both 'lc01@primary' and 'lc01@secondary' are by default identical, referring to 'lc01' as a cross-reference to :ref:`lcsyn <parlabel-phoebe-lcsyn>` and :ref:`lcobs <parlabel-phoebe-lcobs>`. Of notable interest is the passband, which defaults to Johnson V, and passband luminosity ``pblum`` that is set to -1, meaning that it will be computed automatically during :py:func:`run_compute() <phoebe.frontend.bundle.Bundle.run_compute>`;
* you might be surprised to find the passband in lcdep instead of lcobs; this is because both lcobs and lcsyn depend on them, so by definition that makes passbands passband-dependent parameters;
* to change any of the parameters in :ref:`lcobs <parlabel-phoebe-lcobs>`, :ref:`lcsyn <parlabel-phoebe-lcsyn>` or :ref:`lcdep <parlabel-phoebe-lcdep>`, use the already mentioned approach, e.g.:
"""


eb['flux@lc01@lcobs'] = np.random.normal(1.0, 0.1, 201)

"""
However, if you do assign arrays this way, it is your responsibility to ensure that the lengths of all arrays are the same.

So let us compute our light curve. There are several compute options that are stored in the system-level 'compute' container. We can examine them through the bundle:
"""

print eb['compute']
print eb['detailed@compute']

"""
The 'detailed' ParameterSet turns on all relevant physics and will take the longest to compute; the 'legacy' ParameterSet aims to stay as close as possible to the physical considerations in the old PHOEBE version and thus attain maximum compatibility; 'default' takes default values from the PHOEBE 2.0 backend, and 'preview' turns off all time-consuming features and provides a cursory look at the model. Since by default we built a simple binary on a circular orbit, execution time cost is not an issue and we can compute a detailed model:
"""

eb.run_compute('detailed')

"""
This will populate the :ref:`lcsyn <parlabel-phoebe-lcsyn>` structure and we can plot our first light curve!
"""

print eb['lc01@lcsyn']
import matplotlib.pyplot as plt
eb.plot_syn('lc01', fmt='r-')
plt.show()

"""
PHOEBE interfaces `matplotlib <http://www.matplotlib.org>`_ for plotting, another requirement for installing PHOEBE so you already have it and don't have to worry about it. The import statement loads up matplotlib and the bundle method :py:func:`plot_syn() <phoebe.frontend.bundle.Bundle.plot_syn >` plots the theoretical light curve. The first argument is a twig of the synthetic curve you want plotted, and the second argument, 'r-', is the plot specifier that is passed to matplotlib. It tells the plotting method to use the red color (r) and to connect the points with a solid line (-). If, say, you wanted blue circles or green points instead, you would pass 'bo' or 'g.'. Matplotlib's documentation can teach you more -- much more, soooo much more -- pretty much everything you ever wanted to know about plotting, and then some.

The plotted quantities match those that went into creating the :ref:`lcobs <parlabel-phoebe-lcobs>`, in this case phases. You can still access times in the :ref:`lcsyn <parlabel-phoebe-lcsyn>` structure directly:
"""

plt.plot(eb['value@time@new_system@lcsyn@lc01'], eb['value@flux@new_system@lcsyn@lc01'], 'r-')
plt.show()

"""
Whenever you issue :py:func:`run_compute() <phoebe.frontend.bundle.Bundle.run_compute>`, the contents of :ref:`lcsyn <parlabel-phoebe-lcsyn>` are rewritten. However, if you want to explicitly clear the contents, you can do so by running :py:func:`clear_syn() <phoebe.frontend.bundle.Bundle.clear_syn>`:
"""

print eb['lc01@lcsyn']
eb.clear_syn()
print eb['lc01@lcsyn']

"""
This will remove all theoretical data from all bodies in the system and prepare them for a fresh computation. To remove the dataset altogether, you would do:
"""

eb.remove_data('lc01')
print eb.summary()

"""
So what other types of data can we compute? PHOEBE veterans among you are probably already thinking of radial velocity curves, and just you wait for the beta release, for inferferometry, spectra and polarimetry are all coming your way! Each category of data has its own wrapper function, each of which behaves essentially the same way, but as of this alpha release the following two are supported:

eb.lc_fromarrays(), eb.rv_fromarrays()

The inline help on all these wrappers contains all the arrays you can set, so we urge you to spend a couple of minutes browsing through it:
"""

print help(eb.rv_fromarrays)

"""
Yes, the calling sequence might look a little daunting but there's really not much to it -- all arrays default to None and you need to pass only those of interest. There is only one required parameter and that is the object reference, ``objref``. It is how PHOEBE identifies the object to which the data need to be attached. Revisiting:
"""

print eb.summary()

"""
you can see three objects: ``new_system``, ``primary`` and ``secondary``. They are easily recognized by the bold underline font, and by the provided type in parentheses (for those poor souls that don't have ANSI-capable terminals). Attaching a radial velocity dataset to all these objects makes sense, but the result will be different. If you attach a dataset to the primary or the secondary, these will be their respective radial velocities. If, on the other hand, you attach it to the system, then that will be a systemic radial velocity. So let us play with this for a bit, starting from individual components' radial velocity curves.
"""

eb.rv_fromarrays('primary', phase=np.linspace(-0.6, 0.6, 201))
eb.rv_fromarrays('secondary', phase=np.linspace(-0.6, 0.6, 201))
print eb.summary()

"""
We now have the :ref:`rvobs <parlabel-phoebe-rvobs>`, :ref:`rvsyn <parlabel-phoebe-rvsyn>` and :ref:`rvdep <parlabel-phoebe-rvdep>` structures, all of them serving the exact same role as their lc counterparts. You can compute the theoretical radial velocity curves and plot them using the same sequence as before:
"""

eb.run_compute('detailed')
eb.plot_syn('rv01', 'r-')
eb.plot_syn('rv02', 'b-')
plt.show()

"""
Note that :py:func:`run_compute() <phoebe.frontend.bundle.Bundle.run_compute>` computes all available data structures unless they are disabled. How do you disable datasets, you wonder? Easily:
"""

eb.disable_data('rv02')

"""
In consequence, you will see them crossed out in the eb.summary() output (again, if you are a lucky owner of the ANSI-capable terminal). To enable them back, issue:
"""

eb.enable_data('rv02')

"""
The same holds true, of course, for light curves as well.

One final task lies before us: we need to save the parameter file. The current format of choice for PHOEBE is JSON. It stands for JavaScript Object Notation, but please ignore that -- nothing about PHOEBE is JavaScript, and the reason why we use JSON out of the box is because of its very legible ASCII representation of file contents. To save the file, issue:
"""

eb.save('my_1st_binary.phoebe')

"""
We encourage you to take a peak into that file and see how it is constructed and what it contains. To open such a file, you would simply load it into the bundle, similar to what we showed you in Tutorial 1, :doc:`First steps with PHOEBE 2.0-alpha <tutorial.first_steps>`:
"""

eb = phoebe.Bundle('my_1st_binary.phoebe')

"""
Congratulations, you survived your second PHOEBE 2.0-alpha tutorial! Poke around, make things, break things, and please let us know how you fare! More tutorials are coming your way, make sure to check back often to the Documentation page!
"""
