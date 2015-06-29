"""
Advanced Plotting in the Frontend
========================

Last updated: ***time***

This example shows many advanced options for building complex figures and animations in the frontend.

Its probably not advised to use all of these things at once, but hopefully this example shows a wide 
range of options and possibilities.

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

b = phoebe.Bundle()
b['atm@primary'] = 'blackbody'
b['atm@secondary'] = 'blackbody'
b['incl'] = 88.
b['syncpar@primary'] = 1.2
b['q'] = 0.8

# Create synthetic model
# ---------------------

b.lc_fromarrays(time=np.arange(0.5,2.5,0.01), dataref='lc01')
b.rv_fromarrays(time=np.arange(1,2,0.01), objref=['primary','secondary'], dataref='rv01')
b.sp_fromarrays(time=np.arange(1,2,0.01), wavelength=np.linspace(399,401,500), dataref='sp01')

b.run_compute('preview')


# Plotting
# ---------

# As in the <testsuite.frontent_binary_animation> example, we'll use the attach_plot commands to build a figure, and later
# draw the figure at a list of times to build an animation.

# The first panel will be the mesh, with surface elements colored by their effective temperatures.

b.attach_plot_mesh(dataref='lc01', select='teff', axesloc=(2,2,1), axesref='meshax', figref='fig')

# To avoid the subplot rescaling as the time changes, we'll set the limits
b['xlim@meshax'] = (-7,7)
b['ylim@meshax'] = (-7,7)


# The second panel will be a top-down view of the orbit

b.attach_plot_orbit(objref='primary', dataref='sp01', times=b['time@lc01@lcsyn'], xselect='x', yselect='z', fmt='b-', highlight=True, highlight_fmt='bo', axesloc=(2,2,3), axesref='orbax')   
b.attach_plot_orbit(objref='secondary', dataref='sp01', times=b['time@lc01@lcsyn'], xselect='x', yselect='z', fmt='r-', highlight=True, highlight_fmt='ro')

# In the third panel we'll show three subplots - for the light curve, radial velocities, and spectra.
# We'll also throw in some complex options for noting the time of the current frame and time-spans of various datasets.

# For the light curve plot, we'll enable highlighting, uncovering, and scrolling so that the current time is always on the center with 1 day to the left and right.
b.attach_plot_syn('lc01', fmt='k-', axesloc=(3,2,2), highlight=True, highlight_fmt='ko', uncover=True, scroll=True, scroll_xlim=(-1,1), plotref='lcplot', axesref='lcax')

# Since we've enabled uncover, the ylimits of the subplot will change as the light curve gets uncovered.  To prevent this, we'll set the
# limits manually (comment this out to see the difference)
b['ylim@lcax'] = (0.9*min(b['value@flux@lc01@lcsyn']), 1.1*max(b['value@flux@lc01@lcsyn']))
# Note: Since we're hardcoding these values, the limits of the plot will not automatically adjust if the amplitude of your light curve changes if you change parameters of your model and recompute.

# For the radial velocity plot, we'll enable highlight for the primary and uncover for the secondary
b.attach_plot_syn('rv01@primary', fmt='b-', axesloc=(3,2,4), highlight=True, highlight_fmt='bo', axesref='rvax')
b.attach_plot_syn('rv01@secondary', fmt='r-', highlight=False, uncover=True)

# Spetra plots will automatically select and plot the spectra for the current frame's time
b.attach_plot_syn('sp01', fmt='k-', ylim=(0.955,1.005), axesloc=(3,2,6), axesref='spax')

# Since we don't have spectra for every time in the light curve and since the yscale of the
# spectra may be different at each time point, let's again harcode the axes limits
b['xlim@spax'] = (399, 401)
b['ylim@spax'] = (0.955, 1.005)

# Advanced Plotting
# ------------------

# You can also attach any command sent through plt.  For example, to add a simple static text annotation that will execute plt.text(1.0, 2000, 'my label')
# each time the figure is drawn, you can do the following:
b.attach_plot_mplcmd('text', (399.5, 0.96, 'my text'), axesref='spax')

# Note: you'll notice here that we're not adding this to the latest axes anymore, so we specify the axesref
# which we already defined earlier.  That way this plotting call will be added to the light curve subplot
# which we created much earlier.

# But this isn't entirely flexible - these functions must be static and are unaware of the current state of
# the bundle and the time.  If you need more flexibility, you can define your own plotting processing function.
# These functions must take the bundle itself and the time as the first two arguments and then return
# the matplotlib function name (string), the arguments to pass to that function (tuple), the keyword-arguments
# to pass to that function (dict), and a dictionary of default options which will be saved to the stored
# parameter set (dict).

def current_time_text(b, t, **kwargs):
    # we'll pop x and y since they need to be args instead of kwargs
    x = kwargs.pop('x', 0)
    y = kwargs.pop('y', 0)
    
    # mpl_func_name must be a method of a matplotlib axes
    mpl_func_name = 'text'
    
    # arguments must be provided as a tuple or list (even if a single argument)
    args_for_mpl_func = (x, y, "time={:0.02f}".format(t) if t is not None else "")
    
    # and here we'll just pass all kwargs onto the matplotlib function
    kwargs_for_mpl_func = kwargs
    
    # dictionary_of_defaults can assign parameters at the plot, axes, or figure level
    # For full control, give a dictionary which assigns to 'plot', 'axes', or 'figure' and give description
    dictionary_of_defaults = {}
    dictionary_of_defaults['horizontalalignment'] = {'ps': 'plot', 'value': kwargs.get('horizontalalignment', 'center'), 'description': 'horizontal alignment of text'}
    dictionary_of_defaults['verticalalignment'] = {'ps': 'plot', 'value': kwargs.get('verticalalignment', 'center'), 'description': 'vertical alignment of text'}
    
    # or to allow automatic assignment, simply provide the value
    dictionary_of_defaults['x'] = x
    dictionary_of_defaults['y'] = y
    
    return mpl_func_name, args_for_mpl_func, kwargs_for_mpl_func, dictionary_of_defaults

b.attach_plot(func=current_time_text, x=0, y=0, horizontalalignment='center', plotref='timetxt', axesref='orbax')


# There are also some custom functions pre-defined.  You can attach these by passing the function itself or
# the name of the function (so long as it exists in phoebe.frontend.plotting).

# Here we'll draw a vertical line on the radial velocity subplot at the current time.  We'll also hardcode the
# xlimits to only show the data and not show the line when outside of those limits.

b.attach_plot(func=phoebe.frontend.plotting.time_axvline, linestyle=':', xlim=(1,2), plotref='axvline', axesref='rvax')

# or by the name of the function (so long as it exists in phoebe.frontend.plotting).  In these examples
# we'll draw the timespan of the rv01 dataset on the lc01 subplot.

b.attach_plot(func='ds_text', dataref='primary@rv01', y=32000, plotref='dstext', axesref='lcax')
b.attach_plot(func='ds_axvspan', dataref='primary@rv01', plotref='dsaxvspan', axesref='lcax')


# Creating the animation
# -----------------------

# Now we can draw each frame and compile into an animated gif in a single line of code - without needing 
# to write a custom for loop.

fig = plt.figure(figsize=(10,6))
b.draw('fig', time=b['time@lc01@lcsyn'], fname='binary_advanced.gif')


"""
.. image:: images_tut/binary_advanced.gif
   :width: 750px                 
"""
