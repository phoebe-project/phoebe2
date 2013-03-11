"""
KOI-126 as a hierarchical triple system
=======================================

Last updated: ***time***

KOI-125 is a triple system. In the Science paper of Carter et al. 2011, it 
was modelled as a dynamical system. We construct a hierarchal model of this
system using the ephemeris and parameters given in the paper.


Initialisation
--------------

"""
# First, import the necessary modules and set up a basic logger
import phoebe
from phoebe.backend import observatory
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.get_basic_logger()

# Parameter preparation
# ---------------------

# Creating the system is easy in this case, since it is part of the library.
system = phoebe.create.KOI126()

# Computation of observables
# --------------------------

# We compute the transit light curve only in the specific windows of the
# transits. These transit timings are reasnably close to the ones presented in
# Carter et al. 2011 (Science), but not exactly since this model is hierarchical
# and not dynamical.
times = [2455102.815, 2455136.716, 2455170.465, 2455204.267, 2455238.207+0.25,
         2455271.751+0.5, 2455305.713+0.5, 2455339.496+0.5,  2455259.000+13.25, 2455326.506+13.5]
window = 1.

# For each of these windows, compute the light curve, get the light curve
# and make a nice plot

for i,time in enumerate(times):
    # Clear results and define the window. We set the mesh of the A star pretty
    # coarse, so that the calculations go reasonably quick.
    system.clear_synthetic()
    system[0].params['mesh']['delta']=0.1
    system.reset()
    times_ = np.linspace(time-window/2.,time+window/2.,250)
    # Compute the light curve and retrieve the results
    phoebe.observe(system,times_,lc=True,eclipse_alg='auto')
    times,signal = system.get_lc(ref='light curve')
    # Make an image at the center time, make a high quality image of the final
    # case
    system.reset()
    if i==(len(times)-1):
        system[0].params['mesh']['delta']=0.05
    system.set_time(time)
    observatory.choose_eclipse_algorithm(system)
    #-- if necessary, subdivide and redetect eclipses/horizon
    for k in range(3):
        #for isystem in the_system:
        system.subdivide(threshold=0,algorithm='edge')
        observatory.choose_eclipse_algorithm(system)
    
    phoebe.image(system,select='teff',cmap='blackbody',ref='light curve')
    plt.twinx(plt.gca())
    plt.twiny(plt.gca())
    plt.plot(times,signal,'o-',color='r',lw=2)
    plt.plot([time],np.interp(np.array([time]),times,signal),'rs',markersize=10)
    plt.xlim(times[0],times[-1])
    plt.ylim(signal.min()-0.01*signal.min(),signal.max()+0.01*signal.max())
    plt.savefig('KOI126_lc_{:03d}.png'.format(i))
    
    phoebe.image(system,select='teff',cmap='eye',ref='light curve')
    plt.twinx(plt.gca())
    plt.twiny(plt.gca())
    plt.plot(times,signal,'o-',color='r',lw=2)
    plt.plot([time],np.interp(np.array([time]),times,signal),'rs',markersize=10)
    plt.xlim(times[0],times[-1])
    plt.ylim(signal.min()-0.01*signal.min(),signal.max()+0.01*signal.max())
    plt.savefig('KOI126_im_{:03d}.png'.format(i))
    plt.close()

"""
+------------------------------+------------------------------+------------------------------+
| First transit (BB temps)     | First transit (projected)    | Second transit               |
+------------------------------+------------------------------+------------------------------+
| .. image:: KOI126_lc_000.png | .. image:: KOI126_im_000.png | .. image:: KOI126_lc_001.png |
|    :scale: 95 %              |    :scale: 95 %              |    :scale: 95 %              |
|    :width: 266px             |    :width: 266px             |    :width: 266 px            |
|    :height: 266px            |    :height: 266px            |    :height: 266 px           |
+------------------------------+------------------------------+------------------------------+
| Third transit                | Fourth transit               | Fifth transit                |
+------------------------------+------------------------------+------------------------------+
| .. image:: KOI126_lc_002.png | .. image:: KOI126_lc_003.png | .. image:: KOI126_lc_004.png |
|    :scale: 95 %              |    :scale: 95 %              |    :scale: 95 %              |
|    :width: 266px             |    :width: 266px             |    :width: 266 px            |
|    :height: 266px            |    :height: 266px            |    :height: 266 px           |
+------------------------------+------------------------------+------------------------------+
| Sixth transit                | Seventh transit              | Eight transit                |
+------------------------------+------------------------------+------------------------------+
| .. image:: KOI126_lc_005.png | .. image:: KOI126_lc_006.png | .. image:: KOI126_lc_007.png |
|    :scale: 95 %              |    :scale: 95 %              |    :scale: 95 %              |
|    :width: 266px             |    :width: 266px             |    :width: 266 px            |
|    :height: 266px            |    :height: 266px            |    :height: 266 px           |
+------------------------------+------------------------------+------------------------------+
| Nineth transit               |  Tenth transit               |                              |
+------------------------------+------------------------------+------------------------------+
| .. image:: KOI126_lc_008.png | .. image:: KOI126_lc_009.png |                              |
|    :scale: 95 %              |    :scale: 95 %              |                              |
|    :width: 266px             |    :width: 266px             |                              |
|    :height: 266px            |    :height: 266px            |                              |
+------------------------------+------------------------------+------------------------------+

"""
