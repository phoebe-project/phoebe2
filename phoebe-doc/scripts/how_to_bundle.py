"""
How to use Bundle to attach datasets and plot options
=====================================================

Step 0: Preparations
--------------------
"""

import phoebe
from phoebe.backend.bundle import Bundle
import matplotlib.pyplot as plt

logger = phoebe.get_basic_logger()

"""

Step 1: Create the System
-------------------------

see 'How to create a custom binary system' for different options 
for setting up a hierarchical structure.  
"""

exdir = '../../phoebe/parameters/examples/V380_Cyg_circular/'
system = phoebe.create.from_library(exdir+'V380_Cyg_circular.par', create_body=True)

"""
Step 2: Create the Bundle and Attach the System
-----------------------------------------------

Once we have our system, we can initiate a bundle.  This bundle
will allow us to also save and keep track of computing options and
feedback, plotting options, and higher-level helper functions.
"""

bundle = Bundle(system)

"""
a hierarchical view of the system can be printed using
get_system_structure
"""

print bundle.get_system_structure()

"""
access to the ParameterSets through their labels requires
get_component, get_orbit, get_mesh, or the more general get_ps
"""

print "period:", bundle.get_orbit('V380Cyg').get_value('period')
print "myprimary: teff", bundle.get_component('myprimary').get_value('teff')

"""
Step 3: Loading datasets 
------------------------

The benefit of using a bundle is the ease of attaching and keeping
track of other items such as datasets.

bundle.load_data takes essentially the same information as datasets.parse_*
except it will then also create the spdeps and attach the datasets to the 
correct items in your system hierarchy.
"""

bundle.load_data('rvobs',exdir+'myrv_comp1.rv',
        passband='JOHNSON.V',
        columns=['time','rv','sigma'],
        components=[None,'myprimary','myprimary'],
        ref='myrv_comp1')

bundle.load_data('rvobs',exdir+'myrv_comp2.rv',
        passband='JOHNSON.V',
        columns=['time','rv','sigma'],
        components=[None,'mysecondary','mysecondary'],
        ref='myrv_comp2')

bundle.load_data('lcobs',exdir+'mylc.lc',
        passband='JOHNSON.V',
        columns=['time','flux','sigma'],
        components=[None,'V380Cyg','V380Cyg'],
        ref='mylc')

"""
We can now enable/disable and change whether pblum or l3
are fit across all instances of the same dataset ref
"""

bundle.adjust_obs('mylc',l3=True,pblum=True)

"""
a list of all attached datasets can be retrieved using get_obs
"""

print len(bundle.get_obs()),"datasets:",bundle.get_obs()

"""
which also can be filtered by objectname or dataset ref
"""

print "search: myprimary:",bundle.get_obs('myprimary')
print "search: myrv_comp2:",bundle.get_obs(ref='myrv_comp2')

"""
Step 4: Creating a Model
------------------------

Another advantage to Bundle is the ability to store (multiple)
compute options in the .phoebe file.
"""

bundle.add_compute(label='Preview')
bundle.get_compute('Preview').set_value('subdiv_num',2)

"""
we then just need to refer to the label of the compute option
we wish to use to compute a new model
"""

bundle.run_compute('Preview')

"""
Step 5: Plotting
----------------

We can also save options for plots.  Although it is often simpler
to manually create simple plots of a few datasets, this framework
allows saving these plots for quick recreation and use in the gui.

We first create an axes instance and then add individual plotting 
calls to the axes
"""

bundle.add_axes(category='lc',title='LC Plot')
bundle.get_axes('LC Plot').add_plot(type='lcobs',
        objref='V380Cyg',dataref='mylc',errorbars=False)
bundle.get_axes('LC Plot').add_plot(type='lcsyn',
        objref='V380Cyg',dataref='mylc',color='r')

bundle.add_axes(category='rv',title='RV Plot')
bundle.get_axes('RV Plot').add_plot(type='rvobs',
        objref='myprimary',dataref='myrv_comp1',
        color='b',errorbars=False)
bundle.get_axes('RV Plot').add_plot(type='rvsyn',
        objref='myprimary',dataref='myrv_comp1',
        color='b')
bundle.get_axes('RV Plot').add_plot(type='rvobs',
        objref='mysecondary',dataref='myrv_comp2',
        color='r',errorbars=False)
bundle.get_axes('RV Plot').add_plot(type='rvsyn',
        objref='mysecondary',dataref='myrv_comp2',
        color='r')

"""
we can now quickly plot or replot these axes using
the settings above with one single call
"""

bundle.plot_axes('LC Plot')
plt.show()
plt.cla()
bundle.plot_axes('RV Plot')
plt.show()

"""
Step 6: Saving the Bundle
-------------------------
"""

bundle.save('./how_to_bundle.phoebe')

"""
later we can restore this bundle and everything attached
to it

"""

bundle = phoebe.load_body('./how_to_bundle.phoebe')



