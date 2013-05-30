"""
Contact binary EE Ceti
======================

Compute a movie of a binary system, overplotted with the light curve. This
script should work for every binary recognised by the ``parameters.create``
library. Just change the ``system_name`` variable below.
"""
import matplotlib.pyplot as plt
import numpy as np
import phoebe
from phoebe.backend import observatory
from PIL import Image
import glob
import subprocess
import os

logger = phoebe.get_basic_logger()

# Create the system: we'll use black bodies to compute the light curves just
# in case the Kurucz grid does not cover a wide enough parameter space.
system_name = 'EE Cet'

cmap = 'blackbody_proj'
heating = False
system = phoebe.create.binary_from_deb2011(system_name,create_body=True,pbdep=['lcdep'])

# Generate a time series to simulate the light curve and images on, and compute
# them
period = system.params['orbit']['period']
t0 = system.params['orbit']['t0']
system.params['orbit']['incl'] = 63.50,'deg'
times = np.linspace(t0,t0+period,100)

# Compute the whole time series and get the computed light curve afterwards
phoebe.observe(system,times,lc=True,heating=heating,
               extra_func=[observatory.ef_binary_image],
               extra_func_kwargs=[dict(select='teff',cmap=cmap,ref=0)])
results = system.get_synthetic(category='lc',ref=0,cumulative=True)

times = np.array(results['time'])
signal = np.array(results['flux'])

# Collect all the image files, show the image and overplot the light curve
# show a red dot at the local time
files = sorted(glob.glob('*.png'))
for i,ff in enumerate(files):
    
    data = Image.open(ff)
    shape = data.size
    
    plt.figure(figsize=(8,8*shape[1]/float(shape[0])))
    ax = plt.axes([0,0,1,1],axisbg='k',aspect='equal')
    plt.imshow(data,origin='image')
    plt.twinx(plt.gca())
    plt.twiny(plt.gca())
    plt.plot(times,signal,'o-',color='0.9')
    plt.plot(times[i:i+1],signal[i:i+1],'rs',ms=10)
    ylims = signal.min(),signal.max()
    ylims = ylims[0]-(ylims[1]-ylims[0])*0.1,ylims[1]+(ylims[1]-ylims[0])*0.1
    plt.ylim(ylims)
    plt.xlim(times[0],times[-1])
    
    plt.savefig(ff)
    plt.close()

"""

+---------------------------------------+
| EE Ceti                               |
+---------------------------------------+
| .. image:: images_tut/es_EE_Cet.gif   |
|    :scale: 75 %                       |
+---------------------------------------+

"""

# Make a movie and forget about the figures created
subprocess.call('convert -loop 0 *.png es_{}.gif'.format(system_name.replace(' ','_')),shell=True)
for ff in files: os.unlink(ff)