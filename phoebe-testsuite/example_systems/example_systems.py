"""
Various example binary systems
==============================

Compute a movie of a binary system, overplotted with the light curve. This
script should work for every binary recognised by the ``parameters.create``
library. Just change the ``system_name`` variable below.
"""
import matplotlib.pyplot as plt
import numpy as np
import phoebe
from phoebe.utils import plotlib
from phoebe.backend import observatory
from PIL import Image
import glob
import sys

logger = phoebe.get_basic_logger()

# Create the system: we'll use black bodies to compute the light curves just
# in case the Kurucz grid does not cover a wide enough parameter space.
system_name = 'V380_Cyg'
system_name = 'HV2241'
system_name = 'T_CrB'
system_name = 'KOI126'
if sys.argv[1:]:
    system_name = sys.argv[1]

cmap = 'eye' if (system_name in ['V380_Cyg','HV2241'])  else 'blackbody'
heating = system_name in ['V380_Cyg','HV2241','T_CrB']
atm = 'blackbody'
ld_func = 'linear'
ld_coeffs = [0.5]
system = phoebe.create.from_library(system_name,create_body=True,obs=['lcdep'],\
                              atm=atm,ld_func=ld_func,ld_coeffs=ld_coeffs)


# In this example, we're only interested in Binaries. If there are more complicated
# systems, such as TCrB, we select the stars and neglect other Bodies such as
# disks.
system = system[:2]

# Generate a time series to simulate the light curve and images on, and compute
# them
period = system[0].params['orbit']['period']
t0 = system[0].params['orbit']['t0']
tN = t0+period if not system_name=='KOI126' else t0+period*0.1
times = np.linspace(t0,tN,100)

# Compute the whole time series and get the computed light curve afterwards
phoebe.observe(system,times,lc=True,heating=heating,
      extra_func=[observatory.ef_binary_image],
      extra_func_kwargs=[dict(select='teff',cmap=cmap)])

results = system.get_synthetic('lc',ref=0,cumulative=True)
label = results['ref']
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

# Make a movie and forget about the figures created
    
"""    

+---------------------------------------+---------------------------------------+
| V380 Cygni                            | HV2241                                |
+---------------------------------------+---------------------------------------+
| .. image:: images_tut/es_V380_Cyg.gif | .. image:: images_tut/es_HV2241.gif   |
|    :scale: 100 %                      |    :scale: 100 %                      |
|    :height: 233px                     |    :height: 233px                     |
|    :width: 500px                      |    :width: 500px                      |
+---------------------------------------+---------------------------------------+

+---------------------------------------+
| T CrB                                 |
+---------------------------------------+
| .. image:: images_tut/es_T_CrB.gif    |
|    :scale: 75 %                       |
+---------------------------------------+

+---------------------------------------+
| KOI 126                               |
+---------------------------------------+
| .. image:: images_tut/es_KOI126.gif   |
|    :scale: 100 %                      |
+---------------------------------------+

"""

for ext in ['.gif','.avi']:
    plotlib.make_movie('ef_binary_image*.png',output='es_{}{}'.format(system_name,ext),cleanup=ext=='.avi')