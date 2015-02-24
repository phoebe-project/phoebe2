import oc_d
import phoebe
import numpy as np
from phoebe.atmospheres.roche import exact_lagrangian_points 
from phoebe.units.conversions import convert
import matplotlib.pyplot as plt

logger = phoebe.get_basic_logger()

# load the default phoebe bundle and attach a lc
eb = phoebe.Bundle()

''' must load a test lightcurve because the compute lightcurve function works by
reading the timeseries from there - probably should update it so it can take 
only an array of times '''

lc = np.loadtxt('oc1_200.lc', dtype = 'float', comments = '#', delimiter = 
'\t')
eb.lc_fromarrays(time=lc[:,0], flux=lc[:,1], sigma=lc[:,2])

# set the fillout factor and delta of the OC mesh
FF = 0.5
delta = 0.25

times_oc, flux_oc, c3 = oc_d.oc_lc_bundle(eb,FF=FF,sigma=1.,beta=0.,delta=delta)

#plt.figure()
#plt.plot(times_oc, flux_oc, 'ko-')
#plt.savefig("oc_d_test.png")
#plt.close()

#c3.plot3D(select='teff')
