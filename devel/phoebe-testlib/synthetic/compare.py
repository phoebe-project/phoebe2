import sys
import numpy as np
import phoebe
from phoebe.io import parsers
from phoebe.parameters import datasets
from matplotlib import pyplot as plt

logger = phoebe.get_basic_logger(clevel=Info)
#logger = phoebe.get_basic_logger(clevel=None,filename='detached_1.log')

if sys.argv[1:]:
    name = sys.argv[1]
else:
    name = "detached_1"

system = parsers.legacy_to_phoebe("{}.phoebe".format(name), create_body=True)
system.set_label("Test-suite: D1")

system.list()

system[0].params['mesh']['gridsize'] = 20
system[1].params['mesh']['gridsize'] = 20

system[0].params['component']['atm'] = 'atmcof.dat'
system[1].params['component']['atm'] = 'atmcof.dat'

# phoebe 1.0 results:
time, flux1 = np.loadtxt("{}.lc.1.0.data".format(name), unpack=True)[:2]

# phoebe 2.0 results:
try:
    flux2 = np.loadtxt("{}.lc.2.0.data".format(name), usecols=(1,), unpack=True)
except:
    refs = system.get_refs(category='lc')

    obs = datasets.LCDataSet(time=time, columns=['time'], ref=refs[0])
    system[0].params['pbdep']['lcdep'][refs[0]]['atm'] = 'atmcof.dat'
    system[1].params['pbdep']['lcdep'][refs[0]]['atm'] = 'atmcof.dat'

    system.add_obs(obs)

    phoebe.compute(system, eclipse_alg='convex')
    lc = system.get_synthetic()

    fout = open("{}.lc.2.0.data".format(name), "w")
    for i in range(len(lc['time'])):
        fout.write("%lf\t%lf\n" % (lc['time'][i], lc['flux'][i]))
    fout.close()

    flux2 = np.asarray(lc['flux'])

diffs = 4*np.pi*flux2-flux1

# Plotting.
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.set_xlim(-0.5,5.75)
ax.set_xlabel('Phase')
ax.set_ylabel('Flux')
plt.plot(time, 4*np.pi*flux2, 'b-')
plt.plot(time, flux1, 'r-')

pd = fig.add_subplot(2, 1, 2)
pd.set_xlim(-0.5,5.75)
pd.set_xlabel('Phase')
pd.set_ylabel('Differences (new-old)')
plt.plot(time, diffs, 'b-')

plt.show()

