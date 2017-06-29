"""
can also be run with mpirun -np 8 python test_mpi.py
but won't run detached (ie. detach=True will be ignored)
"""

# for fair timing comparisons, let's disable checking for online passbands
import os
os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE'

import phoebe
import numpy as np

phoebe.mpi_on(np=8)

def test_mpi(plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,1001))

    print "calling compute"
    b.run_compute(irrad_method='none', detach=True)
    if not phoebe.backends._use_mpi:
        print "attaching to model"
        print b['model'].status
        b['model'].attach()

    print "model received"

    if plot:
        b.plot(show=True)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    b = test_mpi(plot=False)
