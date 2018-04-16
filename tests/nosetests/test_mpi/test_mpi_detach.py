"""
can also be run with mpirun -np 8 python test_mpi.py
but won't run detached (ie. detach=True will be ignored)
"""

# for fair timing comparisons, let's disable checking for online passbands
import os
os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE'

import phoebe
import numpy as np

def test_mpi(plot=False):
    phoebe.reset_settings()
    # causes issue in travis
    # phoebe.mpi_on(np=4)

    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,101))

    if plot: print "calling compute"
    b.run_compute(irrad_method='none', detach=True)
    if not phoebe.backends._within_mpirun:
        if plot:
            print "attaching to model"
            print b['model'].status
        b['model'].attach()

    if plot: print "model received"

    if plot:
        b.plot(show=True)

    phoebe.reset_settings()

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    b = test_mpi(plot=False)
