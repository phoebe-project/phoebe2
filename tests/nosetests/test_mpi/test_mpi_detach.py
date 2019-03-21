"""
can also be run with mpirun -np 8 python test_mpi.py
but won't run detached (ie. detach=True will be ignored)
"""

import phoebe
import numpy as np
import sys

def test_mpi(plot=False, npoints=8):
    phoebe.reset_settings()
    phoebe.mpi_on(4)

    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,npoints))

    if plot: print("calling compute")
    b.run_compute(irrad_method='none', ntriangles=1000, detach=True)
    if plot:
        print("attaching to model")
        print(b['model'].status)
    b['model'].attach()

    if plot: print("model received")

    if plot:
        b.plot(show=True)

    phoebe.reset_settings()
    phoebe.mpi_off()

    return b

if sys.version_info[0] >= 3:
    test_mpi.__test__ = False

if __name__ == '__main__':
    # for fair timing comparisons, let's disable checking for online passbands
    import os
    os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE'

    logger = phoebe.logger(clevel='WARNING')

    b = test_mpi(plot=False, npoints=1001)
