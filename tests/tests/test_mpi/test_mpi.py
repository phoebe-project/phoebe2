"""
can also be run with mpirun -np 8 python test_mpi.py
"""
import phoebe
import numpy as np
import sys


def test_mpi(verbose=False, plot=False, npoints=8, turn_mpi_off_after=True):
    phoebe.reset_settings()
    phoebe.mpi_on(4)

    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,npoints))

    if verbose: print("calling compute")
    b.run_compute(irrad_method='none', model='phoebe2model')
    if verbose: print("model received")

    if plot:
        b.plot(show=True)

    phoebe.reset_settings()
    if turn_mpi_off_after:
        # we need to turn this off for the next test in nosetests... but
        # if running from python or mpirun, we don't want to release all the
        # workers or they'll just pickup all the previous tasks
        phoebe.mpi_off()

    return b

# disable testing within nosetests/Travis
test_mpi.__test__ = False

if __name__ == '__main__':
    # for fair timing comparisons, let's disable checking for online passbands
    import os
    os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE'


    logger = phoebe.logger(clevel='INFO')

    b = test_mpi(verbose=True, plot=False, npoints=101, turn_mpi_off_after=False)
