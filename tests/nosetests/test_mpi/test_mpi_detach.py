"""
"""

import phoebe
import numpy as np

phoebe.mpi_on(np=8)

def test_mpi(detach=True, plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,101))

    print "calling compute"
    b.run_compute(irrad_method='none', model='phoebe2model', detach=detach)
    if detach:
        print "attaching to model"
        print b['model'].status
        b['model'].attach()
    
    print "model received"

    if plot:
        b.plot(show=True)

    return b

if __name__ == '__main__':
    #~ logger = phoebe.logger(clevel='INFO')

    b = test_mpi(detach=True, plot=True)
    b = test_mpi(detach=False, plot=True)
