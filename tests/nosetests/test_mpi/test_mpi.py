"""
"""
import os
os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE'

import phoebe
import numpy as np

phoebe.mpi_on(8)

def test_mpi(plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,1001))

    b.run_compute(irrad_method='none', model='phoebe2model')

    if len(b.filter('fluxes@model')):
        print b.get_value('fluxes@model')
	if plot:
            b.plot(show=True)

    return b

if __name__ == '__main__':
    #~ logger = phoebe.logger(clevel='INFO')

    b = test_mpi(plot=True)
