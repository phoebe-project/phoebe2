"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()


def test_mpi(plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=np.linspace(0,1,101))

    b.run_compute(irrad_method='none', model='phoebe2model')

    if len(b.filter('fluxes@model')):
        print b.get_value('fluxes@model')
	if plot:
            b.plot(show=True)

    return b

if __name__ == '__main__':
    #~ logger = phoebe.logger(clevel='INFO')

    b = test_mpi(plot=True)
