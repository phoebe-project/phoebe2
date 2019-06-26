"""
"""

import phoebe
import numpy as np



def test_dataset_scaled(verbose=False, plot=False):
    b = phoebe.Bundle.default_binary()

    times = np.linspace(0,1,11)
    fluxes = np.random.random(11)
    b.add_dataset('lc', times=times, fluxes=fluxes, pblum_mode='dataset-scaled')
    b.add_dataset('lc', times=times, fluxes=fluxes, pblum_mode='dataset-scaled')
    b.add_dataset('rv', times=times)

    b.run_compute(irrad_method='none')

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_dataset_scaled(verbose=True, plot=True)
