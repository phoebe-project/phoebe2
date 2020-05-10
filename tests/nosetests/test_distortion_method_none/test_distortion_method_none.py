"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):


    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc', times=[0])
    b.add_dataset('lp', times=[0])
    b.add_dataset('rv', times=[0])
    b.add_dataset('mesh', times=[0])

    for comp in ['primary', 'secondary']:
        b.set_value_all('distortion_method', value='roche')
        b.set_value('distortion_method', component=comp, value='none')
        b.compute_pblums(pblum_method='stefan-boltzmann')
        b.compute_pblums()

        b.run_compute()

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)
