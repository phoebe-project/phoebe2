"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=phoebe.linspace(0,1,21))

    b.set_value('Av', 0.2)
    b.flip_constraint('ebv', solve_for='Av')
    b.set_value('ebv', 0.25)

    # enable and implement assertions once extinction tables are available
    # b.run_compute(irrad_method='none')

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)
