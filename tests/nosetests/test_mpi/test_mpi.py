"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()


def test_mpi(plot=False):
    b = phoebe.Bundle.default_binary()

    # Two spherical suns
    b.set_value_all('teff', value=5772.)
    b.set_value('sma', component='binary', value=100.)
    b.set_value('period', component='binary', value=81.955)

    b.add_dataset('lc', times=np.linspace(0,100,21))
    b.add_compute('phoebe', compute='phoebe2')

    # set matching atmospheres
    b.set_value_all('atm', 'extern_planckint')

    # turn off limb-darkening:
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    b.run_compute(compute='phoebe2', irrad_method='none', model='phoebe2model')

    #~ assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-3, atol=0.))

    return b

if __name__ == '__main__':
    #~ logger = phoebe.logger(clevel='INFO')

    b = test_mpi(plot=False)
