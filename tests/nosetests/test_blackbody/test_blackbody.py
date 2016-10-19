"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    # Two spherical suns
    b.set_value_all('teff', value=5772.)
    b.set_value('sma', component='binary', value=100.)
    b.set_value('period', component='binary', value=81.955)

    b.add_dataset('lc', times=np.linspace(0,100,21))
    b.add_compute('phoebe', compute='phoebe2')
    b.add_compute('legacy', compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm@phoebe2', 'extern_planckint')
    b.set_value_all('atm@phoebe1', 'blackbody')

    # turn off limb-darkening:
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    print "running phoebe2 model..."
    b.run_compute(compute='phoebe2', reflection_method='none', model='phoebe2model')
    print "running phoebe1 model..."
    b.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')

    phoebe2_val = b.get_value('fluxes@phoebe2model')
    phoebe1_val = b.get_value('fluxes@phoebe1model')

    if plot:
        b.plot(dataset='lc01')
        plt.legend()
        plt.show()

    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-3, atol=0.))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)
