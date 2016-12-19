"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    period = b.get_value('period@binary')
    b.add_dataset('lc', times=np.linspace(0,period,21))
    b.add_compute('phoebe', irrad_method='none', compute='phoebe2')
    b.add_compute('legacy', refl_num=0, compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm', 'extern_planckint')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])
    # b.set_value_all('ecc', 0.2)

    for dpdt in [-0.5, -0.25, 0.25, 0.5]:
        b.set_value('dpdt', dpdt)


        print "running phoebe2 model..."
        b.run_compute(compute='phoebe2', model='phoebe2model')
        print "running phoebe1 model..."
        b.run_compute(compute='phoebe1', model='phoebe1model')

        phoebe2_val = b.get_value('fluxes@phoebe2model')
        phoebe1_val = b.get_value('fluxes@phoebe1model')

        if plot:
            b.plot(dataset='lc01')
            plt.legend()
            plt.show()

            print "max (rel):", abs((phoebe2_val-phoebe1_val)/phoebe1_val).max()

        assert(np.allclose(phoebe2_val, phoebe1_val, rtol=5e-3, atol=0.))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)
