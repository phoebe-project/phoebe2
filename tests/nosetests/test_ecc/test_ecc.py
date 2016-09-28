"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()

def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()


    b.add_dataset('lc', times=np.linspace(0,3,101))
    b.add_compute('phoebe', reflection_method='none', compute='phoebe2')
    b.add_compute('legacy', refl_num=0, compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm@phoebe2', 'extern_planckint')
    b.set_value_all('atm@phoebe1', 'blackbody')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    for ecc in [0.3, 0.6, 0.675]:
        b.set_value('ecc', ecc)


        print "running phoebe2 model..."
        b.run_compute(compute='phoebe2', reflection_method='none', model='phoebe2model')
        print "running phoebe1 model..."
        b.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')

        phoebe2_val = b.get_value('fluxes@phoebe2model')
        phoebe1_val = b.get_value('fluxes@phoebe1model')

        if plot:
            print "ecc: {} max (rel): {}".format(ecc , abs((phoebe2_val-phoebe1_val)/phoebe1_val).max())

            b.plot(dataset='lc01')
            plt.legend()
            plt.show()



        # ecc: max(rel)
        # 0.3: 0.00068
        # 0.6: 0.00078
        # 0.65: 0.00125
        # 0.67: 0.00269
        # 0.672: 0.002931
        # 0.675: 0.00349
        # 0.68: overflowing at periastron (via system checks)
        assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-3 if ecc < 0.65 else 5e-3, atol=0.))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)