"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    b.set_value('sma', component='binary', value=3.0)
    # b.set_value('teff', component='primary', value=6000)
    # b.set_value('teff', component='secondary', value=8000)
    b.set_value('rpole', component='primary', value=0.5)
    b.set_value('rpole', component='secondary', value=0.5)

    b.add_dataset('LC', time=np.linspace(0,3,101))
    b.add_compute('phoebe', compute='phoebe2', refl=True)
    b.add_compute('legacy', compute='phoebe1', mult_refl=True)

    # set matching atmospheres
    b.set_value_all('atm@phoebe2', 'extern_planckint')
    b.set_value_all('atm@phoebe1', 'blackbody')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    for alb in np.linspace(0, 1.0, 4):
        print "alb = {}".format(alb)
        b.set_value_all('alb_refl_bol', alb)

        print "running phoebe2 model..."
        b.run_compute(compute='phoebe2', delta=0.1, model='phoebe2model')
        print "running phoebe1 model..."
        b.run_compute(compute='phoebe1', gridsize=30, model='phoebe1model')

        if plot:
            b.plot()
            plt.legend()
            plt.title("alb = {}".format(alb))
            plt.show()

        phoebe2_val = b.get_value('flux@phoebe2model')
        phoebe1_val = b.get_value('flux@phoebe1model')
        assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-3, atol=0.))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    b = test_binary(plot=True)