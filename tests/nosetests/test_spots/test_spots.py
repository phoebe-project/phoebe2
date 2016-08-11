"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_spot(component='primary', relteff=0.8, radius=20, colat=45, colon=90, feature='spot01')

    b.add_dataset('LC', time=np.linspace(0,3,101))
    b.add_compute('phoebe', compute='phoebe2')
    b.add_compute('legacy', compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm@phoebe2', 'extern_planckint')
    b.set_value_all('atm@phoebe1', 'blackbody')

    print "running phoebe2 model..."
    b.run_compute(compute='phoebe2', model='phoebe2model')
    print "running phoebe1 model..."
    b.run_compute(compute='phoebe1', model='phoebe1model')

    if plot:
        b.plot()
        plt.legend()
        plt.show()

if __name__ == '__main__':
    logger = phoebe.logger()


    test_binary(plot=True)