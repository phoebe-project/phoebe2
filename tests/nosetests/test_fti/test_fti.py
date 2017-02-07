"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os

phoebe.devel_on()
phoebe.interactive_off()

def test_binary(plot=False):
    every = 2

    dir = os.path.dirname(os.path.realpath(__file__))


    b = phoebe.Bundle.from_legacy(os.path.join(dir, 'kic12004834.phoebe'))
    b.set_value_all('atm', 'blackbody')
    b.set_value('irrad_method', 'none')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])

    # b.set_value_all('passband', 'Johnson:V')

    period = b.get_value('period', kind='orbit', context='component')
    times = b.to_time(np.linspace(-0.6, 0.6, 100))[::every]
    b.set_value('times', times)

    # return b

    b.run_compute(fti_method='none')
    fluxes_legacy = np.loadtxt(os.path.join(dir, 'kic12004834.nofti.data'), unpack=True, usecols=(1,))[::every]
    fluxes = b.get_value('fluxes', context='model')

    if plot:
        print "fti off"
        print abs(fluxes_legacy-fluxes).max()
        plt.plot(times, fluxes_legacy)
        b.plot(show=True)
    # assert(np.allclose(fluxes, fluxes_legacy, rtol=0, atol=1e-3))


    b.run_compute(fti_method='oversample', fti_oversample=5)
    fluxes_legacy = np.loadtxt(os.path.join(dir, 'kic12004834.fti.data'), unpack=True, usecols=(1,))[::every]
    fluxes = b.get_value('fluxes', context='model')

    if plot:
        print "fti on"
        print abs(fluxes_legacy-fluxes).max()
        plt.plot(times, fluxes_legacy)
        b.plot(show=True)
    # assert(np.allclose(fluxes, fluxes_legacy, rtol=0, atol=1e-3))





    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_binary(plot=True)