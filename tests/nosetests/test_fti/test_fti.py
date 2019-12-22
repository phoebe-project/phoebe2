"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os

def test_binary(plot=False):
    dir = os.path.dirname(os.path.realpath(__file__))

    b = phoebe.Bundle.from_legacy(os.path.join(dir, 'kic12004834.phoebe'))
    # this phoebe legacy file uses extern_planckint and with albedos to 0
    # and exptime already defined
    b.set_value_all('atm', kind='phoebe', value='blackbody')
    b.set_value_all('irrad_method', 'none')

    fluxes_legacy = np.loadtxt(os.path.join(dir, 'kic12004834.nofti.data'), unpack=True, usecols=(1,))

    times = np.linspace(55002.04045, 55002.30277, len(fluxes_legacy))
    b.set_value('times', times)

    b.run_compute(kind='phoebe', fti_method='none')
    fluxes = b.get_value('fluxes', context='model')

    if plot:
        print("fti off")
        print(abs(fluxes_legacy-fluxes).max())
        plt.plot(times, fluxes_legacy, 'k-')
        b.plot(show=True)
    assert(np.allclose(fluxes, fluxes_legacy, rtol=0, atol=1e-3))

    b.run_compute(kind='phoebe', fti_method='oversample', fti_oversample=10)
    fluxes_legacy = np.loadtxt(os.path.join(dir, 'kic12004834.fti.data'), unpack=True, usecols=(1,))
    fluxes = b.get_value('fluxes', context='model')

    if plot:
        print("fti on")
        print(abs(fluxes_legacy-fluxes).max())
        plt.plot(times, fluxes_legacy, 'k-')
        b.plot(show=True)
    assert(np.allclose(fluxes, fluxes_legacy, rtol=0, atol=1e-3))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    b = test_binary(plot=True)
