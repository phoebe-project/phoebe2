"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os

def test_binary(plot=False, gen_comp=False):
    b = phoebe.Bundle.default_binary()

    b.add_spot(component='primary', relteff=0.9, radius=20, colat=45, long=90, feature='spot01')

    b.add_dataset('lc', times=np.linspace(0,1,26))
    b.add_dataset('mesh', times=[0], columns=['teffs'])
    b.add_compute('phoebe', compute='phoebe2')
    if gen_comp:
        b.add_compute('legacy', compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm', 'extern_planckint')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'logarithmic')
    b.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.0, 0.0])


    #turn off albedos (legacy requirement)
    b.set_value_all('irrad_frac_refl_bol',  0.0)

    print("running phoebe2 model...")
    b.run_compute(compute='phoebe2', irrad_method='none', model='phoebe2model')
    if gen_comp:
        print("running phoebe1 model...")
        b.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')
        b.filter(model='phoebe1model').save('test_spots.comp.model')
    else:
        b.import_model(os.path.join(os.path.dirname(__file__), 'test_spots.comp.model'), model='phoebe1model')

    phoebe2_val = b.get_value('fluxes@phoebe2model')
    phoebe1_val = b.get_value('fluxes@phoebe1model')

    if plot:
        print("rel: ", ((phoebe2_val-phoebe1_val)/phoebe2_val).max())
        print("abs: ", (phoebe2_val-phoebe1_val).max())

        # b.plot(dataset='mesh01', show=True)

        b.plot(dataset='lc01', legend=True, show=True)

    assert(np.allclose(phoebe2_val, phoebe1_val, rtol=2e-3, atol=5e-4))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='DEBUG')


    b = test_binary(plot=True, gen_comp=True)
