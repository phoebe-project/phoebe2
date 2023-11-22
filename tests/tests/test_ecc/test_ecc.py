"""
"""

import phoebe
import numpy as np
import os

def test_binary(plot=False, gen_comp=False):
    b = phoebe.default_binary()

    period = b.get_value('period@binary')
    b.add_dataset('lc', times=np.linspace(0,period,21))
    b.add_compute('phoebe', irrad_method='none', compute='phoebe2')
    if gen_comp:
        b.add_compute('legacy', refl_num=0, compute='phoebe1')

    # set matching atmospheres
    b.set_value_all('atm', 'extern_planckint')

    # set matching limb-darkening, both bolometric and passband
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'linear')
    b.set_value_all('ld_coeffs_bol', [0.0])

    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.0])

    #turn off albedos (legacy requirement)
    b.set_value_all('irrad_frac_refl_bol',  0.0)


    # 0.51 starts to overlap
    for ecc in [0.3, 0.505]:
        b.set_value('ecc', ecc)

        for per0 in [45, 90, 90+45]:
            b.set_value('per0', per0)

            print("running phoebe2 model...")
            b.run_compute(compute='phoebe2', irrad_method='none', model='phoebe2model', overwrite=True)
            if gen_comp:
                print("running phoebe1 model...")
                b.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model', overwrite=True)
                b.filter(model='phoebe1model').save('test_ecc_{}_{}.comp.model'.format(ecc, per0))
            else:
                b.import_model(os.path.join(os.path.dirname(__file__), 'test_ecc_{}_{}.comp.model'.format(ecc, per0)), model='phoebe1model', overwrite=True)

            phoebe2_val = b.get_value('fluxes@phoebe2model')
            phoebe1_val = b.get_value('fluxes@phoebe1model')

            if plot:
                print("ecc: {} per0:: {} max (rel): {}".format(ecc , per0, abs((phoebe2_val-phoebe1_val)/phoebe1_val).max()))

                b.plot(dataset='lc01', show=True)

            assert(np.allclose(phoebe2_val, phoebe1_val, rtol=1e-3 if ecc < 0.5 else 5e-3, atol=0.))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='DEBUG')


    b = test_binary(plot=True, gen_comp=True)
