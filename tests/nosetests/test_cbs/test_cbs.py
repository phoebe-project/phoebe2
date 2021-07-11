import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os


def test_binary(plot=False, gen_comp=False):

    cb = phoebe.Bundle.default_binary(contact_binary=True)
    cb.flip_constraint('pot', solve_for='requiv@primary')
    cb['pot@contact_envelope@component'] = 3.5
    cb['q'] = 1.0
    cb['teff@primary'] = 5000.
    cb['teff@secondary'] = 5000.

    # cb.set_value_all('incl',90.0)

    times = cb.to_time(np.linspace(-.1,1.1,100))
    cb.add_dataset('lc', times=times, dataset='lc01')
    cb.add_dataset('rv', time=times, dataset='rv01')

    cb.add_compute('phoebe', ntriangles=4000, compute='phoebe2', mesh_method='marching')
    if gen_comp:
        cb.add_compute('legacy', gridsize=100, compute='phoebe1')

    cb.set_value_all('atm', 'extern_planckint')

    # turn off limb-darkening:
    cb.set_value_all('ld_mode_bol', 'manual')
    cb.set_value_all('ld_func_bol', 'linear')
    cb.set_value_all('ld_coeffs_bol', [0.0])

    cb.set_value_all('ld_mode', 'manual')
    cb.set_value_all('ld_func', 'linear')
    cb.set_value_all('ld_coeffs', [0.0])

    cb.set_value_all('rv_grav', False)
    cb.set_value_all('ltte', False)

    #turn off albedos (legacy requirement)
    cb.set_value_all('irrad_frac_refl_bol',  0.0)

    print("running phoebe2 model...")
    cb.run_compute(compute='phoebe2', irrad_method='none', model='phoebe2model')
    if gen_comp:
        print("running phoebe1 model...")
        cb.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')
        cb.filter(model='phoebe1model').save('test_cbs.comp.model')
    else:
        cb.import_model(os.path.join(os.path.dirname(__file__), 'test_cbs.comp.model'), model='phoebe1model')

    phoebe2_val_lc = cb.get_value('fluxes@phoebe2model')
    phoebe1_val_lc = cb.get_value('fluxes@phoebe1model')
    phoebe2_val_rv1 = cb.get_value('rvs@primary@phoebe2model')[2:48]
    phoebe1_val_rv1 = cb.get_value('rvs@primary@phoebe1model')[2:48]
    phoebe2_val_rv2 = cb.get_value('rvs@secondary@phoebe2model')[2:48]
    phoebe1_val_rv2 = cb.get_value('rvs@secondary@phoebe1model')[2:48]

    if plot:
        print("max lc atol {}: rtol: {}".format(np.max(phoebe2_val_lc - phoebe1_val_lc), np.max((phoebe2_val_lc - phoebe1_val_lc)/phoebe1_val_lc)))
        print("max rv1 atol: {} rtol: {}".format(np.max(phoebe2_val_rv1 - phoebe1_val_rv1), np.max((phoebe2_val_rv1 - phoebe1_val_rv1)/phoebe1_val_rv1)))
        print("max rv2 atol: {} rtol: {}".format(np.max(phoebe2_val_rv2 - phoebe1_val_rv2), np.max((phoebe2_val_rv2 - phoebe1_val_rv2)/phoebe1_val_rv2)))
        cb.plot(dataset='lc01', legend=True, show=True)
        cb.plot(dataset='rv01', legend=True, show=True)

    assert(np.allclose(phoebe2_val_lc, phoebe1_val_lc, rtol=7e-3, atol=0.))
    # note we can't use relative tolerances because those blow up near 0, so
    # instead we'll fake a relative tolerance by using the amplitude of the RV curve.
    rv_ampl = np.max(np.abs(phoebe1_val_rv1))
    rtol = 7e-3
    assert(np.allclose(phoebe2_val_rv1, phoebe1_val_rv1, rtol=0., atol=rtol*rv_ampl))
    assert(np.allclose(phoebe2_val_rv2, phoebe1_val_rv2, rtol=0., atol=rtol*rv_ampl))
    return cb

if __name__ == '__main__':
    logger = phoebe.logger(clevel='debug')

    cb = test_binary(plot=True, gen_comp=True)
