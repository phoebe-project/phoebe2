import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def test_binary(plot=False):

    cb = phoebe.Bundle.default_binary(contact_binary=True)
    cb['pot@contact_envelope'] = 3.5
    cb['q'] = 1.0
    cb['teff@primary'] = 5000.
    cb['teff@secondary'] = 5000.
    # cb.set_value_all('incl',90.0)


    cb.add_dataset('lc', times=np.linspace(0,3,50), dataset='lc01')
    cb.add_dataset('rv', time=np.linspace(0,3,50), dataset='rv01')

    cb.add_compute('phoebe', ntriangles=4000, compute='phoebe2', mesh_method='marching')
    cb.add_compute('legacy', compute='phoebe1')

    cb.set_value_all('atm', 'extern_planckint')

    # turn off limb-darkening:
    cb.set_value_all('ld_func_bol', 'logarithmic')
    cb.set_value_all('ld_coeffs_bol', [0.0, 0.0])

    cb.set_value_all('ld_func', 'logarithmic')
    cb.set_value_all('ld_coeffs', [0.0, 0.0])

    cb.set_value_all('rv_grav', False)
    cb.set_value_all('ltte', False)

    print "running phoebe2 model..."
    cb.run_compute(compute='phoebe2', irrad_method='none', model='phoebe2model')
    print "running phoebe1 model..."
    cb.run_compute(compute='phoebe1', refl_num=0, model='phoebe1model')

    phoebe2_val_lc = cb.get_value('fluxes@phoebe2model')
    phoebe1_val_lc = cb.get_value('fluxes@phoebe1model')
    phoebe2_val_rv1 = cb.get_value('rvs@primary@phoebe2model')[2:48]
    phoebe1_val_rv1 = cb.get_value('rvs@primary@phoebe1model')[2:48]
    phoebe2_val_rv2 = cb.get_value('rvs@secondary@phoebe2model')[2:48]
    phoebe1_val_rv2 = cb.get_value('rvs@secondary@phoebe1model')[2:48]

    if plot:
        print "max rtol lc:", np.max((phoebe2_val_lc - phoebe1_val_lc)/phoebe1_val_lc)
        print "max atol rv1:", np.max(phoebe2_val_rv1 - phoebe1_val_rv1)
        print "max atol rv2:", np.max(phoebe2_val_rv2 - phoebe1_val_rv2)

        cb.plot(dataset='lc01')
        plt.legend()
        plt.show()

        cb.plot(dataset='rv01')
        plt.legend()
        plt.show()

    assert(np.allclose(phoebe2_val_lc, phoebe1_val_lc, rtol=7e-3, atol=0.))
    # note the amplitude is about 100, so this is comperable to a relative
    # tolerance of 7e-3 (but we can't do relative because then everything blows
    # up near 0)
    assert(np.allclose(phoebe2_val_rv1, phoebe1_val_rv1, rtol=0., atol=0.7))
    assert(np.allclose(phoebe2_val_rv2, phoebe1_val_rv2, rtol=0., atol=0.7))
    return cb

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    cb = test_binary(plot=True)
